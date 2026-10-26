# 结构化数据清洗归一 + 入库
# - 日线：中文列名 -> 英文蛇形命名，类型规范化，UPSERT 进 fin_quotes
# - 财务：宽表 melt 为 股票 x 报告期 x 指标 的长表，UPSERT 进 fin_financials
# - 快照：fin_quote_snapshot 整表覆盖式 UPSERT
# 清洗产物同时落 data/processed/，作为"归一化中间态"
import argparse
import sys
from pathlib import Path

import pandas as pd
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fetch"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from common import PROJECT_ROOT, load_config, load_env, setup_logging  # noqa: E402
from db.connection import connect  # noqa: E402

# 日线中文列 -> 表字段
DAILY_COLUMN_MAP = {
    "日期": "trade_date",
    "股票代码": "stock_code",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude_pct",
    "涨跌幅": "change_pct",
    "涨跌额": "change_amount",
    "换手率": "turnover_rate",
}


def normalize_daily(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"股票代码": str, "stock_code": str})
    # 兼容 akshare 中文列与 baostock 英文列两种输出
    if "股票代码" in df.columns:
        df = df.rename(columns=DAILY_COLUMN_MAP)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    numeric_cols = ["open", "close", "high", "low", "amount",
                    "amplitude_pct", "change_pct", "change_amount",
                    "turnover_rate"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")
    return df[[
        "stock_code", "trade_date", "open", "close", "high", "low",
        "volume", "amount", "amplitude_pct", "change_pct",
        "change_amount", "turnover_rate",
    ]]


def normalize_financials(csv_path: Path, stock_code: str) -> pd.DataFrame:
    # 前两列为分类、指标，其余为 YYYYMMDD 报告期列
    df = pd.read_csv(csv_path, dtype=str)
    df = df.rename(columns={"选项": "category", "指标": "metric"})
    date_cols = [c for c in df.columns if c.isdigit() and len(c) == 8]
    long = df.melt(
        id_vars=["category", "metric"],
        value_vars=date_cols,
        var_name="report_date",
        value_name="value",
    )
    long["stock_code"] = stock_code
    long["report_date"] = pd.to_datetime(long["report_date"], format="%Y%m%d").dt.date
    # akshare 用空串表示无数据；数值去掉千分位后转 numeric
    long["value"] = (
        long["value"].astype(str).str.replace(",", "", regex=False)
        .replace({"": None, "nan": None, "None": None})
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    # 全期无披露的指标行无入库价值
    long = long.dropna(subset=["value"])
    # 约 10 个指标（净利润/毛利率等）在多个分类中重复且数值完全一致，
    # 保留首次出现的分类（源文件中即"常用指标"），避免主键冲突后分类被覆盖
    long = long.drop_duplicates(
        subset=["stock_code", "report_date", "metric"], keep="first"
    )
    return long[["stock_code", "report_date", "category", "metric", "value"]]


def normalize_snapshot(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"code": str})
    df = df.rename(columns={
        "code": "stock_code",
        "name": "name",
        "price": "price",
        "change_pct": "change_pct",
        "volume": "volume",
        "amount_wan": "amount_wan",
        "turnover_rate": "turnover_rate",
        "pe_ttm": "pe_ttm",
        "circ_mv_yi": "circ_mv_yi",
        "total_mv_yi": "total_mv_yi",
        "pb": "pb",
    })
    df["stock_code"] = df["stock_code"].str.zfill(6)
    for col in ["price", "change_pct", "amount_wan", "turnover_rate",
                "pe_ttm", "circ_mv_yi", "total_mv_yi", "pb"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")
    return df


def upsert(conn, table: str, df: pd.DataFrame, conflict_cols: tuple,
           update_cols: list) -> int:
    # psycopg3 的 executemany 自动走批处理，配合 ON CONFLICT 实现幂等 UPSERT
    cols = list(df.columns)
    placeholders = "(" + ", ".join("%s" for _ in cols) + ")"
    sql = (
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES {placeholders} "
        f"ON CONFLICT ({', '.join(conflict_cols)}) DO UPDATE SET "
        + ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
    )
    rows = [tuple(None if pd.isna(v) else v for v in row)
            for row in df.itertuples(index=False, name=None)]
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", nargs="*", help="只入库指定股票")
    args = parser.parse_args()

    load_env()
    cfg = load_config()
    log = setup_logging()

    raw_quotes = PROJECT_ROOT / cfg["paths"]["raw_quotes"]
    raw_fin = PROJECT_ROOT / cfg["paths"]["raw_financials"]
    proc_quotes = PROJECT_ROOT / cfg["paths"]["processed_quotes"]
    proc_fin = PROJECT_ROOT / cfg["paths"]["processed_financials"]
    proc_quotes.mkdir(parents=True, exist_ok=True)
    proc_fin.mkdir(parents=True, exist_ok=True)

    codes = args.codes or sorted(
        p.stem.split("_")[0]
        for p in raw_fin.glob("*_financials.csv")
    )

    with connect() as conn:
        # 日线
        daily_frames = []
        for code in codes:
            csv_path = raw_quotes / f"{code}_daily.csv"
            if not csv_path.exists():
                log.warning("日线缺失，跳过 %s", code)
                continue
            daily_frames.append(normalize_daily(csv_path))
        if daily_frames:
            daily = pd.concat(daily_frames, ignore_index=True)
            daily.to_csv(proc_quotes / "quotes_normalized.csv",
                         index=False, encoding="utf-8-sig")
            n = upsert(
                conn, "fin_quotes", daily,
                conflict_cols=("stock_code", "trade_date"),
                update_cols=["open", "close", "high", "low", "volume",
                             "amount", "amplitude_pct", "change_pct",
                             "change_amount", "turnover_rate"],
            )
            log.info("fin_quotes UPSERT %s 行", n)

        # 财务长表
        fin_frames = []
        for code in codes:
            csv_path = raw_fin / f"{code}_financials.csv"
            if not csv_path.exists():
                log.warning("财务文件缺失，跳过 %s", code)
                continue
            fin_frames.append(normalize_financials(csv_path, code))
        if fin_frames:
            financials = pd.concat(fin_frames, ignore_index=True)
            financials.to_csv(proc_fin / "financials_long.csv",
                              index=False, encoding="utf-8-sig")
            n = upsert(
                conn, "fin_financials", financials,
                conflict_cols=("stock_code", "report_date", "metric"),
                update_cols=["category", "value"],
            )
            log.info("fin_financials UPSERT %s 行", n)

        # 实时快照
        snap_path = raw_quotes / "snapshot.csv"
        if snap_path.exists():
            snap = normalize_snapshot(snap_path)
            n = upsert(
                conn, "fin_quote_snapshot", snap,
                conflict_cols=("stock_code",),
                update_cols=["name", "price", "change_pct", "volume",
                             "amount_wan", "turnover_rate", "pe_ttm",
                             "circ_mv_yi", "total_mv_yi", "pb", "fetched_at"],
            )
            log.info("fin_quote_snapshot UPSERT %s 行", n)

    log.info("入库完成，归一化产物在 data/processed/")


if __name__ == "__main__":
    main()
