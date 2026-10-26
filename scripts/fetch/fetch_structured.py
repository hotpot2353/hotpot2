# 脚本 B：获取行情与财务结构化数据，落盘 data/raw/structured/
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    PROJECT_ROOT, RateLimitedSession, load_config, load_env,
    read_stock_list, setup_logging, tencent_symbol,
)


def parse_tencent_quote(text: str) -> dict:
    # 腾讯行情返回 v_sh600519="1~名称~代码~现价~..."; 按 ~ 切分后取关键位
    fields = text.split('"')[1].split("~")

    def at(idx: int) -> str:
        return fields[idx] if idx < len(fields) else ""

    return {
        "name": at(1),
        "code": at(2),
        "price": at(3),
        "change_pct": at(32),
        "volume": at(36),
        "amount_wan": at(37),
        "turnover_rate": at(38),
        "pe_ttm": at(39),
        "circ_mv_yi": at(44),
        "total_mv_yi": at(45),
        "pb": at(46),
    }


def fetch_quotes(cfg: dict, codes: list, session, log) -> list:
    # 单只失败不影响其他股票；返回失败代码列表
    url = cfg["datasource"]["tencent_quote_url"]
    rows, failed = [], []
    for code in codes:
        try:
            resp = session.get(f"{url}={tencent_symbol(code)}")
            resp.encoding = "gbk"
            if '="' not in resp.text:
                raise ValueError("返回内容不是行情串")
            rows.append(parse_tencent_quote(resp.text))
            log.info("行情快照已获取 %s", code)
        except Exception as exc:
            log.error("行情快照失败 %s: %s", code, exc)
            failed.append(code)
    out = PROJECT_ROOT / cfg["paths"]["raw_quotes"]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "snapshot.csv", index=False, encoding="utf-8-sig")
    log.info("行情快照写入 snapshot.csv（成功 %s，失败 %s）", len(rows), len(failed))
    return failed


def fetch_history(cfg: dict, codes: list, log) -> None:
    # 前复权日线，主源 baostock（直连稳定；adjustflag=2 前复权）
    # 输出与入库层一致的英文蛇形列；volume 由股转手（/100）
    # 涨跌额、振幅基于前收盘价计算；全市场统一 baostock 口径
    import baostock as bs

    start = f"{min(cfg['datasource']['years'])}-01-01"
    out = PROJECT_ROOT / cfg["paths"]["raw_quotes"]
    out.mkdir(parents=True, exist_ok=True)

    def bs_symbol(code: str) -> str:
        return ("sh." if code[0] in ("6", "9") else "sz.") + code

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {login.error_msg}")

    failed = []
    try:
        for code in codes:
            rs = bs.query_history_k_data_plus(
                bs_symbol(code),
                "date,open,high,low,close,volume,amount,turn,pctChg",
                start_date=start,
                end_date=pd.Timestamp.today().strftime("%Y-%m-%d"),
                frequency="d",
                adjustflag="2",
            )
            if rs.error_code != "0":
                log.error("历史行情查询失败 %s: %s", code, rs.error_msg)
                failed.append(code)
                continue
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                log.warning("历史行情为空 %s", code)
                failed.append(code)
                continue

            df = pd.DataFrame(rows, columns=[
                "trade_date", "open", "high", "low", "close",
                "volume_raw", "amount", "turnover_rate", "change_pct",
            ])
            for col in ["open", "high", "low", "close", "amount",
                        "turnover_rate", "change_pct"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            # baostock volume 单位为股，转手
            df["volume"] = (pd.to_numeric(df["volume_raw"], errors="coerce")
                            / 100).round().astype("Int64")
            # 基于前收盘价补涨跌额与振幅
            prev_close = df["close"].shift(1)
            df["change_amount"] = (df["close"] - prev_close).round(2)
            df["amplitude_pct"] = (
                (df["high"] - df["low"]) / prev_close * 100
            ).round(3)
            df.insert(0, "stock_code", code)
            df = df[[
                "stock_code", "trade_date", "open", "close", "high", "low",
                "volume", "amount", "amplitude_pct", "change_pct",
                "change_amount", "turnover_rate",
            ]]
            df.to_csv(out / f"{code}_daily.csv",
                      index=False, encoding="utf-8-sig")
            log.info("历史行情 %s_daily.csv (%s 行, baostock)", code, len(df))
    finally:
        bs.logout()

    if failed:
        log.warning("以下股票日线获取失败，可重跑补齐: %s", ",".join(failed))
    return failed


def fetch_financials(cfg: dict, codes: list, log) -> list:
    # 单只失败重试 3 次后跳过；返回失败代码列表
    import time

    import akshare as ak

    out = PROJECT_ROOT / cfg["paths"]["raw_financials"]
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for code in codes:
        for attempt in range(1, 4):
            try:
                df = ak.stock_financial_abstract(symbol=code)
                df.to_csv(out / f"{code}_financials.csv",
                          index=False, encoding="utf-8-sig")
                log.info("财务指标 %s_financials.csv (%s 行)", code, len(df))
                break
            except Exception as exc:
                if attempt == 3:
                    log.error("财务指标失败 %s: %s", code, exc)
                    failed.append(code)
                else:
                    time.sleep(2 ** attempt)
        time.sleep(0.3)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", nargs="*")
    parser.add_argument("--skip-history", action="store_true")
    parser.add_argument("--history-only", action="store_true",
                        help="只拉日线，跳过快照与财务")
    args = parser.parse_args()

    load_env()
    cfg = load_config()
    log = setup_logging()
    ds = cfg["datasource"]
    codes = args.codes or read_stock_list(cfg["paths"]["stock_list"])

    if args.history_only:
        fetch_history(cfg, codes, log)
        return

    session = RateLimitedSession(
        ds["request_interval_sec"], ds["max_retries"], ds["timeout"]
    )

    failed_quotes = fetch_quotes(cfg, codes, session, log)
    failed_fin = fetch_financials(cfg, codes, log)
    failed_hist = []
    if not args.skip_history:
        # fetch_history 内部自行记录，返回值对齐另两类
        failed_hist = fetch_history(cfg, codes, log)

    all_failed = sorted(set(failed_quotes) | set(failed_fin) | set(failed_hist))
    if all_failed:
        log.warning("采集失败汇总（%s 只）: %s", len(all_failed), ",".join(all_failed))


if __name__ == "__main__":
    main()
