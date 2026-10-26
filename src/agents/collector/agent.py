# 数据采集Agent：池校验 + 结构化财务走库 + 行情/舆情增量拉取
# 显式降级设计：池外标的财务缺失必须声明，禁止静默、禁止 LLM 兜底
# 输出契约：financials/quotes/news/missing
# 数据边界：财务走 fin_financials（池内独有）；估值 pe/pb 走 fin_quote_snapshot；
#   行情走 库内历史 + akshare 增量合并；新闻走 东财个股页资讯接口 requests 直连
import datetime

import psycopg
import requests

from agents.collector.pool import contains as _in_pool
from agents.collector.pool import suggest as _pool_suggest
from agents.orchestrator.state import AgentState
from core.logger import get_logger
from db.connection import connect

logger = get_logger(__name__)

# 库内中文指标 → 规范指标键：子串匹配，同键多命中取指标名最短者（最接近规范口径）
_METRIC_ALIASES = {
    "gross_margin": ("毛利率",),
    "net_margin": ("净利率",),
    "roe": ("净资产收益率", "ROE"),
    "cash_to_net_profit": ("经营活动净现金/归属母公司",),
    "receivable_turnover": ("应收账款周转",),
    "inventory_turnover": ("存货周转",),
}

# 指标池锁死 8 项：财务 6 项走 fin_financials，估值 2 项走快照表
_FINANCIAL_INDICATORS = ("gross_margin", "net_margin", "roe", "cash_to_net_profit",
                         "receivable_turnover", "inventory_turnover")

# 行情合并后保留的最近交易日数
_QUOTES_WINDOW = 120


def run(state: AgentState) -> dict:
    # 采集入口：按意图拉数据，任何数据源失败都降级为 missing 留痕，不阻断链路
    code = state.intent["stock_code"]
    name = state.intent.get("stock_name", "")
    years = state.intent["years"]
    dims = state.intent.get("dimensions", [])

    raw = {
        "stock_code": code,
        "financials": {},
        "quotes": {"dates": [], "close": [], "volume": []},
        "news": [],
        "missing": [],
    }

    if not _in_pool(code):
        _handle_out_of_pool(state, code, name, dims, raw)
    else:
        raw["financials"] = _fetch_financials(code, years, raw["missing"])
        _fetch_valuation(code, raw["financials"], raw["missing"])

    raw["quotes"] = _fetch_quotes(code, raw["missing"])
    raw["news"] = _fetch_news(code, raw["missing"])
    logger.info("[collector] %s 财务 %d 项 行情 %d 条 新闻 %d 条 missing=%s",
                code, len(raw["financials"]), len(raw["quotes"]["dates"]),
                len(raw["news"]), raw["missing"])
    return raw


def _handle_out_of_pool(state: AgentState, code: str, name: str,
                        dims: list, raw: dict) -> None:
    # 池外显式降级：可支撑维度只剩 technical/sentiment；全不可支撑则整单拒绝
    if not {"technical", "sentiment"} & set(dims):
        tip = _pool_suggest(name)
        msg = f"标的 {code} {name} 不在年报覆盖池，所选维度全部依赖池内数据，无法研判"
        if tip:
            msg += f"；您是否想问池内标的：{tip}"
        raise ValueError(msg)
    # 照常跑可支撑维度：财务标记缺失 + errors 留痕（研报阶段据此写覆盖声明）
    raw["missing"].append("financials:out_of_pool")
    state.errors.append({"type": "out_of_pool", "stock_code": code,
                         "detail": "结构化财务与年报RAG不可用，仅支撑技术面/舆情"})


def _fetch_financials(code: str, years: list, missing: list) -> dict:
    # 长表转宽表：只取年报期（12-31），再按别名表映射成规范键
    # 年报口径原因：规则引擎方向判定假设"年度序列相邻两期即同比"，
    # 季度累计值逐季递增是累计的数学性质，混入会系统性扭曲方向判定
    # 年份范围前扩一年：分析年需要上一年年报作同比基期
    lo, hi = min(years) - 1, max(years)
    out = {}
    try:
        with connect() as cn, cn.cursor() as cur:
            cur.execute(
                "SELECT metric, report_date, value FROM fin_financials "
                "WHERE stock_code = %s AND EXTRACT(year FROM report_date)::int BETWEEN %s AND %s "
                "ORDER BY report_date",
                (code, lo, hi),
            )
            rows = cur.fetchall()
    except psycopg.Error as exc:
        missing.append("financials:db_error")
        logger.warning("财务查询失败 %s: %s", code, exc)
        return out

    # 按指标归组：metric -> [(report_date, value), ...]，仅保留年报期
    grouped = {}
    for metric, report_date, value in rows:
        if report_date.month != 12 or report_date.day != 31:
            continue
        grouped.setdefault(metric, []).append((report_date, value))

    # 别名映射：同键多命中取指标名最短者；未命中的规范键留 missing（映射缺口可观测）
    for canonical in _FINANCIAL_INDICATORS:
        aliases = _METRIC_ALIASES[canonical]
        candidates = [m for m in grouped if any(a.lower() in m.lower() for a in aliases)]
        if not candidates:
            missing.append(f"financials:{canonical}:unmapped_or_empty")
            continue
        metric = min(candidates, key=len)
        # value 为 NULL 表示该期未披露：剔除 None 保序列可被数值分析直接消费
        vals = [float(v) for _, v in grouped[metric] if v is not None]
        if vals:
            out[canonical] = vals
    return out


def _fetch_valuation(code: str, financials: dict, missing: list) -> None:
    # 估值两项取快照表最新一行，序列仅 1 期（查询时点值，非时序指标）
    try:
        with connect() as cn, cn.cursor() as cur:
            cur.execute("SELECT pe_ttm, pb FROM fin_quote_snapshot WHERE stock_code = %s",
                        (code,))
            row = cur.fetchone()
    except psycopg.Error as exc:
        missing.append("financials:valuation_db_error")
        logger.warning("估值查询失败 %s: %s", code, exc)
        return
    if row is None:
        missing.append("financials:valuation:empty")
        return
    pe, pb = row
    if pe is not None:
        financials["pe"] = [float(pe)]
    else:
        missing.append("financials:pe:empty")
    if pb is not None:
        financials["pb"] = [float(pb)]
    else:
        missing.append("financials:pb:empty")


def _fetch_quotes(code: str, missing: list) -> dict:
    # 增量策略：库内最近 120 交易日 + akshare 缺口合并（按日期去重，新覆盖旧）
    db_rows = _quotes_from_db(code, missing)
    fresh_rows = _quotes_from_akshare(code, missing)
    merged = {d: (c, v) for d, c, v in db_rows}
    for d, c, v in fresh_rows:
        merged[d] = (c, v)
    dates = sorted(merged)[-_QUOTES_WINDOW:]
    # 极值只保留不剔除（项目铁律：涨跌停/放量长阳是技术面关键信号）
    return {
        "dates": dates,
        "close": [merged[d][0] for d in dates],
        "volume": [merged[d][1] for d in dates],
    }


def _quotes_from_db(code: str, missing: list) -> list:
    # 库内历史日线；池外标的库中无数据，自然返回空列表
    try:
        with connect() as cn, cn.cursor() as cur:
            cur.execute(
                "SELECT trade_date, close, volume FROM fin_quotes "
                "WHERE stock_code = %s ORDER BY trade_date DESC LIMIT %s",
                (code, _QUOTES_WINDOW),
            )
            return [(d.isoformat(), float(c), int(v)) for d, c, v in cur.fetchall()]
    except psycopg.Error as exc:
        missing.append("quotes:db_error")
        logger.warning("行情查询失败 %s: %s", code, exc)
        return []


def _quotes_from_akshare(code: str, missing: list) -> list:
    # 前复权日线拉近 200 自然日；外部数据源异常一律降级不阻断（边界处允许宽捕获）
    try:
        import akshare as ak
        end = datetime.date.today()
        start = end - datetime.timedelta(days=200)
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=start.strftime("%Y%m%d"),
                                end_date=end.strftime("%Y%m%d"), adjust="qfq")
    except Exception as exc:
        missing.append("quotes:akshare_error")
        logger.warning("akshare 行情失败 %s: %s", code, exc)
        return []
    if df is None or df.empty:
        missing.append("quotes:akshare_empty")
        return []
    return [(str(r["日期"]), float(r["收盘"]), int(r["成交量"])) for _, r in df.iterrows()]


def _fetch_news(code: str, missing: list) -> list:
    # 东财个股页资讯列表接口 requests 直连，取标题 Top20
    # 换源原因：akshare stock_news_em 新版内嵌 V8 跑 JS 生成加密参数（\u 报错即 V8 正则语法错），
    # 旧版明文参数调用已 400；本接口即个股页真实资讯流，直接 JSON 无需剥壳
    # mTypeAndCode 前缀：沪市 1（6/9 开头），深市 0；返回结构 data.list[].Art_Title
    mt = "1" if code.startswith(("6", "9")) else "0"
    try:
        resp = requests.get(
            "https://np-listapi.eastmoney.com/comm/web/getListInfo",
            params={"cfh": 1, "client": "web", "mTypeAndCode": f"{mt}.{code}",
                    "type": 1, "pageSize": 20, "pageIndex": 1},
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"},
        )
        resp.raise_for_status()
        items = (resp.json().get("data") or {}).get("list") or []
    except Exception as exc:
        missing.append("news:em_error")
        logger.warning("东财新闻失败 %s: %s", code, exc)
        return []
    titles = [str(i.get("Art_Title", "")).strip() for i in items]
    titles = [t for t in titles if t][:20]
    if not titles:
        missing.append("news:em_empty")
    return titles
