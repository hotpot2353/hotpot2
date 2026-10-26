# 基本面Agent：规则引擎组合研判（零 LLM）+ LLM 叙述要点提炼 + RAG 证据增强
# RAG：命中年报子块 → 定向摘要 → 叙述参考证据 + evidence_ref 溯源；
#   RAG 是增强不是依赖，检索/摘要任一失败降级为无证据叙述，链路不中断
# LLM 边界：规则判定全部由 RuleEngine 完成，LLM 只把快照数字组织成叙述，
#   指标组合的 含义/风险警告 直接取 yaml 原文，LLM 不改写规则结论
# 数据口径：财务序列由 collector 按年报期（12-31）提供，相邻两期即同比，与规则引擎假设一致
# 已知边界：pe/pb 为查询时点单期值，不参与方向判定，涉及其方向/绝对值条件的规则
#   （7/8/9/15/18/19）暂不命中，合成历史估值后放开
# evidence_ref 契约：数组每项含 chunk_id/stock_code/year/section_path，
#   填命中年报子块引用；风控/报告凭 chunk_id 懒加载回链原文（Small-to-Big）
import json

import requests

from agents.orchestrator.state import AgentState
from core.config import get_settings
from core.logger import get_logger
from core.prompt_loader import load_prompt
from core.trace import record_retrieval, record_tokens
from llm.chat_client import get_chat_client
from rag.compressor import compress, format_chunks
from rag.retriever import HybridRetriever
from rules.engine import RuleEngine

logger = get_logger(__name__)

# 指标中文名表：进提示词供 LLM 引用，LLM 无需猜英文键术语
_IND_NAMES = {
    "gross_margin": "毛利率",
    "net_margin": "净利率",
    "roe": "ROE",
    "cash_to_net_profit": "净现比",
    "receivable_turnover": "应收账款周转率",
    "inventory_turnover": "存货周转率",
    "pe": "PE",
    "pb": "PB",
}

# 引擎方向词 → 中文
_DIR_NAMES = {"up": "上行", "down": "下行", "stable": "稳定"}


def run(state: AgentState) -> dict:
    # 基本面分析入口：引擎判定 → LLM 叙述 → 契约输出
    fin = state.raw_data.get("financials") or {}
    missing = state.raw_data.get("missing") or []

    # 财务不可用（池外标记或采集失败）：结构化降级不调 LLM，缺数据不硬分析
    if not fin:
        logger.warning("[fundamental] 财务数据不可用 missing=%s 降级输出", missing)
        return _empty("财务数据不可用，本报告不含基本面分析")

    # 只有 ≥2 期序列可做同比方向判定；pe/pb 单期值被排除出规则研判
    series = {k: v for k, v in fin.items() if len(v) >= 2}
    skipped = sorted(set(fin) - set(series))
    if skipped:
        logger.info("[fundamental] 单期指标不参与方向判定: %s", skipped)
    if not series:
        logger.warning("[fundamental] 财务指标无时序无法判方向 missing=%s", missing)
        return _empty("财务指标缺少时序数据，无法进行组合研判")

    engine = RuleEngine()
    snapshot = engine.classify(series)
    hits = engine.match(snapshot, top_n=2)

    # RAG 证据检索：命中年报子块 → trace 记账 → 定向摘要 → evidence_ref
    # RAG 是增强不是依赖：检索/摘要任一失败降级为无证据叙述，链路不中断
    intent = state.intent
    code = str(intent.get("stock_code", ""))
    years = intent.get("years") or []
    evidence, refs, results = None, [], []
    try:
        query = _retrieval_query(snapshot)
        results = HybridRetriever().retrieve(
            query, stock_code=code, year=years[0] if years else None)
        record_retrieval(state, "fundamental", query, results)
        evidence, refs = compress(state, results)
    except Exception as exc:
        logger.warning("[fundamental] 证据检索失败 降级无证据叙述: %s", exc)
        state.errors.append({"type": "fundamental_rag_error", "detail": str(exc)[:200]})

    # 证据文案三态：摘要优先 / 压缩开关关闭直传原文 / 失败显式声明（缺席不等于没有检索）
    if evidence:
        evidence_text = "证据摘要：\n" + evidence
    elif refs:
        evidence_text = "检索命中年报原文：\n" + format_chunks(results)
    else:
        evidence_text = "无检索证据，本次仅基于结构化数据研判"

    # LLM 叙述层：解析失败不炸链路，降级为规则结论兜底并留 errors 记录
    profit_text, positives, negatives = _narrate(state, snapshot, hits, evidence_text)

    logger.info("[fundamental] 组合规则命中 %d 条 证据 %d 块 方向快照=%s",
                len(hits), len(refs), {k: v["direction"] for k, v in snapshot.items()})
    return {
        "盈利分析": profit_text,
        "指标组合": [
            {"id": rule["id"], "含义": rule["meaning"], "风险警告": rule["risk_warning"]}
            for rule, _n in hits
        ],
        # 方向快照透传给风控第2层异动校验复用（latest/values 引擎已算，risk 不重复 classify）
        "方向快照": {
            k: {"direction": s["direction"], "magnitude": s["magnitude"],
                "persist": s["persist"], "latest": s["latest"], "prev": s["values"][-2]}
            for k, s in snapshot.items()
        },
        "利好要点": positives,
        "利空要点": negatives,
        # 摘要失败/检索失败时 refs=[]：结论没消费证据就不留溯源标记（compressor 降级语义）
        "evidence_ref": refs,
    }


def _empty(reason: str) -> dict:
    # 降级输出：契约形状恒定，内容空但附边界说明，报告层据此写局限性声明
    return {
        "盈利分析": reason,
        "指标组合": [],
        "方向快照": {},
        "利好要点": [],
        "利空要点": [],
        "evidence_ref": [],
    }


def _snapshot_lines(snapshot: dict) -> list:
    # 指标快照人话化：每个指标一行，数字全部来自引擎，LLM 只负责组织语言
    lines = []
    for ind, snap in snapshot.items():
        name = _IND_NAMES.get(ind, ind)
        desc = f"{name} 最新 {snap['latest']} 方向{_DIR_NAMES[snap['direction']]}"
        if snap["persist"] >= 2:
            desc += f" 连续{snap['persist']}期"
        if snap["magnitude"] == "large":
            desc += " 大幅变动"
        desc += f" 序列 {snap['values']}"
        lines.append(desc)
    return lines


def _retrieval_query(snapshot: dict) -> str:
    # 检索 query 构造：固定盈利主题词 + 方向快照指标中文名
    # snapshot 只含 ≥2 期序列指标，单期 pe/pb 天然不在其中，query 无噪声词
    names = " ".join(_IND_NAMES.get(k, k) for k in snapshot)
    return f"盈利能力 经营分析 现金流 {names}"


def _narrate(state: AgentState, snapshot: dict, hits: list, evidence_text: str) -> tuple:
    # LLM 叙述：装载提示词 → replace 填充 → chat → JSON 三字段规范化
    prompt = load_prompt("fundamental_agent")
    rules_text = "\n".join(
        f"规则{rule['id']} 含义：{rule['meaning']} 风险警告：{rule['risk_warning']}"
        for rule, _n in hits
    ) or "未命中任何组合规则"

    # 占位符用 replace：system 段含 JSON 示例花括号，str.format 会 KeyError
    user_content = (prompt.user
                    .replace("{stock_name}", str(state.intent.get("stock_name", "")))
                    .replace("{stock_code}", str(state.intent.get("stock_code", "")))
                    .replace("{years}", str(state.intent.get("years", [])))
                    .replace("{snapshot}", "\n".join(_snapshot_lines(snapshot)))
                    .replace("{rules_hit}", rules_text)
                    .replace("{evidence}", evidence_text))

    client = get_chat_client(model=get_settings().ark.llm_model)
    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": user_content},
    ]
    try:
        # 叙述任务低温保稳定；thinking=disabled 省思考开销（叙述质量不足时再开思考对照）
        usage = {}
        text = client.chat(messages, temperature=0.2, max_tokens=1000, thinking="disabled", usage=usage)
        record_tokens(state, "fundamental", usage)
        data = _extract_json(text)
    except (ValueError, requests.RequestException) as exc:
        # JSONDecodeError 是 ValueError 子类一并覆盖；网络失败同走兜底（分析层失败不阻断链路）
        logger.warning("fundamental LLM 叙述失败，降级规则兜底: %s", exc)
        state.errors.append({"type": "fundamental_llm_error", "detail": str(exc)[:200]})
        return _fallback(snapshot, hits)

    # 契约规范化：键名中英双兼容（LLM 键漂移回落），盈利分析非空、要点为字符串列表
    profit = str(data.get("盈利分析") or data.get("earnings_analysis") or "").strip()
    positives = [str(x).strip() for x in (data.get("利好要点") or data.get("bullish_points") or [])
                 if str(x).strip()]
    negatives = [str(x).strip() for x in (data.get("利空要点") or data.get("bearish_points") or [])
                 if str(x).strip()]
    if not profit:
        return _fallback(snapshot, hits)
    return profit, positives, negatives


def _fallback(snapshot: dict, hits: list) -> tuple:
    # 规则兜底：LLM 不可用时报告仍有数据支撑的结论，方向概览拼接为盈利分析
    parts = [f"{_IND_NAMES.get(ind, ind)}最新{s['latest']}{_DIR_NAMES[s['direction']]}"
             for ind, s in snapshot.items()]
    profit = "LLM 叙述暂不可用，依据规则引擎判定：" + "；".join(parts) + "。"
    return profit, [], []


def _extract_json(text: str) -> dict:
    # 从 LLM 回复提取 JSON：剥代码围栏后取首尾花括号之间解析
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"基本面回复无 JSON 结构: {text[:200]}")
    return json.loads(text[start:end + 1])
