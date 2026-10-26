# 风控Agent：三层递进核验 + 跨 Agent 矛盾轻检测
#   第1层 组合规则命中风险：直接复用 fundamental 输出的指标组合（引擎已判定，不重复跑）
#   第2层 财务异动硬规则：yaml risk_industry_thresholds 行业阈值 × 同比相对变化（复用方向快照）
#         + 数据完整性：raw_data.missing / state.errors 转风险项与局限性陈述（显式降级铁律）
#   第3层 轻量 LLM 幻觉标记：仅 evidence_ref 可核验时触发，LLM 只出候选
#         （编造/矛盾/证据不足三型白名单），等级由代码按型定——标记不定案
#         证据文本优先取叙述者实际消费的定向摘要（trace.fundamental.retrieval.summary，
#         核验"叙述没超出其输入"语义更严谨且省 ~3k prompt）；摘要不存在时回链子块原文
# 开关 hallucination.enabled（评测可整体关闭，关闭即退回零 LLM 双层）
import json

import requests

from agents.orchestrator.state import AgentState
from core.config import get_settings, load_yaml
from core.logger import get_logger
from core.prompt_loader import load_prompt
from core.trace import record_tokens
from db.connection import connect
from llm.chat_client import get_chat_client

logger = get_logger(__name__)

# 规则文件相对 configs/ 的路径（与规则引擎同源）
_RULES_FILE = "rules/analysis_rules.yaml"

# 幻觉候选型 → 风险等级（LLM 不定案：型→级的映射由代码锁死）
_TYPE_LEVELS = {"编造": "高", "矛盾": "中", "证据不足": "低"}
_VALID_TYPES = ("编造", "矛盾", "证据不足")

# 指标键 → 中文名（风险项描述可读性；与 yaml indicator_labels 口径一致）
_LABELS = {
    "gross_margin": "毛利率",
    "net_margin": "净利率",
    "roe": "ROE",
    "cash_to_net_profit": "净现比",
    "receivable_turnover": "应收账款周转率",
    "inventory_turnover": "存货周转率",
    "pe": "PE",
    "pb": "PB",
}

# 舆情结论白名单（矛盾检测只对正向情绪与空头形态对冲）
_POSITIVE_MOODS = ("偏正面", "中性偏正")

# 采集缺口机器码 → 用户可读描述：风险清单与局限性声明面向用户，不暴露内部编码；
# 未知编码原样兜底（可观测性优先于美观，trace.errors 仍保留原始编码）
_MISSING_TEXT = {
    "financials:out_of_pool": "标的不在年报覆盖池，无结构化财务与年报数据",
    "financials:db_error": "财务数据库查询失败",
    "financials:valuation_db_error": "估值（PE/PB）数据库查询失败",
    "financials:valuation:empty": "估值快照缺失",
    "financials:pe:empty": "市盈率（PE）未披露",
    "financials:pb:empty": "市净率（PB）未披露",
    "quotes:db_error": "行情数据库查询失败",
    "quotes:akshare_error": "行情增量接口（akshare）调用失败",
    "quotes:akshare_empty": "行情接口返回为空",
    "news:em_error": "新闻资讯接口调用失败",
    "news:em_empty": "该标的暂无新闻资讯",
}


def _missing_label(code: str) -> str:
    # 缺口码翻译：整码命中直查；财务指标级缺口（financials:{指标}:{原因}）拼中文名
    if code in _MISSING_TEXT:
        return _MISSING_TEXT[code]
    parts = code.split(":")
    if len(parts) == 3 and parts[0] == "financials":
        label = _LABELS.get(parts[1], parts[1])
        if parts[2] == "unmapped_or_empty":
            return f"财务指标「{label}」未披露或无数据"
        if parts[2] == "empty":
            return f"财务指标「{label}」未披露"
    return code


def run(state: AgentState) -> dict:
    # 风控入口：三分析汇合后执行（flow and_ 保证）
    # 第1/2层零 LLM 可追溯规则或数据记录；第3层唯一 LLM 调用且开关门控、只标记不定案
    fund = state.fundamental or {}
    tech = state.technical or {}
    senti = state.sentiment or {}
    missing = state.raw_data.get("missing") or []
    code = state.intent["stock_code"]

    # 行业归属与异动阈值：yaml 现读（与规则引擎同源，改配置不改代码）
    cfg = load_yaml(_RULES_FILE)
    industry = (cfg.get("industry_by_code") or {}).get(code)
    th_cfg = cfg.get("risk_industry_thresholds") or {}
    ind_th = (th_cfg.get("industries") or {}).get(industry) or {}
    default_th = float(th_cfg.get("default", 0.50))

    risks = []

    # 第1层 组合规则命中 → 风险项（警告为 yaml 原文，fundamental 已保证 LLM 无权改写）
    for item in fund.get("指标组合") or []:
        risks.append({"等级": "中", "类型": f"组合规则{item['id']}", "描述": item["风险警告"]})

    # 第2层 财务异动：同比相对变化绝对值超行业阈值即标记（复用快照不重算）
    #    prev 为 0 或缺失跳过（除零无意义）；下行异动等级高（恶化方向），上行异动等级中（仅关注）
    for key, s in (fund.get("方向快照") or {}).items():
        cur, prev = s.get("latest"), s.get("prev")
        if cur is None or not prev:
            continue
        rel = abs(cur - prev) / abs(prev)
        th = float(ind_th.get(key, ind_th.get("default", default_th)))
        if rel >= th:
            level = "高" if s["direction"] == "down" else "中"
            risks.append({
                "等级": level,
                "类型": "财务异动",
                "描述": f"{_LABELS.get(key, key)} 同比变化 {rel:+.0%}，超出{industry or '默认'}阈值 {th:.0%}",
            })

    # 第2层 数据完整性：采集缺口逐条转风险项——缺失必须显式声明，禁止被下游误读为无风险
    #    描述面向用户（翻译为人话）；原始编码留在 trace.errors 供评测回溯
    for m in missing:
        risks.append({"等级": "中", "类型": "数据缺失", "描述": _missing_label(m)})

    # 第3层 幻觉标记：evidence_ref 回链证据原文 → LLM 候选 → 代码按型定级并入风险清单
    risks += _hallucination_check(state, fund)

    # 跨 Agent 矛盾轻检测：技术面趋势与舆情情绪背离（evidence 级核验随第3层增强）
    conflicts = []
    if tech.get("均线形态") == "空头排列" and senti.get("整体情绪") in _POSITIVE_MOODS:
        conflicts.append("技术面空头排列但舆情偏正面，注意价格趋势与市场情绪背离")

    limits = _limitations(missing, state.errors, industry)
    if not risks:
        risks.append({"等级": "低", "类型": "常规", "描述": "组合规则、财务异动、数据完整性校验均未触发"})
    logger.info("[risk] %s 行业=%s 风险 %d 项 矛盾 %d 项 局限 %d 项",
                code, industry, len(risks), len(conflicts), len(limits))
    return {"风险清单": risks, "矛盾结论": conflicts, "局限性": limits}


def _hallucination_check(state: AgentState, fund: dict) -> list:
    # 第3层入口：三重门控（开关/有证据/有叙述）不满足即零开销跳过
    refs = fund.get("evidence_ref") or []
    narrative = str(fund.get("盈利分析") or "").strip()
    if not get_settings().hallucination.enabled:
        return []
    if not refs or not narrative:
        return []  # 无证据可核验（降级路径）：跳过，缺口已由局限性声明覆盖

    try:
        # 证据文本二态：叙述者看过摘要（压缩开/成功）就对摘要核验——验证"叙述没超出
        # 其输入"且省 ~3k prompt；摘要不存在（压缩关/摘要失败）回链子块原文
        summary = ((state.trace.get("nodes", {}).get("fundamental", {}).get("retrieval") or {})
                   .get("summary") or "")
        if summary:
            evidence_text, source = summary, f"摘要{len(summary)}字"
        else:
            chunks = _fetch_chunks([r["chunk_id"] for r in refs])
            if not chunks:
                logger.warning("[risk] 证据回链 0/%d 块 跳过幻觉标记", len(refs))
                return []
            evidence_text = "\n\n".join(
                f"[证据{i}] 章节：{path}\n{content}"
                for i, (_cid, path, content) in enumerate(chunks, 1))
            source = f"原文{len(chunks)}块"
        candidates = _mark(state, narrative, evidence_text)
    except (ValueError, requests.RequestException) as exc:
        logger.warning("[risk] 幻觉标记失败 跳过: %s", exc)
        state.errors.append({"type": "risk_hallucination_error", "detail": str(exc)[:200]})
        return []

    # 候选 → 风险项：白名单外丢弃；等级查表代码定（LLM 标记不定案）
    out = []
    for c in candidates:
        t = str(c.get("type", "")).strip()
        if t not in _VALID_TYPES:
            continue
        sent = str(c.get("原句", "")).strip()[:80]
        note = str(c.get("说明", "")).strip()
        out.append({"等级": _TYPE_LEVELS[t], "类型": f"幻觉候选·{t}",
                    "描述": f"叙述“{sent}”：{note}"})
    logger.info("[risk] 幻觉标记 %d/%d 候选入清单 (证据%s)",
                len(out), len(candidates), source)
    return out


def _fetch_chunks(chunk_ids: list) -> list:
    # 凭 evidence_ref 的 chunk_id 回链证据原文（模块1 回链机制在核验侧的消费）
    # 取子块原文（检索时 rerank 精排的精确跨度），父块整节留给需深挖的场景
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, section_path, content FROM chunks "
                "WHERE chunk_id = ANY(%s) ORDER BY chunk_id", (sorted(chunk_ids),))
            return cur.fetchall()


def _mark(state: AgentState, narrative: str, evidence_text: str) -> list:
    # LLM 候选标记：叙述 + 证据文本（摘要或原文）一次调用；temperature=0 保判定可复现
    prompt = load_prompt("risk_hallucination")
    user_content = (prompt.user
                    .replace("{narrative}", narrative)
                    .replace("{chunks}", evidence_text))
    client = get_chat_client(model=get_settings().ark.llm_model)
    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": user_content},
    ]
    usage = {}
    text = client.chat(messages, temperature=0.0, max_tokens=500,
                       thinking="disabled", usage=usage)
    record_tokens(state, "risk", usage)
    data = _extract_json(text)
    return data.get("候选") or []


def _limitations(missing: list, errors: list, industry) -> list:
    # 局限性陈述：报告 Agent 直接引用，声明数据边界与免责
    limits = [f"采集缺口：{_missing_label(m)}" for m in missing]
    for e in errors or []:
        limits.append(f"异常记录：{e.get('type')} {e.get('detail', '')}")
    if industry is None:
        limits.append("标的行业未登记，财务异动校验走默认阈值")
    limits.append("本报告由规则引擎与程序化计算生成，仅供研究参考，不构成投资建议")
    return limits


def _extract_json(text: str) -> dict:
    # 剥代码围栏 + 首尾花括号截取，防 LLM 输出夹带说明文字
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").lstrip("json").strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"幻觉标记回复无 JSON 结构: {text[:200]}")
    return json.loads(t[start:end + 1])
