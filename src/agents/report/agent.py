# 报告Agent：纯模板渲染聚合五节点结论，零 LLM（LLM 边界）
# 单写者：全链路唯一写 reports/ 的节点；固定文件名重跑覆盖（幂等可复现）
# summary 由 state 现算——禁止硬编码结论
import datetime

from agents.orchestrator.state import AgentState
from core.logger import get_logger
from core.paths import CONFIGS_DIR, REPORTS_DIR

logger = get_logger(__name__)

# 模板锚定 configs/（项目约定：研报模板本地加载，不走 RAG 避免污染向量域）
_TEMPLATE = CONFIGS_DIR / "report_template.md"

# 均线形态 → 终端结论词（summary 用）
_TECH_WORDS = {"多头排列": "偏多", "空头排列": "偏空", "均线纠缠": "震荡"}


def run(state: AgentState) -> dict:
    # 输出契约：report_path + summary
    intent = state.intent
    code = intent.get("stock_code", "000000")
    name = intent.get("stock_name", code)
    path = REPORTS_DIR / f"{code}_report.md"

    # 占位符用 replace 填充（与 prompt_loader 同款思路，规避 str.format 花括号冲突）
    content = (_TEMPLATE.read_text(encoding="utf-8")
               .replace("{TITLE}", f"{name}（{code}）多维研判报告")
               .replace("{GENERATED_AT}", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
               .replace("{YEARS}", "/".join(str(y) for y in intent.get("years", [])))
               .replace("{DIMENSIONS}", "/".join(intent.get("dimensions", [])))
               .replace("{TRACE_ID}", state.trace.get("trace_id", "-"))
               .replace("{FUNDAMENTAL}", _fmt_fundamental(state.fundamental))
               .replace("{TECHNICAL}", _fmt_technical(state.technical))
               .replace("{SENTIMENT}", _fmt_sentiment(state.sentiment))
               .replace("{RISK}", _fmt_risk(state.risk))
               .replace("{CONFLICTS}", _fmt_conflicts(state.risk))
               .replace("{ADVICE}", _fmt_advice(state.advice))
               .replace("{LIMITS}", _fmt_limits(state.risk)))
    path.write_text(content, encoding="utf-8")

    summary = _summary(state, name, code, path.name)
    logger.info("[report] %s 研报已写入 %s", code, path)
    return {"report_path": str(path), "summary": summary}


def _fmt_fundamental(fund: dict) -> str:
    # 基本面节：叙述 + 命中规则 + 利好利空；降级输出自带原因文案，直接透传
    if not fund:
        return "基本面数据不可用，本报告不含基本面分析。"
    lines = [fund.get("盈利分析", "")]
    hits = fund.get("指标组合") or []
    if hits:
        lines.append("")
        lines.append("**命中组合规则**")
        lines += [f"- 规则{h['id']}：{h['风险警告']}" for h in hits]
    pos = fund.get("利好要点") or []
    neg = fund.get("利空要点") or []
    if pos or neg:
        lines += ["", f"利好要点：{'；'.join(pos) if pos else '无'}",
                  f"利空要点：{'；'.join(neg) if neg else '无'}"]
    # 证据引用计数：叙述有年报原文支撑时透明披露，凭 trace 可回链
    refs = fund.get("evidence_ref") or []
    if refs:
        lines += ["", f"证据引用：{len(refs)} 条年报片段（凭 trace_id 可回链原文）"]
    return "\n".join(lines)


def _fmt_technical(tech: dict) -> str:
    # 技术面节：叙述 + 价格区间 + 风险点；四类风险点未触发时显式声明（缺席不等于没有检查）
    if not tech:
        return "技术面数据不可用，本报告不含技术面分析。"
    lines = [tech.get("趋势判断", "")]
    money = tech.get("资金结论")
    if money:
        lines += ["", f"资金面：{money}"]
    box = tech.get("关键价格区间") or {}
    support, pressure = box.get("支撑"), box.get("压力")
    if support is not None and pressure is not None:
        lines += ["", f"关键价格区间：支撑 {support:.2f} 元，压力 {pressure:.2f} 元"]
    risks = tech.get("风险点") or []
    if risks:
        lines += ["", "技术面风险点"] + [f"- {r}" for r in risks]
    else:
        lines += ["", "四类技术面风险点（放量下跌/跌破20日线/深度回撤/剧烈波动）均未触发"]
    return "\n".join(lines)


def _fmt_sentiment(senti: dict) -> str:
    # 舆情节：占比三项必须齐全才渲染（_empty 降级时占比为 None）
    ratio = (senti or {}).get("占比") or {}
    if ratio.get("正面") is None:
        return "舆情数据不可用，本报告不含舆情分析。"
    lines = [
        f"整体情绪：{senti.get('整体情绪', '')}",
        f"标题占比：正面 {ratio['正面']:.0%} / 中性 {ratio['中性']:.0%} / 负面 {ratio['负面']:.0%}",
    ]
    events = senti.get("关键事件") or []
    if events:
        lines.append("关注事件：" + "；".join(events))
    majors = senti.get("重大负面") or []
    if majors:
        lines += ["", "重大负面信号"] + [f"- {m}" for m in majors]
    return "\n".join(lines)


def _fmt_risk(risk: dict) -> str:
    # 风险清单节：markdown 表格；空清单显式声明"未触发"而非留白
    items = (risk or {}).get("风险清单") or []
    if not items:
        return "组合规则、财务异动、数据完整性校验均未触发风险项。"
    lines = ["| 等级 | 类型 | 描述 |", "|---|---|---|"]
    lines += [f"| {i['等级']} | {i['类型']} | {i['描述']} |" for i in items]
    return "\n".join(lines)


def _fmt_conflicts(risk: dict) -> str:
    conflicts = (risk or {}).get("矛盾结论") or []
    if not conflicts:
        return "各维度结论未发现矛盾。"
    return "\n".join(f"- {c}" for c in conflicts)


def _fmt_limits(risk: dict) -> str:
    limits = (risk or {}).get("局限性") or []
    if not limits:
        return "- 无特别局限性声明"
    return "\n".join(f"- {x}" for x in limits)


def _fmt_advice(advice: dict) -> str:
    # 综合建议节：评级醒目 + 评分依据（可回溯）+ 核心逻辑（LLM 叙述）+ 时机/信号/纪律
    if not advice:
        return "综合建议节点未产出（历史报告无此节）。"
    lines = [f"**综合评级：{advice.get('综合评级', '-')}**（综合评分 {advice.get('评分', '-')}）", ""]
    parts = advice.get("评分依据") or []
    if parts:
        lines += ["**评分依据**"] + [f"- {x}" for x in parts]
    logic = advice.get("核心逻辑") or []
    if logic:
        lines += ["", "**核心逻辑**"] + [f"- {x}" for x in logic]
    if advice.get("操作时机"):
        lines += ["", f"**操作时机**：{advice['操作时机']}"]
    signals = advice.get("关注信号") or []
    if signals:
        lines += ["", "**关注信号**"] + [f"- {x}" for x in signals]
    if advice.get("适合人群"):
        lines += ["", f"**适合人群**：{advice['适合人群']}"]
    disc = advice.get("纪律提醒") or []
    if disc:
        lines += ["", "**纪律提醒**"] + [f"- {x}" for x in disc]
    if advice.get("免责"):
        lines += ["", f"*{advice['免责']}*"]
    return "\n".join(lines)


def _summary(state: AgentState, name: str, code: str, filename: str) -> str:
    # 终端精简结论：三个词全部由 state 现算，风险项计数取自风控清单
    snap = (state.fundamental or {}).get("方向快照") or {}
    ups = sum(1 for s in snap.values() if s["direction"] == "up")
    downs = sum(1 for s in snap.values() if s["direction"] == "down")
    if ups == 0 and downs == 0:
        fund_word = "数据缺失"
    elif ups >= downs:
        fund_word = "偏多" if ups > downs else "中性"
    else:
        fund_word = "偏空"
    tech_word = _TECH_WORDS.get((state.technical or {}).get("均线形态", ""), "数据缺失")
    mood = (state.sentiment or {}).get("整体情绪") or "数据缺失"
    n_risk = len((state.risk or {}).get("风险清单") or [])
    rating = (state.advice or {}).get("综合评级") or ""
    tail = f"，综合建议「{rating}」" if rating else ""
    return (f"{name}({code}) 研判完成：基本面{fund_word}、技术面{tech_word}、"
            f"舆情{mood}；风险 {n_risk} 项{tail}，详见 {filename}")
