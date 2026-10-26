# 综合建议Agent：跨维度确定性评分 → 评级/时机/信号/纪律 → LLM 仅组织"核心逻辑"叙述
# LLM 边界：评级、评分、操作时机、关注信号、适合人群、纪律全部由代码从上游
#   结构化结论推导（可复现、可回溯），LLM 只把评分依据改写成 ≤3 条人话短句，不改写任何结论；
#   叙述失败降级为评分依据原文，链路不中断
# 合规边界：输出定性为"程序化研判参考"，输出契约自带免责字段，报告尾部免责声明保留
import json

import requests

from agents.orchestrator.state import AgentState
from core.config import get_settings
from core.logger import get_logger
from core.prompt_loader import load_prompt
from core.trace import record_tokens
from llm.chat_client import get_chat_client

logger = get_logger(__name__)

# 评级分档（总分阈值，代码锁死）：LLM 与下游均不得改档
_RATING_BANDS = ((2.5, "积极关注"), (1.0, "谨慎参与"), (-1.5, "观望为主"))

# 舆情情绪 → 分值
_MOOD_SCORE = {"偏正面": 1.0, "中性偏正": 0.5, "中性": 0.0, "中性偏负": -0.5, "偏负面": -1.0}

# 免责声明（报告尾部与建议节双保险）
_DISCLAIMER = "以上为程序化研判参考，不构成投资建议；市场有风险，决策需独立判断"


def run(state: AgentState) -> dict:
    # 综合建议入口：跨维度评分 → 评级/时机/信号/纪律 → LLM 核心逻辑叙述 → 契约组装
    fund = state.fundamental or {}
    tech = state.technical or {}
    senti = state.sentiment or {}
    risk = state.risk or {}

    parts, total = [], 0.0

    # —— 基本面：方向快照 上行/下行 净差（缺失计 0 并显式标注）——
    snap = fund.get("方向快照") or {}
    fund_deg = not snap
    ups = sum(1 for s in snap.values() if s["direction"] == "up")
    downs = sum(1 for s in snap.values() if s["direction"] == "down")
    fund_score = max(-3.0, min(3.0, ups - downs)) * 0.5
    total += fund_score
    parts.append(f"基本面 {'无数据' if fund_deg else f'{ups} 项上行 {downs} 项下行'}（{fund_score:+.1f}）")

    # —— 技术面：均线形态 + 20日线位置 + 量价配合 + 区间位置 ——
    align = tech.get("均线形态") or ""
    box = tech.get("关键价格区间") or {}
    support, pressure = box.get("支撑"), box.get("压力")
    align_score = {"多头排列": 1.5, "空头排列": -1.5}.get(align, 0.0)
    tech_score = align_score
    parts.append(f"技术面 {align or '无数据'}（{align_score:+.1f}）")

    # 区间分位/量比等细节从叙述不可反解——技术面契约无数值快照，重复计算违反单一来源；
    # 此处以 均线形态 为准，分位/量比细节进"关注信号"而非评分
    total += tech_score

    # —— 舆情：整体情绪分值 + 重大负面惩罚 ——
    mood = senti.get("整体情绪") or ""
    mood_score = _MOOD_SCORE.get(mood, 0.0)
    majors = len(senti.get("重大负面") or [])
    if majors:
        mood_score -= 0.5
    total += mood_score
    parts.append(f"舆情 {mood or '无数据'}{f'，重大负面 {majors} 条' if majors else ''}（{mood_score:+.1f}）")

    # —— 风险：高级项重罚、中级项轻罚（数据缺失/异动均为中，封顶防淹没结论）——
    items = risk.get("风险清单") or []
    n_high = sum(1 for x in items if x.get("等级") == "高")
    n_mid = sum(1 for x in items if x.get("等级") == "中")
    risk_score = -1.0 * n_high - min(1.0, 0.2 * n_mid)
    total += risk_score
    parts.append(f"风险 高 {n_high} 项 / 中 {n_mid} 项（{risk_score:+.1f}）")

    rating = next((label for th, label in _RATING_BANDS if total >= th), "暂时回避")
    if fund_deg and not align:
        rating = "观望为主（数据不足）"

    timing = _timing(tech, fund_deg)
    signals = _signals(tech, senti)
    audience = _audience(rating)
    discipline = _discipline(support)

    logic = _narrate(state, rating, parts)
    logger.info("[advice] %s 评分 %.2f 评级=%s 时机=%s",
                state.intent.get("stock_code"), total, rating, timing[:24])
    return {
        "综合评级": rating,
        "评分": round(total, 2),
        "评分依据": parts,
        "核心逻辑": logic,
        "操作时机": timing,
        "关注信号": signals,
        "适合人群": audience,
        "纪律提醒": discipline,
        "免责": _DISCLAIMER,
    }


def _timing(tech: dict, fund_deg: bool) -> str:
    # 操作时机：均线形态 × 关键价位 的条件式建议（纯代码，可复现）
    align = tech.get("均线形态") or ""
    box = tech.get("关键价格区间") or {}
    support, pressure = box.get("支撑"), box.get("压力")
    sup = f"{support:.2f}" if isinstance(support, (int, float)) else "支撑位"
    pre = f"{pressure:.2f}" if isinstance(pressure, (int, float)) else "压力位"
    if align == "多头排列":
        text = (f"趋势顺势：回踩 20 日均线或支撑 {sup} 附近企稳可分批参与；"
                f"收盘连续两日跌破 20 日均线则减仓离场")
    elif align == "空头排列":
        text = (f"右侧等待：收盘站回 20 日均线且放量突破压力 {pre} 再考虑参与；"
                f"左侧抄底仅适合长线资金的极小仓位")
    elif align == "均线纠缠":
        text = f"区间思路：支撑 {sup} 附近低吸、压力 {pre} 附近减仓，破位则离场观望"
    else:
        text = "行情数据不足，无法给出时机建议"
    if fund_deg:
        text += "；财务口径缺失，参与仓位建议减半，仅以短周期视角对待"
    return text


def _signals(tech: dict, senti: dict) -> list:
    # 关注信号：只列可计算、可验证的触发条件
    out = []
    box = tech.get("关键价格区间") or {}
    if box.get("支撑") is not None:
        out.append(f"跌破支撑 {box['支撑']:.2f} 元（止损触发条件）")
    if box.get("压力") is not None:
        out.append(f"放量突破压力 {box['压力']:.2f} 元（趋势确认信号）")
    out.append("20 日均线的站回或跌破（中期强弱分界）")
    out.append("量比放大至 1.5 以上并配合价格方向（量价确认）")
    majors = senti.get("重大负面") or []
    if majors:
        out.append(f"舆情重大负面事件发酵（当前 {len(majors)} 条）")
    return out


def _audience(rating: str) -> str:
    return {
        "积极关注": "趋势型投资者可正常参与；稳健型投资者小仓位跟随，不追高",
        "谨慎参与": "稳健型投资者小仓位试错，严格设定止损；新手建议观望",
        "观望为主": "非必要不参与；已持有者关注减仓信号，破位即执行",
        "暂时回避": "风险偏好较低者规避；短线资金仅博弈超跌反弹，不恋战",
    }.get(rating, "数据不足，建议等待信息补齐后再评估")


def _discipline(support) -> list:
    out = ["单标的仓位建议不超过总资金的 20%"]
    if isinstance(support, (int, float)):
        out.append(f"止损纪律：收盘有效跌破支撑 {support:.2f} 元无条件执行，不抗单")
    out.append("财报披露窗口前后波动放大，控制仓位敏感性")
    out.append("评分随数据更新动态变化，本结论仅代表当前数据快照")
    return out


def _narrate(state: AgentState, rating: str, parts: list) -> list:
    # LLM 叙述：把评分依据改写成 ≤3 条核心逻辑短句；失败降级为评分依据原文
    prompt = load_prompt("advice_agent")
    user_content = (prompt.user
                    .replace("{stock_name}", str(state.intent.get("stock_name", "")))
                    .replace("{stock_code}", str(state.intent.get("stock_code", "")))
                    .replace("{rating}", rating)
                    .replace("{score_lines}", "\n".join(parts)))
    client = get_chat_client(model=get_settings().ark.llm_model)
    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": user_content},
    ]
    try:
        usage = {}
        text = client.chat(messages, temperature=0.2, max_tokens=400, thinking="disabled", usage=usage)
        record_tokens(state, "advice", usage)
        data = _extract_json(text)
    except (ValueError, requests.RequestException) as exc:
        logger.warning("综合建议 LLM 叙述失败，降级评分依据: %s", exc)
        state.errors.append({"type": "advice_llm_error", "detail": str(exc)[:200]})
        return parts
    logic = [str(x).strip() for x in (data.get("核心逻辑") or []) if str(x).strip()]
    return logic[:3] or parts


def _extract_json(text: str) -> dict:
    # 剥代码围栏 + 首尾花括号截取，防 LLM 输出夹带说明文字
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").lstrip("json").strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"综合建议回复无 JSON 结构: {text[:200]}")
    return json.loads(t[start:end + 1])
