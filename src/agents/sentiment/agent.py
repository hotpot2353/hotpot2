# 舆情Agent：LLM 标题三分类打标 + 代码聚合占比与事件
# LLM 边界：LLM 只做单条标题的 positive/neutral/negative 分类与事件短语，
#   占比统计、整体情绪判定、事件去重截取全部代码完成，结论可复现
# 数据边界：仅消费 collector 的新闻标题列表；新闻缺失走结构化降级不调 LLM
import json

import requests

from agents.orchestrator.state import AgentState
from core.config import get_settings
from core.logger import get_logger
from core.prompt_loader import load_prompt
from core.trace import record_tokens
from llm.chat_client import get_chat_client

logger = get_logger(__name__)

# 三分类白名单：白名单外标签直接丢弃（契约层防线）
_VALID_LABELS = ("positive", "neutral", "negative")

# 关键事件与重大负面清单的截取上限
_MAX_EVENTS = 5
_MAX_MAJOR = 3


def run(state: AgentState) -> dict:
    # 舆情入口：新闻非空才调 LLM 打标；占比与结论全部代码产出
    titles = state.raw_data.get("news") or []
    if not titles:
        logger.warning("[sentiment] 无舆情数据 降级输出")
        return _empty("无舆情数据，本报告不含舆情分析")

    labels_events = _classify(state, titles)
    if not labels_events:
        # LLM 打标全失败（_classify 内部已留痕）：结构化降级
        return _empty("舆情标注不可用，本报告不含舆情分析")

    stats = _aggregate(labels_events)
    logger.info("[sentiment] %s 标注 %d/%d 情绪=%s 重大负面=%d",
                state.intent.get("stock_code"), len(labels_events), len(titles),
                stats["整体情绪"], len(stats["重大负面"]))
    # evidence_ref 契约同 fundamental：本节点只消费标题列表，无原文引用
    stats["evidence_ref"] = []
    return stats


def _empty(reason: str) -> dict:
    # 降级输出：契约形状恒定，占比显式置空并附边界说明
    return {
        "整体情绪": reason,
        "占比": {"正面": None, "中性": None, "负面": None},
        "关键事件": [],
        "重大负面": [],
        "evidence_ref": [],
    }


def _classify(state: AgentState, titles: list) -> list:
    # LLM 批量打标：一次调用 20 条标题（省 token），返回 [(label, event), ...]
    prompt = load_prompt("sentiment_agent")
    lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))
    user_content = (prompt.user
                    .replace("{stock_name}", str(state.intent.get("stock_name", "")))
                    .replace("{stock_code}", str(state.intent.get("stock_code", "")))
                    .replace("{titles}", lines))
    client = get_chat_client(model=get_settings().ark.llm_model)
    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": user_content},
    ]
    try:
        usage = {}
        text = client.chat(messages, temperature=0.0, max_tokens=800, thinking="disabled", usage=usage)
        record_tokens(state, "sentiment", usage)
        arr = _extract_json_arr(text)
    except (ValueError, requests.RequestException) as exc:
        # ValueError 覆盖 JSONDecodeError；网络异常同属边界可恢复故障
        logger.warning("舆情 LLM 打标失败: %s", exc)
        state.errors.append({"type": "sentiment_llm_error", "detail": str(exc)})
        return []

    out = []
    for item in arr[:len(titles)]:
        label = str(item.get("label", "")).strip().lower()
        if label not in _VALID_LABELS:
            continue  # 白名单外标签丢弃，不猜测
        out.append((label, str(item.get("event", "")).strip()))
    return out


def _aggregate(labels_events: list) -> dict:
    # 代码聚合：占比、整体情绪规则判定、事件去重截取（LLM 不参与任何统计）
    n = len(labels_events)
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    events, majors = [], []
    for label, event in labels_events:
        counts[label] += 1
        if event and event not in events:
            events.append(event)
        if label == "negative" and event:
            majors.append(event)

    pos, neu, neg = counts["positive"] / n, counts["neutral"] / n, counts["negative"] / n
    # 整体情绪判定规则（阈值与优先级固定，可复现）
    if pos >= 0.5 and neg <= 0.2:
        mood = "偏正面"
    elif neg >= 0.4:
        mood = "偏负面"
    elif pos > neg:
        mood = "中性偏正"
    elif neg > pos:
        mood = "中性偏负"
    else:
        mood = "中性"

    return {
        "整体情绪": mood,
        "占比": {"正面": round(pos, 3), "中性": round(neu, 3), "负面": round(neg, 3)},
        "关键事件": events[:_MAX_EVENTS],
        "重大负面": majors[:_MAX_MAJOR],
    }


def _extract_json_arr(text: str) -> list:
    # 剥代码围栏 + 首尾方括号截取，防 LLM 输出夹带说明文字
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").lstrip("json").strip()
    start, end = t.find("["), t.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("输出不含 JSON 数组")
    return json.loads(t[start:end + 1])
