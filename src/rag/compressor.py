# 检索证据定向摘要器：命中子块 → LLM 抽取式摘要
# 摘要 + evidence_ref 进 state，原文不进；token 上限 =
#   config.yaml compression.max_tokens，开关 compression.enabled（debug 置 false 直传全文）
# 降级语义（关键约定）：
#   (None, refs) = 压缩开关关闭，调用方应直传子块原文，refs 有效（叙述确实消费了原文）
#   (None, [])   = 摘要失败/为空，叙述退回无证据路径，refs 一并清空——结论没消费
#                  证据就不许留溯源标记，杜绝假引用
import json

import requests

from agents.orchestrator.state import AgentState
from core.config import get_settings
from core.logger import get_logger
from core.prompt_loader import load_prompt
from core.trace import record_tokens
from llm.chat_client import get_chat_client

logger = get_logger(__name__)


def format_chunks(results: list) -> str:
    # 命中子块 → 编号文本块：摘要与直传原文共用同一格式，评测口径一致
    return "\n\n".join(
        f"[{i}] 章节：{r.chunk.section_path}\n{r.chunk.text}"
        for i, r in enumerate(results, 1))


def evidence_refs(results: list) -> list:
    # 命中子块 → evidence_ref 清单（契约：chunk_id/stock_code/year/section_path，
    # 见 fundamental/agent.py 头注释；chunk_id 同时是父块回链与去重键）
    return [{"chunk_id": r.chunk.chunk_id, "stock_code": r.chunk.stock_code,
             "year": r.chunk.year, "section_path": r.chunk.section_path}
            for r in results]


def compress(state: AgentState, results: list) -> tuple:
    # 定向摘要入口：返回 (摘要文本 or None, evidence_ref)
    if not results:
        return None, []
    refs = evidence_refs(results)

    comp = get_settings().compression
    if not comp.enabled:
        logger.info("[compressor] 压缩开关关闭 直传 %d 块原文", len(results))
        return None, refs

    prompt = load_prompt("rag_compressor")
    intent = state.intent
    user_content = (prompt.user
                    .replace("{stock_name}", str(intent.get("stock_name", "")))
                    .replace("{stock_code}", str(intent.get("stock_code", "")))
                    .replace("{year}", str(results[0].chunk.year or ""))
                    .replace("{chunks}", format_chunks(results)))
    client = get_chat_client(model=get_settings().ark.llm_model)
    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": user_content},
    ]
    try:
        # 抽取式任务低温保稳定；token 上限即压缩预算（yaml 参数化）
        usage = {}
        text = client.chat(messages, temperature=0.1,
                           max_tokens=comp.max_tokens, thinking="disabled", usage=usage)
        record_tokens(state, "fundamental", usage)
        data = _extract_json(text)
    except (ValueError, requests.RequestException) as exc:
        # ValueError 覆盖 JSONDecodeError；网络失败同走降级（RAG 是增强不是依赖）
        logger.warning("证据摘要失败 降级无证据叙述: %s", exc)
        state.errors.append({"type": "fundamental_compress_error", "detail": str(exc)[:200]})
        return None, []

    summary = str(data.get("摘要") or data.get("summary") or "").strip()
    if not summary:
        # 空摘要等同失败：不消费证据就清空 refs
        logger.warning("证据摘要为空 降级无证据叙述")
        state.errors.append({"type": "fundamental_compress_error", "detail": "empty summary"})
        return None, []
    logger.info("[compressor] %d 块 → 摘要 %d 字 (max_tokens=%s)",
                len(results), len(summary), comp.max_tokens)
    # 摘要文本与体量进 trace（评测消费：压缩比 + check_fidelity 数值保真检查）
    entry = state.trace["nodes"].setdefault("fundamental", {})
    entry.setdefault("retrieval", {}).update({
        "summary": summary,
        "summary_chars": len(summary),
        "chunks_chars": sum(len(r.chunk.text) for r in results),
    })
    return summary, refs


def _extract_json(text: str) -> dict:
    # 剥代码围栏 + 首尾花括号截取，防 LLM 输出夹带说明文字
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").lstrip("json").strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"摘要回复无 JSON 结构: {text[:200]}")
    return json.loads(t[start:end + 1])
