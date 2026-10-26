# trace 节点记账助手（可观测）
# 约定：flow 侧 _timed 写 ms，本模块写 tokens，Agent 侧不直接操作 trace 结构
from agents.orchestrator.state import AgentState


def record_tokens(state: AgentState, node: str, usage: dict) -> None:
    # LLM Agent 调用 chat(usage=...) 成功后回填分项 token 到节点 trace 段
    # 累加而非覆盖：一节点多次 LLM 调用（如 fundamental 压缩+叙述）逐次相加，
    # 覆盖语义会丢前次账
    # usage 为空说明请求失败走了降级路径，跳过不记——trace 不造假数据
    if not usage:
        return
    entry = state.trace["nodes"].setdefault(node, {})
    tokens = entry.setdefault("tokens", {})
    for k, v in usage.items():
        tokens[k] = tokens.get(k, 0) + v


def record_retrieval(state: AgentState, node: str, query: str, results: list) -> None:
    # 检索段记账：query/命中清单/腿归因/分数进节点 trace 段
    # results: rag.schemas.RetrievalResult 列表；空列表同样记录——零命中也是可观测事实
    entry = state.trace["nodes"].setdefault(node, {})
    entry["retrieval"] = {
        "query": query,
        "top_k": len(results),
        "chunk_ids": [r.chunk.chunk_id for r in results],
        "sources": [r.source for r in results],
        "scores": [r.score for r in results],
    }
