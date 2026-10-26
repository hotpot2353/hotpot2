# rag.yaml → RagSettings 冻结配置
# 与主配置分离，仅承载 P1 检索链参数；链路代码（切块/嵌入/入库/检索）启动须用户明确指令
# 与通用 RAG 模板的两处关键差异：
#   1. 融合采用 RRF（基于排名），不设 vector/bm25 权重——两路分数量纲不可比，加权需归一化、调参脆
#   2. 切块为标题感知语义切块（节 > 小节 > 子条），长度仅作兜底，不做固定 chunk_size+overlap 切分
from dataclasses import dataclass
from functools import lru_cache

from core.config import load_yaml


@dataclass(frozen=True)
class EmbeddingConfig:
    # 嵌入模型参数
    model: str
    # 向量维度；须与 PGVector 表列定义一致（建表前以火山方舟文档核对为准）
    dimension: int


@dataclass(frozen=True)
class StoreConfig:
    # PGVector 存储参数
    # 连接复用主配置 db 段（core.config.get_settings().db），此处不重复
    table_chunks: str


@dataclass(frozen=True)
class RetrievalConfig:
    # 混合检索参数：两路召回 → RRF 融合 → rerank 精排
    vector_top_k: int    # 向量召回候选量
    bm25_top_k: int      # BM25 召回候选量（jieba 预分词 tsvector）
    bm25_operator: str   # BM25 连接词：and / and_or_fallback（零命中降级 OR）
    rrf_k: int           # RRF 常数，标准取 60
    rerank: bool         # 是否启用精排
    rerank_top_n: int    # 融合后送入 rerank 的候选量
    rerank_model: str    # 精排模型
    rerank_instruction: str  # 重排指令（仅 doubao-seed-rerank 生效；空串走服务端默认）
    final_top_k: int     # 最终返回条数


@dataclass(frozen=True)
class ChunkingConfig:
    # 标题感知语义切块参数：优先级 节 > 小节 > 子条，绝不跨越小节边界；长度仅兜底
    chunker_version: str        # 切块规则版本，变更后触发全量重切（manifest 幂等键组成）
    child_max_chars: int      # 子块（嵌入召回单元）长度兜底上限
    parent_max_chars: int     # 父块（喂 LLM 的整节原文）长度兜底上限
    short_section_chars: int  # 低于该长度的过短小节向同父级相邻小节合并，防碎片


@dataclass(frozen=True)
class RagSettings:
    # RAG 专属配置聚合
    embedding: EmbeddingConfig
    store: StoreConfig
    retrieval: RetrievalConfig
    chunking: ChunkingConfig


@lru_cache(maxsize=1)
def get_rag_settings() -> RagSettings:
    # 加载并缓存 rag.yaml；重复调用返回同一实例
    raw = load_yaml("rag.yaml")

    embedding_raw = raw.get("embedding", {})
    store_raw = raw.get("store", {})
    retrieval_raw = raw.get("retrieval", {})
    chunking_raw = raw.get("chunking", {})

    return RagSettings(
        embedding=EmbeddingConfig(
            model=embedding_raw.get("model", ""),
            dimension=int(embedding_raw.get("dimension", 2048)),
        ),
        store=StoreConfig(
            table_chunks=store_raw.get("table_chunks", "chunks"),
        ),
        retrieval=RetrievalConfig(
            vector_top_k=int(retrieval_raw.get("vector_top_k", 30)),
            bm25_top_k=int(retrieval_raw.get("bm25_top_k", 30)),
            bm25_operator=retrieval_raw.get("bm25_operator", "and"),
            rrf_k=int(retrieval_raw.get("rrf_k", 60)),
            rerank=bool(retrieval_raw.get("rerank", True)),
            rerank_top_n=int(retrieval_raw.get("rerank_top_n", 30)),
            rerank_model=retrieval_raw.get("rerank_model", "m3-v2-rerank"),
            rerank_instruction=retrieval_raw.get("rerank_instruction", ""),
            final_top_k=int(retrieval_raw.get("final_top_k", 8)),
        ),
        chunking=ChunkingConfig(
            chunker_version=chunking_raw.get("chunker_version", "chunking/1"),
            child_max_chars=int(chunking_raw.get("child_max_chars", 1200)),
            parent_max_chars=int(chunking_raw.get("parent_max_chars", 6000)),
            short_section_chars=int(chunking_raw.get("short_section_chars", 200)),
        ),
    )


if __name__ == "__main__":
    # 直接运行时打印 RAG 配置摘要自检
    r = get_rag_settings()
    print("embedding:", r.embedding.model, r.embedding.dimension)
    print("store:", r.store.table_chunks)
    print("retrieval:", r.retrieval)
    print("chunking:", r.chunking)
