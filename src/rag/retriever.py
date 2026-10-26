# 混合检索器：向量腿 + BM25 腿 → RRF 融合 → (rerank 精排) → Small-to-Big 父块回取
# 查询期组件：chunks 表父子块 + HNSW/GIN 索引（embed_store 离线管道产物）
# 链路：rerank=false 融合序直接截断 final_top_k；
#       rerank=true 融合序截断 rerank_top_n → VikingDB doubao-seed-rerank 精排 → final_top_k
# BM25 查询侧分词与入库同口径（jieba cut_for_search + simple 配置），词元对齐才能命中
# 归因：source 标记 vector / bm25 / both，供评测分路统计（Recall/Precision/MRR 按腿归因）
import re
from collections import defaultdict

import jieba

from core.logger import get_logger
from db.connection import connect
from llm.embedding_client import get_embedding_client
from llm.rerank_client import get_rerank_client
from rag.config import RagSettings, get_rag_settings
from rag.schemas import Chunk, RetrievalResult

logger = get_logger(__name__)

# BM25 查询词元安全过滤：仅字母/数字/汉字词元（剔除标点）；单字词区分度不足一并剔除
_SAFE_TOKEN = re.compile(r"^[\w\u4e00-\u9fff]+$")


class HybridRetriever:
    # 混合检索入口：两腿召回 + RRF 融合；无状态，可复用单实例（嵌入客户端按参数单例）

    def __init__(self, settings: RagSettings = None):
        self.settings = settings or get_rag_settings()
        self.emb_client = get_embedding_client(
            self.settings.embedding.model, self.settings.embedding.dimension)
        # 首次建词典约 1s，初始化时预热避免首查延迟突刺
        jieba.initialize()

    def retrieve(self, query: str, stock_code: str = None, year: int = None,
                 legs: tuple = ("vector", "bm25")) -> list:
        # 检索主入口；legs 供评测单腿归因（vector-only / bm25-only / 混合）
        # stock_code / year 召回前过滤，杜绝跨股票/跨年份召回污染
        cfg = self.settings.retrieval
        vec = None
        if "vector" in legs:
            # 查询嵌入：与库内向量同模型同维度（客户端已锁显式 Model ID，向量可比）
            vec = self.emb_client.embed_texts([query])[0]
        rows_v, rows_b = [], []
        with connect() as conn:
            with conn.cursor() as cur:
                if vec is not None:
                    rows_v = self._vector_leg(cur, vec, stock_code, year, cfg.vector_top_k)
                if "bm25" in legs:
                    rows_b = self._bm25_leg(cur, query, stock_code, year, cfg.bm25_top_k)
        merged = self._rrf_merge(rows_v, rows_b)
        # rerank=false：融合序直接截断 final_top_k；rerank=true：截断 rerank_top_n 精排后回截
        if cfg.rerank and merged:
            top = self._rerank(query, merged)
        else:
            top = merged[: cfg.final_top_k]
        results = [
            RetrievalResult(
                chunk=Chunk(
                    chunk_id=row[0], stock_code=row[1], year=row[2],
                    section_path=row[3], parent_id=row[4], text=row[5]),
                score=round(score, 6),
                source=source,
            )
            for row, score, source in top
        ]
        logger.info("检索完成 query=%r legs=%s 过滤=(%s,%s) 命中=%s",
                    query, legs, stock_code, year, len(results))
        return results

    def fetch_parents(self, results: list) -> dict:
        # Small-to-Big：命中子块凭 parent_id 回取父块整节原文；按 parent_id 去重批量查
        parent_ids = sorted({r.chunk.parent_id for r in results if r.chunk.parent_id})
        if not parent_ids:
            return {}
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT chunk_id, content FROM chunks WHERE chunk_id = ANY(%s)",
                    (parent_ids,))
                return dict(cur.fetchall())

    def _vector_leg(self, cur, vec: list, stock_code: str, year: int, top_k: int) -> list:
        # 向量腿：HNSW 余弦近邻（halfvec_cosine_ops）；查询向量文本形态传入按目标列解析
        conds = ["chunk_role = 'child'"]  # 只检索子块
        params = []
        if stock_code: # 过滤股票代码
            conds.append("stock_code = %s") 
            params.append(stock_code) 
        if year: # 过滤年份
            conds.append("year = %s")
            params.append(year)
        sql = (
            "SELECT chunk_id, stock_code, year, section_path, parent_id, content "
            "FROM chunks WHERE " + " AND ".join(conds) +
            " ORDER BY embedding <=> %s::halfvec LIMIT %s")
        # str(vec)：psycopg 驱动传数组不可靠，字符串形态传入后由 ::halfvec 解析成向量
        params += [str(vec), top_k]
        cur.execute(sql, params)
        return cur.fetchall()

    def _bm25_leg(self, cur, query: str, stock_code: str, year: int, top_k: int) -> list:
        # BM25 腿：jieba 分词 → AND 语义 tsquery → tsvector 匹配 + ts_rank 排序
        # and_or_fallback：AND 全词元命中过严，零命中降级 OR 放宽——
        #   AND 有结果时行为与严格模式完全一致（单调改进）；OR 候选噪声由 rerank 精排压制（协同）
        #   单词元 query 的 OR 串与 AND 串相同，跳过重复查询
        tsq = self._tsquery(query)
        if not tsq:
            # 全部词元被过滤（如纯标点/单字 query）：BM25 腿空手而归，不阻塞向量腿
            return []
        rows = self._bm25_query(cur, tsq, stock_code, year, top_k)
        if not rows and self.settings.retrieval.bm25_operator == "and_or_fallback":
            tsq_or = self._tsquery(query, "|")
            if tsq_or != tsq:
                rows = self._bm25_query(cur, tsq_or, stock_code, year, top_k)
        return rows

    def _bm25_query(self, cur, tsq: str, stock_code: str, year: int, top_k: int) -> list:
        # 单次 BM25 查询：tsq 已就绪，动态 WHERE 过滤 + ts_rank 排序截断
        # @@ 条件走 GIN 倒排索引：从词元反查命中块
        conds = ["chunk_role = 'child'", "content_tsv @@ to_tsquery('simple', %s)"]
        params = [tsq]
        if stock_code:
            conds.append("stock_code = %s")
            params.append(stock_code)
        if year:
            conds.append("year = %s")
            params.append(year)
        # 次级排序键 chunk_id：ts_rank 分数相同按字典序，保证同数据每次查询返回顺序一致
        sql = (
            "SELECT chunk_id, stock_code, year, section_path, parent_id, content "
            "FROM chunks WHERE " + " AND ".join(conds) +
            " ORDER BY ts_rank(content_tsv, to_tsquery('simple', %s)) DESC, chunk_id"
            " LIMIT %s")
        params += [tsq, top_k]
        cur.execute(sql, params)
        return cur.fetchall()

    @staticmethod
    def _tsquery(query: str, operator: str = "&") -> str:
        # 分词口径与入库一致（cut_for_search）；去重保序、剔除单字与含标点词元后按 operator 拼接
        # operator：& 全词元命中（精确） / | 任一词元命中（放宽），ts_rank 偏向多词元命中文档（软 AND）
        tokens = []
        seen = set()
        for t in jieba.cut_for_search(query):
            t = t.strip()
            if len(t) > 1 and t not in seen and _SAFE_TOKEN.match(t):
                seen.add(t)
                tokens.append(t)
        return f" {operator} ".join(tokens)

    def _rerank(self, query: str, merged: list) -> list:
        # 精排腿：RRF 融合序截断 rerank_top_n → VikingDB 得分重排 → final_top_k
        # score 语义切换为 rerank 分（schemas.RetrievalResult.score 注释口径），source 归因保留
        cfg = self.settings.retrieval
        cand = merged[: cfg.rerank_top_n]
        scores = get_rerank_client().scores(
            query, [row[5] for row, _, _ in cand],
            model=cfg.rerank_model, instruction=cfg.rerank_instruction)
        ranked = sorted(
            ((row, sc, src) for (row, _, src), sc in zip(cand, scores)),
            # 分数降序、chunk_id 决胜，保证输出顺序确定
            key=lambda t: (-t[1], t[0][0]))
        return ranked[: cfg.final_top_k]

    def _rrf_merge(self, rows_v: list, rows_b: list) -> list:
        # RRF：score = Σ 1/(k + rank)，纯排名融合无权重（两腿分数量纲不可比，加权需归一化且调参脆）
        cfg = self.settings.retrieval
        scores = defaultdict(float)  # chunk_id → 累计 RRF 得分
        sources = defaultdict(set)   # chunk_id → 命中腿集合（归因用）
        meta = {}                    # chunk_id → 原始行数据
        for rows, leg in ((rows_v, "vector"), (rows_b, "bm25")):
            for rank, row in enumerate(rows, start=1):
                cid = row[0]
                scores[cid] += 1.0 / (cfg.rrf_k + rank)
                sources[cid].add(leg)
                meta[cid] = row
        # 得分降序、chunk_id 升序决胜，保证同数据输出顺序一致
        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [
            (meta[cid], sc,
             "both" if len(sources[cid]) == 2 else next(iter(sources[cid])))
            for cid, sc in ordered
        ]
