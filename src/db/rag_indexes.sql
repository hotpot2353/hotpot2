-- RAG 检索加速索引（重索引）：由离线入库管道在批量灌数完成后执行
-- init_db.py 刻意不执行本文件：空表先建 HNSW/GIN，灌数时每行插入都要维护索引结构，批量场景慢数倍
-- 幂等：IF NOT EXISTS，入库管道可重复执行

-- HNSW 向量近邻索引：halfvec_cosine_ops 对应半精度余弦距离算子 <=>（模型输出已归一化）
-- 列类型 halfvec(2048) 而非 vector：pgvector 索引对 vector 上限 2000 维，2048 维必须走 halfvec（上限 4000）
-- m / ef_construction 用默认值（16/64）即可——数万级向量无需调参；查询期可 SET hnsw.ef_search 调召回率
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding halfvec_cosine_ops);

-- GIN 倒排索引：加速 content_tsv 的 @@ 匹配（"词 → 块"清单），BM25 召回腿
CREATE INDEX IF NOT EXISTS idx_chunks_tsv_gin
    ON chunks USING gin (content_tsv);
