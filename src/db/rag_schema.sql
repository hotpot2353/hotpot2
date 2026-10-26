-- RAG 检索层：年报切块父子表（chunks）
-- 与 schema.sql 分层：那边是行情/财务的结构化数值层，本表是语义检索的文本块层
-- 口径对应 configs/rag.yaml：表名 = store.table_chunks；向量维度 = embedding.dimension
--   （2048，doubao-embedding-vision 251215 版，输出已归一化）
-- 幂等：全部 IF NOT EXISTS，init_db.py 可重复执行无副作用
-- 注意：IF NOT EXISTS 只保证"不存在则建"，不会自动更新已有表结构——改字段需手动迁移

-- 激活 pgvector 扩展（幂等）：提供 vector(N) 列类型与余弦距离算子 <=>
-- docker 镜像 pgvector/pgvector:pg17 已自带扩展文件，此句只是在当前库挂载激活
CREATE EXTENSION IF NOT EXISTS vector;

-- 一行 = 一个块；父子同表（父块量级 ≈ 章节数，远小于子块，不值得分表）
-- child = 可检索子块：带 embedding + content_tsv 两条检索腿的原料，是向量/BM25 召回的对象
-- parent = 整节父块：不带任何检索原料，仅按 parent_id 被回取后喂 LLM（Small-to-Big，父块不花嵌入费）
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        CHAR(64)     PRIMARY KEY, -- 内容哈希主键 = sha256("股票|年份|章节|角色|原文")。稳定性：重跑切块产出同一 ID；幂等由此保证（入库 ON CONFLICT (chunk_id) DO NOTHING）；章节入哈希使同文本在不同章节仍算不同块
    stock_code      CHAR(6)      NOT NULL,    -- 股票代码：召回前过滤的硬边界，杜绝跨股票召回污染
    year            SMALLINT     NOT NULL,    -- 年报年度
    section_path    TEXT         NOT NULL,    -- 章节面包屑（normalize 产物，已验证无孤儿）：可选过滤维度，也是哈希原料
    chunk_role      VARCHAR(6)   NOT NULL,    -- 块角色：parent = 整节父块 / child = 可检索子块
    parent_id       CHAR(64),                -- 子块 → 所属父块的 chunk_id；父块自身为 NULL。不建外键：批量灌数逐行校验外键拖慢写入，且父子由切块脚本同批产出、同报同事务删除，无悬挂风险
    content         TEXT         NOT NULL,    -- 块原文（markdown，表格也是文本形态）。嵌入输入 = 上下文前缀 + 原文，前缀不落库
    content_tsv     TSVECTOR,                -- BM25 原料：jieba 在应用侧预分词 → 空格拼接 → to_tsvector('simple', …) 写入。PG 不懂中文分词，智能在应用侧，数据库只存只查；仅 child 填写
    embedding       halfvec(2048),           -- 语义向量（doubao-embedding-vision，输出已归一化 → 检索用余弦 <=>）。维度是建表契约：改维度 = 清表重灌，嵌入费用全损；仅 child 填写
                                  -- 用 halfvec 而非 vector：pgvector 索引（HNSW/IVFFlat）对 vector 上限 2000 维，2048 维只能走 halfvec（索引上限 4000 维）；
                                  -- fp16 精度损失 ~1e-3，远小于 HNSW 自身近似误差，归一化向量检索无实际影响
    embedding_model VARCHAR(48),             -- 实际生成向量的模型 ID（显式锁版本，如 doubao-embedding-vision-251215）。防"别名漂移"导致新旧版向量混存不可比——追溯向量是谁生成的；仅 child 填写
    page_start      SMALLINT,                -- 溯源：块起始页码（研报可引用"见年报第 N 页"）
    page_end        SMALLINT,                -- 溯源：块结束页码
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(), -- 写入时刻，审计用

    -- 角色契约（一条约束锁死两侧）：
    -- child 必须带全套检索原料（嵌入在入库前已算好，无"先插后补"）；parent 必须纯净（绝不混入检索池）
    CONSTRAINT chk_chunks_role CHECK (
        (chunk_role = 'child'
            AND parent_id IS NOT NULL
            AND embedding IS NOT NULL
            AND content_tsv IS NOT NULL
            AND embedding_model IS NOT NULL)
        OR
        (chunk_role = 'parent'
            AND parent_id IS NULL
            AND embedding IS NULL
            AND content_tsv IS NULL
            AND embedding_model IS NULL)
    )
);

-- 按股按年定位索引：服务整报"先删后插"重建（normalize/chunker 版本 bump 后）与入库账目核对
-- 幂等不靠它（靠 chunk_id 主键），它只加速 DELETE / WHERE 定位
CREATE INDEX IF NOT EXISTS idx_chunks_report ON chunks (stock_code, year);

COMMENT ON TABLE chunks IS '年报切块父子表：child 带向量+词袋供混合检索，parent 存整节原文供 Small-to-Big 回取；主键=内容哈希实现幂等';
