-- 结构化数据层：行情日线 / 财务指标长表 / 实时快照
-- 不含向量列；RAG 阶段的文档切块表另行创建

-- 前复权日线行情：每只股票每个交易日一行
CREATE TABLE IF NOT EXISTS fin_quotes (
    stock_code    CHAR(6)      NOT NULL,  -- 股票代码，如 600519
    trade_date    DATE         NOT NULL,  -- 交易日期，2024-01-02
    open          NUMERIC(12, 2), -- 开盘价
    close         NUMERIC(12, 2),        -- 收盘价
    high          NUMERIC(12, 2),        -- 最高价
    low           NUMERIC(12, 2),        -- 最低价
    volume        BIGINT,                -- 成交量（手），1 手 = 100 股
    amount        NUMERIC(18, 2),        -- 成交额（元）
    amplitude_pct NUMERIC(8, 3),         -- 振幅 % =（最高-最低）/昨收
    change_pct    NUMERIC(10, 4),        -- 涨跌幅 % =（今收-昨收）/昨收
    change_amount NUMERIC(12, 2),        -- 涨跌额（元）
    turnover_rate NUMERIC(10, 4),        -- 换手率 % = 成交量/流通股本，骤升常是资金异动信号
    PRIMARY KEY (stock_code, trade_date)
);

COMMENT ON TABLE fin_quotes IS '前复权日线行情，来源 akshare stock_zh_a_hist';

-- 财务指标长表：每股每报告期每指标一行
CREATE TABLE IF NOT EXISTS fin_financials (
    stock_code  CHAR(6)     NOT NULL,-- 股票代码
    report_date DATE        NOT NULL,    -- 报告期末（0331/0630/0930/1231）
    category    VARCHAR(16) NOT NULL,    -- 常用指标/每股指标/盈利能力/...
    metric      VARCHAR(64) NOT NULL,    -- 指标名
    value       NUMERIC(20, 4),          -- 空串表示该期无披露
    PRIMARY KEY (stock_code, report_date, metric)
);

COMMENT ON TABLE fin_financials IS '财务摘要长表，来源 akshare stock_financial_abstract';

-- 行情实时快照（每次采集整表覆盖），每只股票一行
CREATE TABLE IF NOT EXISTS fin_quote_snapshot (
    stock_code    CHAR(6)     PRIMARY KEY,
    name          VARCHAR(32),
    price         NUMERIC(12, 2),
    change_pct    NUMERIC(10, 4),
    volume        BIGINT,
    amount_wan    NUMERIC(18, 2),        -- 成交额（万元）
    turnover_rate NUMERIC(10, 4),
    pe_ttm        NUMERIC(12, 4),-- 市盈率 TTM = 股价 / 近 12 个月每股收益
    circ_mv_yi    NUMERIC(18, 2),        -- 流通市值（亿元）= 股价 × 流通股本（市场上实际可交易部分）
    total_mv_yi   NUMERIC(18, 2),        -- 总市值（亿元）= 股价 × 总股本
    pb            NUMERIC(12, 4), -- 市净率 = 股价 / 每股净资产（银行利润波动大，PB 更稳定，银行股常用）
    fetched_at    TIMESTAMPTZ NOT NULL DEFAULT now() -- 写入时自动填当前时刻
);

COMMENT ON TABLE fin_quote_snapshot IS '腾讯行情实时快照，每次采集 UPSERT 覆盖';
-- 索引列顺序 (stock_code, metric, report_date)：匹配下游最常见查询模式
-- "某股、某指标、按时间排序取序列"（按查询模式反推索引设计）。fin_quotes 主键自带索引，无需额外建
CREATE INDEX IF NOT EXISTS idx_fin_financials_lookup
    ON fin_financials (stock_code, metric, report_date);
