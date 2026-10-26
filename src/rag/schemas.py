# 检索链数据结构：Chunk（子块）与 RetrievalResult（检索结果）
# P1 切块 / 入库 / 检索 / 评测共用；字段口径与 spec 对齐：
#   - Small-to-Big 父子索引：子块召回，命中后凭 parent_id 回取整节父块原文
#   - 元数据三件套 stock_code / year / section_path 支撑召回前过滤（杜绝跨股票召回污染）
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    # 检索子块（Small-to-Big 的小块）
    chunk_id: str       # 稳定唯一 ID：sha256("股票|年份|章节|角色|原文")，重跑切块产出同 ID，入库幂等键
    stock_code: str     # 股票代码，前置过滤维度
    year: int           # 年报年份，前置过滤维度
    section_path: str   # 章节面包屑（normalize 产物），可选过滤维度
    parent_id: str      # 父块 ID：命中子块后回取整节原文
    text: str           # 块原文；嵌入输入 = 上下文前缀 + 原文，前缀不落库存储


@dataclass(frozen=True)
class RetrievalResult:
    # 单条检索结果：块内容 + 排序信息
    chunk: Chunk
    score: float        # rerank 后得分；未开 rerank 时为 RRF 融合分
    source: str         # 命中来源标记：vector / bm25 / both，供评测分路归因
