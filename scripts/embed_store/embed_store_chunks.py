# 嵌入+入库编排：chunks JSONL → 方舟嵌入 → jieba 分词 → PGVector chunks 表（离线预建管道）
# 输入：data/processed/chunks/<code>/<year>_chunks.jsonl（切块产物，按 chunks_manifest 最新记录过滤版本）
# 输出：chunks 表行（child 带 embedding+content_tsv 双检索腿原料，parent 纯净）、
#       embed_store_manifest.jsonl 账目、全部灌完后执行 rag_indexes.sql 建 HNSW/GIN 索引
# 幂等与断点续传（三层）：
#   1. 写入幂等：chunk_id 内容哈希主键 + ON CONFLICT (chunk_id) DO NOTHING
#   2. 嵌入幂等：入库前查 (stock_code, year) 已有 chunk_id，已入库子块跳过嵌入——
#      断点粒度=单条，长任务中断重跑只补缺口，不重复花嵌入费
#   3. 账目：embed_store_manifest.jsonl 逐份登记（写入/跳过/tokens），追加式
# 事务边界：每批写入一个 commit；单份失败跳过汇总（对齐采集惯例），已 commit 批次保留，重跑补缺
# 索引时机：完整跑且零失败才自动建——空表/半表先建 HNSW 会拖慢后续灌数数倍（rag_indexes.sql 头注释同口径）
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "normalize"))

import jieba  # noqa: E402

from blocks_io import append_manifest  # noqa: E402
from core.logger import get_logger  # noqa: E402
from db.connection import connect  # noqa: E402
from llm.embedding_client import get_embedding_client  # noqa: E402
from rag.config import get_rag_settings  # noqa: E402

logger = get_logger(__name__)

CHUNKS_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"
MANIFEST_PATH = CHUNKS_DIR / "chunks_manifest.jsonl"
STORE_MANIFEST_PATH = CHUNKS_DIR / "embed_store_manifest.jsonl"

# 子块批大小：嵌入是花钱的网络请求，批内失败（重试耗尽）损失上限=一批
BATCH_CHILD = 50
# parent 无嵌入开销，批量纯写
BATCH_PARENT = 500

INSERT_SQL = (
    "INSERT INTO chunks "
    "(chunk_id, stock_code, year, section_path, chunk_role, parent_id, "
    " content, content_tsv, embedding, embedding_model, page_start, page_end) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, "
    " to_tsvector('simple', %s), %s::halfvec, %s, %s, %s) "
    "ON CONFLICT (chunk_id) DO NOTHING"
)


def load_latest_manifest() -> dict:
    # doc_id → 最新记录（追加式 manifest 同 doc 多条取最后一条，与切块/normalize 同口径）
    latest = {}
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            latest[rec["doc_id"]] = rec
    return latest


def fetch_existing_ids(cur, code: str, year: int) -> set:
    # 库中该报已有 chunk_id：断点续传依据，已入库的子块不再嵌入
    cur.execute(
        "SELECT chunk_id FROM chunks WHERE stock_code = %s AND year = %s",
        (code, year),
    )
    return {r[0] for r in cur.fetchall()}


def tokenize(text: str) -> str:
    # jieba 搜索引擎模式分词 → 空格拼接；PG simple 配置按空白切词元建 tsvector
    # 查询侧（检索腿）须用同一分词口径，保证词元对齐可命中
    return " ".join(jieba.cut_for_search(text))


def parent_row(c: dict) -> tuple:
    # parent 纯净行：不带任何检索原料（契约 chk_chunks_role 锁死）
    return (c["chunk_id"], c["stock_code"], c["year"], " / ".join(c["section_path"]),
            "parent", None, c["content"], None, None, None,
            c.get("page_start"), c.get("page_end"))


def child_row(c: dict, vec: list, model: str) -> tuple:
    # child 全套检索原料行：jieba 词袋 + 向量（文本形态传入，PG 按目标列解析）+ 模型版本
    return (c["chunk_id"], c["stock_code"], c["year"], " / ".join(c["section_path"]),
            "child", c["parent_id"], c["content"],
            tokenize(c["content"]), str(vec), model,
            c.get("page_start"), c.get("page_end"))


def write_batch(cur, rows: list) -> None:
    cur.executemany(INSERT_SQL, rows)


def process_document(conn, chunks_path: Path, code: str, year: int,
                     client, model: str) -> dict:
    # 单份年报：查库补缺——parent 先批量落库（无嵌入开销），child 逐批嵌入+写入
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    tokens_before = client.total_prompt_tokens
    n_written = n_skipped = 0
    with conn.cursor() as cur:
        existing = fetch_existing_ids(cur, code, year)
        for c in chunks:
            if c["chunk_id"] in existing:
                n_skipped += 1

        pending_parents = [c for c in chunks
                           if c["chunk_role"] == "parent" and c["chunk_id"] not in existing]
        for i in range(0, len(pending_parents), BATCH_PARENT):
            write_batch(cur, [parent_row(c) for c in pending_parents[i:i + BATCH_PARENT]])
            conn.commit()
        n_written += len(pending_parents)

        pending_children = [c for c in chunks
                            if c["chunk_role"] == "child" and c["chunk_id"] not in existing]
        for i in range(0, len(pending_children), BATCH_CHILD):
            batch = pending_children[i:i + BATCH_CHILD]
            # 嵌入输入 = 上下文前缀 + 原文（前缀不落库）；失败即抛，本份标记失败，
            # 已 commit 批次保留，重跑按查库补缺
            texts = [c.get("embed_input") or c["content"] for c in batch]
            vectors = client.embed_texts(texts)
            write_batch(cur, [child_row(c, v, model) for c, v in zip(batch, vectors)])
            conn.commit()
            n_written += len(batch)

    return {
        "n_written": n_written,
        "n_skipped": n_skipped,
        "prompt_tokens": client.total_prompt_tokens - tokens_before,
    }


def create_indexes() -> None:
    # 全量灌完后执行重索引 DDL（HNSW + GIN，幂等 IF NOT EXISTS 可重复执行）
    sql = (PROJECT_ROOT / "src" / "db" / "rag_indexes.sql").read_text(encoding="utf-8")
    t0 = time.time()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
    logger.info("检索索引创建完成: 耗时 %.1fs", time.time() - t0)


def main() -> None:
    parser = argparse.ArgumentParser(description="嵌入+入库编排（chunks → PGVector）")
    parser.add_argument("--doc", help="只处理指定股票代码，如 600519")
    parser.add_argument("--limit", type=int, help="最多处理份数（试跑用）")
    parser.add_argument("--indexes", action="store_true", help="只创建 HNSW/GIN 索引")
    parser.add_argument("--skip-indexes", action="store_true", help="灌数后不建索引")
    args = parser.parse_args()

    if args.indexes:
        create_indexes()
        return

    rag = get_rag_settings()
    model = rag.embedding.model

    # 跨版本过滤：manifest 追加式含历史 chunker 版本，只入库当前版本产物
    latest = load_latest_manifest()
    targets = {doc_id: rec for doc_id, rec in latest.items()
               if rec.get("chunker_version") == rag.chunking.chunker_version}
    if args.doc:
        targets = {k: v for k, v in targets.items() if k.startswith(args.doc + "_")}
    docs = sorted(targets.values(), key=lambda r: r["doc_id"])
    if args.limit:
        docs = docs[:args.limit]
    if not docs:
        logger.warning("无可处理文档: manifest 最新记录 %s 份, chunker=%s",
                       len(latest), rag.chunking.chunker_version)
        return

    client = get_embedding_client(model, rag.embedding.dimension)
    jieba.initialize()

    t0 = time.time()
    processed = failed = 0
    total_written = total_skipped = total_tokens = 0
    failed_docs = []
    for rec in docs:
        doc_id = rec["doc_id"]
        code = doc_id.split("_")[0]
        year = int(doc_id.split("_")[1])
        chunks_path = PROJECT_ROOT / rec["out_path"]
        try:
            # 每份年报独立连接：故障天然隔离，单份失败不影响后续
            with connect() as conn:
                st = process_document(conn, chunks_path, code, year, client, model)
            processed += 1
            total_written += st["n_written"]
            total_skipped += st["n_skipped"]
            total_tokens += st["prompt_tokens"]
            append_manifest(STORE_MANIFEST_PATH, {
                "doc_id": doc_id,
                "chunker_version": rag.chunking.chunker_version,
                "embedding_model": model,
                "n_written": st["n_written"],
                "n_skipped": st["n_skipped"],
                "prompt_tokens": st["prompt_tokens"],
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            logger.info("入库完成 %s: 写入=%s 跳过=%s tokens=%s 累计耗时=%.0fs",
                        doc_id, st["n_written"], st["n_skipped"],
                        st["prompt_tokens"], time.time() - t0)
        except Exception as exc:
            failed += 1
            failed_docs.append(doc_id)
            logger.error("入库失败 %s: %s", doc_id, exc)

    logger.info(
        "入库结束: 处理=%s 失败=%s | 写入=%s 跳过=%s tokens=%s 耗时=%.0fs (model=%s)",
        processed, failed, total_written, total_skipped, total_tokens,
        time.time() - t0, model)
    if failed_docs:
        logger.error("失败清单: %s", ", ".join(failed_docs))

    # 索引：完整跑且零失败才自动建；试跑（--doc/--limit）或有失败时跳过，
    # 半量数据建索引浪费且拖慢后续灌数，待全量完成后用 --indexes 补建
    full_run = not (args.limit or args.doc)
    if not args.skip_indexes and full_run and failed == 0:
        create_indexes()
        # 账目核对：库内行数 vs manifest 期望（父+子）
        expected = sum(r.get("n_parent", 0) + r.get("n_child", 0) for r in docs)
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM chunks")
                db_rows = cur.fetchone()[0]
                cur.execute(
                    "SELECT count(*) FROM chunks "
                    "WHERE chunk_role = 'child' AND embedding IS NOT NULL")
                db_child = cur.fetchone()[0]
                cur.execute(
                    "SELECT count(*) FROM chunks "
                    "WHERE chunk_role = 'parent' AND embedding IS NULL")
                db_parent = cur.fetchone()[0]
        logger.info("账目核对: 库内=%s 期望=%s | child带向量=%s parent纯净=%s",
                    db_rows, expected, db_child, db_parent)
    elif not args.skip_indexes:
        logger.info("跳过索引创建（试跑或存在失败），全量完成后执行 --indexes 补建")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
