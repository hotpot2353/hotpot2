# 检索效果评测：三腿（vector / bm25 / hybrid）对比 Recall@k / Precision@k / MRR / Hit@k
# 金标集 golden_queries.yaml；GT=过滤范围内满足章节子串+全部内容关键词的 child 块集合（库内实测）
# 口径：三腿共用同一 GT，指标可横向对比；G 标记不全时 Precision 绝对值偏保守，不影响腿间相对结论
# 产物：data/processed/eval/retrieval_baseline.json（rerank A/B 对照的基线存档）
# 用法：--validate 仅校验金标（gt_size>=1）；默认全量评测
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.logger import get_logger  # noqa: E402
from db.connection import connect  # noqa: E402
from rag.config import get_rag_settings  # noqa: E402
from rag.retriever import HybridRetriever  # noqa: E402

logger = get_logger(__name__)

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_queries.yaml"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "eval"

# 评测腿：vector-only / bm25-only / hybrid；同一 GT 下横向对比
LEGS = {
    "vector": ("vector",),
    "bm25": ("bm25",),
    "hybrid": ("vector", "bm25"),
}


def load_golden() -> list:
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["queries"]


def ground_truth_ids(entry: dict) -> list:
    # GT：过滤范围内 section_contains 与全部 text_contains 同时命中的 child 块
    conds = ["chunk_role = 'child'"]
    params = []
    if entry.get("stock_code"):
        conds.append("stock_code = %s")
        params.append(entry["stock_code"])
    if entry.get("year"):
        conds.append("year = %s")
        params.append(entry["year"])
    if entry.get("section_contains"):
        conds.append("section_path ILIKE %s")
        params.append(f"%{entry['section_contains']}%")
    for kw in entry.get("text_contains") or []:
        conds.append("content LIKE %s")
        params.append(f"%{kw}%")
    sql = "SELECT chunk_id FROM chunks WHERE " + " AND ".join(conds)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [r[0] for r in cur.fetchall()]


def metrics_of(ranked_ids: list, gt: set, k: int) -> dict:
    # ranked_ids：检索返回 chunk_id 有序列表（全量，截断在指标处做）
    top = ranked_ids[:k]
    n_hit_top = len(set(top) & gt)
    hit_rank = next((i for i, cid in enumerate(ranked_ids, 1) if cid in gt), None)
    return {
        "recall": n_hit_top / len(gt) if gt else 0.0,
        "precision": n_hit_top / k,
        "mrr": 1.0 / hit_rank if hit_rank else 0.0,
        "hit": 1 if hit_rank and hit_rank <= k else 0,
        "first_rank": hit_rank or 0,
    }


def validate(golden: list) -> bool:
    # 金标校验：gt_size>=1 才有效；同时提示过宽标记（>50）
    ok = True
    for e in golden:
        gt = ground_truth_ids(e)
        n = len(gt)
        flag = "OK " if 1 <= n <= 50 else ("EMPTY" if n == 0 else "WIDE ")
        if n == 0:
            ok = False
        stocks = ""
        if not e.get("stock_code"):
            with connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT DISTINCT stock_code FROM chunks WHERE chunk_id = ANY(%s)",
                        (gt,))
                    stocks = ",".join(sorted(r[0] for r in cur.fetchall()))
        print(f"  [{flag}] {e['qid']}: gt={n} {stocks}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="检索效果评测（三腿对比）")
    parser.add_argument("--validate", action="store_true", help="仅校验金标集")
    parser.add_argument("--no-instruction", action="store_true",
                        help="rerank 指令 A/B：置空 rerank_instruction 走服务端默认")
    args = parser.parse_args()

    golden = load_golden()
    if args.validate:
        sys.exit(0 if validate(golden) else 1)

    cfg = get_rag_settings()
    if args.no_instruction:
        # 冻结 dataclass 不原地改，replace 生成置空指令的副本
        cfg = replace(cfg, retrieval=replace(cfg.retrieval, rerank_instruction=""))
    k = cfg.retrieval.final_top_k
    retriever = HybridRetriever(cfg)

    per_query = []
    detail_rows = []
    for e in golden:
        gt = set(ground_truth_ids(e))
        if not gt:
            logger.warning("金标为空跳过: %s", e["qid"])
            continue
        row = {"qid": e["qid"], "gt_size": len(gt), "legs": {}}
        line = f"{e['qid']} gt={len(gt)}"
        for leg_name, legs in LEGS.items():
            t0 = time.time()
            res = retriever.retrieve(
                e["query"], stock_code=e.get("stock_code"), year=e.get("year"),
                legs=legs)
            elapsed = time.time() - t0
            ranked = [r.chunk.chunk_id for r in res]
            m = metrics_of(ranked, gt, k)
            m["latency_s"] = round(elapsed, 3)
            row["legs"][leg_name] = m
            line += f" | {leg_name}: r{m['first_rank'] or '-'} R{m['recall']:.2f} M{m['mrr']:.2f}"
        per_query.append(row)
        detail_rows.append(line)
        print(line)

    # 聚合：三腿各自均值
    agg = {}
    for leg_name in LEGS:
        rows = [q["legs"][leg_name] for q in per_query]
        n = len(rows)
        agg[leg_name] = {
            "hit_at_k": sum(r["hit"] for r in rows) / n,
            "recall_at_k": sum(r["recall"] for r in rows) / n,
            "precision_at_k": sum(r["precision"] for r in rows) / n,
            "mrr": sum(r["mrr"] for r in rows) / n,
            "avg_latency_s": sum(r["latency_s"] for r in rows) / n,
        }

    print(f"\n{'腿':<8}{'Hit@'+str(k):<10}{'Recall@'+str(k):<12}{'Precision@'+str(k):<15}{'MRR':<10}{'延迟s':<8}")
    for leg_name, m in agg.items():
        print(f"{leg_name:<8}{m['hit_at_k']:<10.3f}{m['recall_at_k']:<12.3f}"
              f"{m['precision_at_k']:<15.3f}{m['mrr']:<10.3f}{m['avg_latency_s']:<8.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 存档按参数组合分流：retrieval_{baseline|rerank}[_andfb].json，A/B 可复现对照
    name = "retrieval_rerank" if cfg.retrieval.rerank else "retrieval_baseline"
    if cfg.retrieval.bm25_operator == "and_or_fallback":
        name += "_andfb"
    if args.no_instruction:
        name += "_noinstr"
    out_path = OUT_DIR / f"{name}.json"
    out_path.write_text(json.dumps({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "k": k,
        "rerank": cfg.retrieval.rerank,
        "rerank_model": cfg.retrieval.rerank_model,
        "rerank_instruction": bool(cfg.retrieval.rerank_instruction),
        "bm25_operator": cfg.retrieval.bm25_operator,
        "n_queries": len(per_query),
        "aggregate": agg,
        "per_query": per_query,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n基线存档: {out_path}")


if __name__ == "__main__":
    main()
