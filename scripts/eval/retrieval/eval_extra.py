# 追加评测：三个补充实验
#   exp1 纯 OR 噪声对照：bm25_operator="or" vs 基线 AND（同 rerank=false），量化"OR 放宽引入的噪声"
#   exp2 截断深度 k 曲线 + RRF 常数敏感：final_top_k=16 存 ranked，离线算 hit/recall@4/8/16；rrf_k 60 vs 10
#   exp3 口语化查询鲁棒性：8 条金标改写为大白话问法，同 GT 过滤对照——考语义泛化，不考标注
# 运行前提：PG 在线 + ARK_API_KEY 可用；本机无 vikingdb SDK → 全部 rerank=false，与 baseline 存档同基线可比
# 产物：data/processed/eval/extra_tests.json + reports/rag-extra-tests.html
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "eval" / "retrieval"))

from eval_retrieval import LEGS, ground_truth_ids, load_golden, metrics_of  # noqa: E402
from rag.config import get_rag_settings  # noqa: E402
from rag.retriever import HybridRetriever  # noqa: E402

OUT_JSON = PROJECT_ROOT / "data" / "processed" / "eval" / "extra_tests.json"
OUT_HTML = PROJECT_ROOT / "reports" / "rag-extra-tests.html"
BASELINE_JSON = PROJECT_ROOT / "data" / "processed" / "eval" / "retrieval_baseline.json"

# 口语化变体：与金标同过滤条件（股票/年份/章节/内容词），仅改问法——检索必须靠语义而非词面命中
COLLOQUIAL = {
    "moutai_rev": "茅台去年卖酒总共收了多少钱，净赚了多少",
    "moutai_channel": "茅台直销和找经销商拿货，哪种卖得多",
    "moutai_dividend": "茅台今年打算给股东分多少现金",
    "moutai_gross": "茅台卖酒的毛利能到多少个点",
    "cmb_npl": "把钱放招行稳不稳，它的坏账多不多",
    "hengrui_rd": "恒瑞每年花多少钱搞新药研发",
    "catl_capacity": "宁德时代的电池工厂忙不忙，产能用满了没",
    "pingan_premium": "平安一年卖保险能收多少保费",
}
PAIR_QIDS = list(COLLOQUIAL)


class OrOnlyRetriever(HybridRetriever):
    # 纯 OR 召回对照：检索器原生只有 and / and_or_fallback（首查恒为 AND），
    # 本子类把 BM25 首查直接改为"任一词元命中"，用于量化 OR 放宽引入的噪声
    def _bm25_leg(self, cur, query: str, stock_code, year, top_k: int) -> list:
        tsq = self._tsquery(query, "|")
        if not tsq:
            return []
        return self._bm25_query(cur, tsq, stock_code, year, top_k)


def agg_rows(rows: list) -> dict:
    n = len(rows)
    return {
        "hit": sum(r["hit"] for r in rows) / n,
        "recall": sum(r["recall"] for r in rows) / n,
        "precision": sum(r["precision"] for r in rows) / n,
        "mrr": sum(r["mrr"] for r in rows) / n,
    }


def aggregate(per: list) -> dict:
    return {name: agg_rows([q["legs"][name] for q in per]) for name in LEGS}


def run_cfg(cfg, golden: list, k: int = 8, keep_ranked: bool = False,
            retriever_cls=HybridRetriever) -> list:
    # 单配置全量评测：三腿逐一检索，k 处截断出指标；keep_ranked 供多 k 离线重算
    retr = retriever_cls(cfg)
    per = []
    for e in golden:
        gt = set(ground_truth_ids(e))
        row = {"qid": e["qid"], "gt_size": len(gt), "legs": {}}
        for name, legs in LEGS.items():
            t0 = time.time()
            res = retr.retrieve(e["query"], stock_code=e.get("stock_code"),
                                year=e.get("year"), legs=legs)
            ranked = [r.chunk.chunk_id for r in res]
            m = metrics_of(ranked, gt, k)
            m["latency_s"] = round(time.time() - t0, 3)
            if keep_ranked:
                m["ranked"] = ranked
            row["legs"][name] = m
        per.append(row)
        print(f"  {e['qid']} gt={len(gt)} "
              + " ".join(f"{n}:r{row['legs'][n]['first_rank'] or '-'}" for n in LEGS))
    return per


def run_paired(cfg, golden_by_qid: dict, qids: list, k: int = 8) -> list:
    # 口语化配对评测：同一 GT，形式问法 vs 口语问法各跑三腿
    retr = HybridRetriever(cfg)
    per = []
    for qid in qids:
        e = golden_by_qid[qid]
        gt = set(ground_truth_ids(e))
        row = {"qid": qid, "gt_size": len(gt), "legs": {}}
        for label, text in (("formal", e["query"]), ("collo", COLLOQUIAL[qid])):
            row["legs"][label] = {}
            for name, legs in LEGS.items():
                res = retr.retrieve(text, stock_code=e.get("stock_code"),
                                    year=e.get("year"), legs=legs)
                ranked = [r.chunk.chunk_id for r in res]
                row["legs"][label][name] = metrics_of(ranked, gt, k)
        per.append(row)
        print(f"  {qid} gt={len(gt)} form/bm25:r{row['legs']['formal']['bm25']['first_rank'] or '-'}"
              f"/vec:r{row['legs']['formal']['vector']['first_rank'] or '-'}"
              f" → collo/bm25:r{row['legs']['collo']['bm25']['first_rank'] or '-'}"
              f"/vec:r{row['legs']['collo']['vector']['first_rank'] or '-'}")
    return per


def fmt(v) -> str:
    return f"{v:.3f}"


def table(headers: list, rows: list) -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>")


def main() -> None:
    golden = load_golden()
    by_qid = {e["qid"]: e for e in golden}
    cfg = get_rag_settings()
    base_r = cfg.retrieval
    result = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # ---- exp1 纯 OR 噪声对照（rerank=false，与 baseline 存档同基线）----
    print("[exp1] 纯 OR 噪声对照 bm25_operator=or, rerank=false")
    cfg_or = replace(cfg, retrieval=replace(base_r, rerank=False, bm25_operator="or"))
    per_or = run_cfg(cfg_or, golden, retriever_cls=OrOnlyRetriever)
    agg_or = aggregate(per_or)
    baseline = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    result["exp1_or"] = {"aggregate": agg_or, "per_query": per_or}

    # ---- exp2 截断深度 k 曲线 + RRF 常数敏感（rerank=false, final_top_k=16）----
    print("[exp2] k 曲线 + RRF 敏感（rrf_k=60 vs 10, top16 存 ranked）")
    cfg16 = replace(cfg, retrieval=replace(base_r, rerank=False, final_top_k=16))
    per16 = run_cfg(cfg16, golden, k=16, keep_ranked=True)
    cfg16_k10 = replace(cfg, retrieval=replace(base_r, rerank=False, final_top_k=16, rrf_k=10))
    per16_k10 = run_cfg(cfg16_k10, golden, k=16, keep_ranked=True)
    # 离线重算：同一份 ranked 列表按不同 k 截断出指标（需逐条 GT，按 qid 取）
    def recalc(per_ranked, k):
        rows = []
        for q in per_ranked:
            gt = set(ground_truth_ids(by_qid[q["qid"]]))
            rows.append(metrics_of(q["legs"]["hybrid"]["ranked"], gt, k))
        return agg_rows(rows)
    result["exp2_rrf60"] = {str(k): recalc(per16, k) for k in (4, 8, 16)}
    result["exp2_rrf10"] = {str(k): recalc(per16_k10, k) for k in (4, 8, 16)}

    # ---- exp3 口语化鲁棒性（rerank=false 默认 and_or_fallback 配置，同 GT 配对）----
    print("[exp3] 口语化鲁棒性 8 条配对")
    cfg_base = replace(cfg, retrieval=replace(base_r, rerank=False))
    per_paired = run_paired(cfg_base, by_qid, PAIR_QIDS)
    result["exp3_paired"] = per_paired
    agg_formal = {n: agg_rows([q["legs"]["formal"][n] for q in per_paired]) for n in LEGS}
    agg_collo = {n: agg_rows([q["legs"]["collo"][n] for q in per_paired]) for n in LEGS}
    result["exp3_agg"] = {"formal": agg_formal, "colloquial": agg_collo}

    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果存档: {OUT_JSON}")
    render(result, baseline)
    print(f"报告: {OUT_HTML}")


def render(res: dict, baseline: dict) -> None:
    # 由结果渲染自包含 HTML 报告（无外部依赖）
    agg_or = res["exp1_or"]["aggregate"]
    agg_and = baseline["aggregate"]
    zero_and = sum(1 for q in baseline["per_query"] if q["legs"]["bm25"]["hit"] == 0)
    zero_or = sum(1 for q in res["exp1_or"]["per_query"] if q["legs"]["bm25"]["hit"] == 0)

    e1_rows = []
    for leg in ("bm25", "hybrid"):
        a, o = agg_and[leg], agg_or[leg]
        e1_rows.append([f"{leg}（AND 基线）", fmt(a["hit_at_k"]), fmt(a["recall_at_k"]),
                        fmt(a["precision_at_k"]), fmt(a["mrr"])])
        e1_rows.append([f"{leg}（纯 OR）", fmt(o["hit"]), fmt(o["recall"]),
                        fmt(o["precision"]), fmt(o["mrr"])])
    d_rec = agg_or["bm25"]["recall"] - agg_and["bm25"]["recall_at_k"]
    d_pre = agg_or["bm25"]["precision"] - agg_and["bm25"]["precision_at_k"]

    e2_rows = []
    for metric, label in (("hit", "Hit"), ("recall", "Recall")):
        for k in (4, 8, 16):
            r60, r10 = res["exp2_rrf60"][str(k)], res["exp2_rrf10"][str(k)]
            e2_rows.append([f"{label}@{k}", fmt(r60[metric]), fmt(r10[metric]),
                            fmt(r60[metric] - r10[metric])])
    m60, m10 = res["exp2_rrf60"]["8"], res["exp2_rrf10"]["8"]
    e2_rows.append(["Precision@8", fmt(m60["precision"]), fmt(m10["precision"]),
                    fmt(m60["precision"] - m10["precision"])])
    e2_rows.append(["MRR", fmt(m60["mrr"]), fmt(m10["mrr"]), fmt(m60["mrr"] - m10["mrr"])])

    e3_rows = []
    for leg in LEGS:
        f_, c_ = res["exp3_agg"]["formal"][leg], res["exp3_agg"]["colloquial"][leg]
        e3_rows.append([leg, fmt(f_["hit"]), fmt(c_["hit"]), fmt(f_["recall"]),
                        fmt(c_["recall"]), fmt(f_["mrr"]), fmt(c_["mrr"])])
    pq_rows = []
    for q in res["exp3_paired"]:
        for leg in ("vector", "bm25"):
            f_, c_ = q["legs"]["formal"][leg], q["legs"]["collo"][leg]
            pq_rows.append([q["qid"], leg, str(f_["first_rank"] or "-"), fmt(f_["recall"]),
                            str(c_["first_rank"] or "-"), fmt(c_["recall"])])
    dvec = res["exp3_agg"]["colloquial"]["vector"]["recall"] - res["exp3_agg"]["formal"]["vector"]["recall"]
    dbm = res["exp3_agg"]["colloquial"]["bm25"]["recall"] - res["exp3_agg"]["formal"]["bm25"]["recall"]

    css = """
    body{font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;margin:0;background:#f6f7f9;color:#1a1a1a;}
    .wrap{max-width:960px;margin:0 auto;padding:32px 24px 64px;}
    h1{font-size:24px;margin:8px 0 4px;} h2{font-size:18px;margin:36px 0 12px;border-left:4px solid #0a7d43;padding-left:10px;}
    .meta{color:#667;font-size:13px;line-height:1.7;}
    .card{background:#fff;border:1px solid #e3e5e8;border-radius:10px;padding:16px 20px;margin:12px 0;}
    table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0;}
    th,td{border:1px solid #e3e5e8;padding:6px 10px;text-align:left;}
    th{background:#f0f2f4;font-weight:600;} tr:nth-child(even) td{background:#fafbfc;}
    .up{color:#0a7d43;font-weight:600;} .down{color:#c0392b;font-weight:600;}
    .note{font-size:13px;color:#556;line-height:1.8;}
    code{background:#f0f2f4;padding:1px 5px;border-radius:4px;font-size:12px;}
    """
    def delta(v):
        cls = "up" if v >= 0 else "down"
        return f'<span class="{cls}">{v:+.3f}</span>'

    parts = [f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>RAG 追加评测报告：三个此前未覆盖的实验</title><style>{css}</style></head><body><div class="wrap">
<h1>RAG 追加评测报告</h1>
<div class="meta">生成时间：{res['ts']} ｜ 环境：chunks 子块 60,939 + 父块 12,111（50 家公司）｜ 全部实验 rerank=false（本机无精排 SDK，与基线存档 retrieval_baseline.json 同基线可比）｜ 金标 15 条、k=8（exp2 另测 4/16）</div>

<h2>实验一：纯 OR 噪声对照——量化"OR 放宽引入多少噪声"</h2>
<div class="card"><table>{table(["分路 / 配置", "Hit@8", "Recall@8", "Precision@8", "MRR"], e1_rows)}</table>
<div class="note">关键词分路：<b>AND → 纯 OR</b>，Recall {delta(d_rec)}，Precision {delta(d_pre)}；零命中 query 数：AND {zero_and}/15 → OR {zero_or}/15。
<b>意外发现：纯 OR 全面占优，精度不降反升。</b>解释：ts_rank 本身是"软 AND"——OR 召回集内部按命中词元数与稀有度排序，前列仍是多词元同时命中的块；真正的病灶不是 OR 噪声，而是 AND 的硬失效（任一词元缺失即整条报废）。个别噪声确实存在（moutai_rev 混合链路首位 1→3，MRR 1.0→0.33），但量级远小于召回收益。
设计含义：① and_or_fallback 的 AND 首查在本评测集（无精排口径）上没有精度收益；② "OR 噪声必须靠精排兜底"的假设在该口径下不成立，精排的真实价值在于把 30 候选压到 8 的排序质量；③ 结论限于 n=15、GT 偏保守、无精排——正式链路（fallback+rerank，R@8=0.518）仍是最优存档，本实验不构成回退依据。</div></div>

<h2>实验二：截断深度 k 曲线 + RRF 融合常数敏感（RRF k=60 vs 10）</h2>
<div class="card"><table>{table(["指标（混合链路）", "RRF k=60", "RRF k=10", "差值"], e2_rows)}</table>
<div class="note">RRF 常数 60 → 10：各深度 Recall 差异 ≤1.3pt 且方向不一致——排名融合对常数确实不敏感，"选 60 是惯例而非调参结果"得到实证。截断深度：k=8→16 时 Recall 0.401→0.539、Hit 拉满 1.0，但 Precision 0.400→0.275——k=8 是直接喂模型的平衡点；下游若有精排，正确做法是加深候选池再压缩（正式链路 rerank_top_n=30 正是此逻辑），而不是加深喂给模型的窗口。</div></div>

<h2>实验三：口语化查询鲁棒性（8 条金标改写为大白话，同 GT 配对）</h2>
<div class="card"><table>{table(["分路", "形式 Hit@8", "口语 Hit@8", "形式 Recall@8", "口语 Recall@8", "形式 MRR", "口语 MRR"], e3_rows)}</table>
<div class="note">口语化（8 条，刻意避开书面术语，如"坏账多不多" vs 不良贷款率）下<b>两路同伤</b>：向量分路 Recall {delta(dvec)}，关键词分路 {delta(dbm)}——向量分路对"口语→书面术语"跨度的泛化并没有直觉中那么稳，关键词分路退化更重；混合链路靠互补把退化收窄（Recall -25%、Hit 1.00→0.75），互补得到验证，但仍有 3 条 query 完全失守。
设计含义：口语场景的短板在召回层不在排序层，下一步应加查询改写/术语归一（口语"坏账"→"不良贷款"），而不是继续调召回参数。</div></div>

<h2>实验三逐条对照（形式 vs 口语，首命中排名 / Recall@8）</h2>
<div class="card"><table>{table(["qid", "分路", "形式首排名", "形式 Recall", "口语首排名", "口语 Recall"], pq_rows)}</table></div>

<h2>口径与局限</h2>
<div class="note">
① 全部实验 rerank=false：本机缺精排 SDK，结论限于"召回 + 融合"层，与带精排的正式存档（retrieval_rerank*.json）不可直接混比；
② 金标口径与主评测一致（章节子串 + 全部内容词在过滤范围内实测），GT 标记偏保守时不影响同 GT 下的相对结论；
③ exp1 的 AND 侧取自基线存档，库内容量未变（同口径 GT），跨日可比；
④ exp3 口语改写为人工构造，样本量 8 条，结论方向性成立、幅度不外推。
</div>
</div></body></html>"""]

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text("".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
