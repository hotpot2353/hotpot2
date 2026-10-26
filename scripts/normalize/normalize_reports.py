# 年报规范化主入口：清洗/结构重建/表格重组/缝合 -> 结构化 blocks JSONL
# - 主表路由：锚点近页存在 pipe 碎片走 markdown 重组，否则走金融文本流状态机
# - 幂等：输入 md 的 sha256 + normalizer 版本，未变跳过；--force 重算
import argparse
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fetch"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parse"))
from common import PROJECT_ROOT, load_config, load_env, setup_logging, now_iso  # noqa: E402
from parse_reports import load_download_manifest, load_parsed_index  # noqa: E402

import clean_pages as C
import sections as S
import tables_markdown as TM
import tables_textflow as TT
import normalize_rules as R
from blocks_io import make_block, write_blocks, load_index, append_manifest

TABLE_PAYLOAD_KEYS = (
    "name", "name_source", "source", "is_main", "scope", "kind",
    "parse_status", "unit", "columns", "header_rows", "multilevel_header",
    "rows", "ncols", "merged", "source_fragments", "source_pages",
    "header_backfilled", "continuation_anchors", "warnings",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def route_anchors(anchors, fragments, cfg):
    # 主表锚点分组，近页有 pipe 碎片 -> markdown；否则 -> textflow
    groups = TT._group_anchors(anchors)
    gap = cfg["max_continuation_page_gap"] + 1
    md, tf = [], []
    for group in groups:
        first_page = group["anchors"][0]["page"]
        first_gno = group["anchors"][0]["gno"]
        has_pipe = any(
            f["page"] <= first_page + gap and f["gno"] > first_gno
            for f in fragments
            if first_page <= f["page"] <= first_page + gap
        )
        (md if has_pipe else tf).extend(group["anchors"])
    return md, tf


def _is_bare_number(text):
    # 孤立数字行（序号/页码/附注号）不参与折行拼接，避免「56 杨利强」类误拼
    return R.is_number_token(text.strip())


# 跨页缝合时，上段若以这些字符结尾视为语义完整（日期/百分比/金额单位/页码等）
_SEW_STOP_CHARS = R.SENTENCE_END_CHARS | set("年月日%页号元万亿佰仟拾0123456789》")
_SEW_NEXT_TITLE_RE = re.compile(
    r"^(?:第[一二三四五六七八九十百0-9]+节|[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）|20\d{2}\s*年)")


def _sewable(prev, nxt):
    # 仅当上段是被截断的汉字半句、下段以汉字且非标题/日期开头时才跨页缝合
    p, n = prev.strip(), nxt.strip()
    if not p or not n or _is_bare_number(p):
        return False
    # 防御：- N - 格式页脚不参与缝合
    if R.FOOTER_DASH_RE.match(p) or R.FOOTER_DASH_RE.match(n):
        return False
    if p[-1] in _SEW_STOP_CHARS:
        return False
    if not re.match(r"^[一-龥]", n) or n.startswith("#"):
        return False
    return not _SEW_NEXT_TITLE_RE.match(n)


def build_prose_segments(prose_gnos, gmap, path_map, generic_path):
    # 物理相邻行折行拼接；跨页且上段未结束则缝合，返回 prose block 描述列表
    segs = []
    cur = None

    def flush():
        nonlocal cur
        if cur and cur["text"].strip():
            segs.append(cur)
        cur = None

    for gno in prose_gnos:
        e = gmap[gno]
        text, page = e["text"], e["page"]
        if cur is None:
            cur = {"gno": gno, "text": text, "p_start": page, "p_end": page,
                   "sewn": False, "seam_pages": [], "last_gno": gno, "last_page": page}
            continue
        adjacent_same_page = (page == cur["last_page"] and gno == cur["last_gno"] + 1)
        cross_page = page != cur["last_page"]
        # 孤立数字行（序号/页码）不与相邻文本折行拼接
        if adjacent_same_page and not _is_bare_number(cur["text"]) \
                and not _is_bare_number(text):
            cur["text"] += text
        elif cross_page and _sewable(cur["text"], text):
            cur["text"] += text
            cur["sewn"] = True
            if page not in cur["seam_pages"]:
                cur["seam_pages"].append(page)
            cur["p_end"] = page
        else:
            flush()
            cur = {"gno": gno, "text": text, "p_start": page, "p_end": page,
                   "sewn": False, "seam_pages": [], "last_gno": gno, "last_page": page}
            continue
        cur["last_gno"] = gno
        cur["last_page"] = page
        if page > cur["p_end"]:
            cur["p_end"] = page
    flush()

    for s in segs:
        path, detected = path_map.get(s["gno"], ([], "fallback"))
        s["path"] = path or list(generic_path)
        s["detected"] = detected
    return segs


def normalize_document(md_path, doc_rec, cfg):
    meta, body = C.read_markdown(md_path)
    lines = C.split_pages(body)
    drop, n_furniture = C.strip_furniture(lines)
    n_toc, toc_titles = C.remove_toc(lines, drop)
    elements, anchors = C.elementize(lines, drop)
    fragments = [e for e in elements if e["kind"] == "frag"]
    gmap = {e["gno"]: e for e in elements}

    # 主表路由与两路表格重组
    md_anchors, tf_anchors = route_anchors(anchors, fragments, cfg)
    md_tables = TM.rebuild(fragments, md_anchors, cfg)
    tf_tables, covered, residues = TT.parse_tables(elements, tf_anchors, cfg) \
        if tf_anchors else ([], set(), [])
    all_tables = md_tables + tf_tables

    # 章节树（文本流主表行作为遮罩，规避伪 ### 标题）
    analysis = S.analyze(elements, anchors, toc_titles, covered)
    path_map = analysis["path_map"]
    generic_path = analysis["generic_fallback"]

    items = []

    def add(sort_gno, p_start, p_end, path, detected, btype, payload):
        items.append((sort_gno, p_start, p_end, list(path), detected, btype, payload))

    # heading
    for h in analysis["headings"]:
        path, _ = path_map[h["gno"]]
        add(h["gno"], h["page"], h["page"], path or generic_path, h["detected_by"],
            "heading", {"level": h["level"], "num": h["num"],
                        "title": h["title"], "text": h["text"]})

    # prose（含跨页缝合）
    for s in build_prose_segments(analysis["prose_gnos"], gmap, path_map, generic_path):
        add(s["gno"], s["p_start"], s["p_end"], s["path"], s["detected"], "prose",
            {"text": s["text"], "sewn": s["sewn"], "seam_pages": s["seam_pages"],
             "degraded": False, "raw_residue": None})

    # 文本流主表降级时的原文兜底 prose（零丢失）
    for r in residues:
        add(r["gno"], r["page"], r["page"], S.FINANCIAL_FALLBACK_PATH, "fallback",
            "prose", {"text": "", "sewn": False, "seam_pages": [],
                      "degraded": True, "raw_residue": r["lines"]})

    # tables
    # 构建 gno -> 元素的快速查找，用于普通表格向前回溯最近 line 元素的章节
    gno_to_elem = {e["gno"]: e for e in elements}
    line_gnos_sorted = [e["gno"] for e in elements if e["kind"] == "line"]
    line_gnos_sorted.sort()

    def _nearest_line_path(gno):
        # 找 gno 之前最近的 line 元素，返回其 section_path；找不到返回 fallback
        import bisect
        idx = bisect.bisect_left(line_gnos_sorted, gno) - 1
        while idx >= 0:
            lg = line_gnos_sorted[idx]
            p, d = path_map.get(lg, ([], "fallback"))
            if p:
                return p, d
            idx -= 1
        return [], "fallback"

    for t in all_tables:
        pages = t["source_pages"]
        p_start, p_end = (min(pages), max(pages)) if pages else (None, None)
        if t.get("anchor_gno") is not None:
            path = S.path_for_anchor(analysis, t["anchor_gno"])
            detected = path_map.get(t["anchor_gno"], ([], "fallback"))[1]
        else:
            path, detected = _nearest_line_path(t["_order"])
            path = path or generic_path
        payload = {k: t[k] for k in TABLE_PAYLOAD_KEYS if k in t}
        add(t["_order"], p_start, p_end, path or S.FINANCIAL_FALLBACK_PATH,
            detected, "table", payload)

    items.sort(key=lambda x: x[0])
    doc_id = doc_rec["doc_id"]
    blocks = []
    for seq, (_, p_start, p_end, path, detected, btype, payload) in enumerate(items):
        blocks.append(make_block(doc_id, seq, btype, p_start, p_end,
                                 path, payload, detected))

    # 统计
    n_main = sum(1 for t in all_tables if t.get("is_main"))
    degraded_main_strict = sum(
        1 for t in all_tables if t.get("is_main")
        and t["parse_status"] == "degraded" and t.get("kind") in ("bs", "is", "cf"))
    degraded_eq = sum(
        1 for t in all_tables if t.get("is_main")
        and t["parse_status"] == "degraded" and t.get("kind") == "eq")
    n_warnings = sum(len(t.get("warnings", [])) for t in all_tables)
    fallback_paths = sum(1 for g, (p, d) in path_map.items() if not p)
    stats = {
        "block_count": len(blocks),
        "heading_count": sum(1 for b in blocks if b["type"] == "heading"),
        "prose_count": sum(1 for b in blocks if b["type"] == "prose"),
        "table_count": len(all_tables),
        "main_table_count": n_main,
        "degraded_table_count": sum(1 for t in all_tables
                                   if t["parse_status"] == "degraded"),
        "degraded_main_bs_is_cf": degraded_main_strict,
        "degraded_main_eq": degraded_eq,
        "sewn_prose_count": sum(1 for b in blocks
                                if b["type"] == "prose" and b.get("sewn")),
        "toc_lines_removed": n_toc,
        "furniture_lines_removed": n_furniture,
        "running_headers_dropped": analysis["running_dropped"],
        "warnings_count": n_warnings,
        "fallback_path_lines": fallback_paths,
    }
    return blocks, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", nargs="*", help="只处理指定股票代码")
    parser.add_argument("--years", nargs="*", type=int, help="只处理指定年份")
    parser.add_argument("--force", action="store_true", help="忽略幂等强制重算")
    args = parser.parse_args()

    load_env()
    cfg_all = load_config()
    cfg = cfg_all["normalize"]
    log = setup_logging()

    meta_dir = PROJECT_ROOT / cfg_all["paths"]["parsed_metadata"]
    blocks_root = PROJECT_ROOT / cfg_all["paths"]["processed_blocks"]
    manifest_path = blocks_root / "normalize_manifest.jsonl"

    docs = load_download_manifest(meta_dir / "manifest.jsonl")
    parsed_index = load_parsed_index(meta_dir / "parsed_manifest.jsonl")
    prior_index = load_index(manifest_path)

    for doc_id, rec in docs.items():
        if args.codes and rec["stock_code"] not in args.codes:
            continue
        if args.years and rec["year"] not in args.years:
            continue
        parsed = parsed_index.get(doc_id)
        if not parsed or parsed.get("status") != "parsed":
            continue

        md_path = PROJECT_ROOT / parsed["markdown_path"]
        if not md_path.exists():
            log.warning("Markdown 缺失，跳过 %s", doc_id)
            continue
        digest = sha256_file(md_path)
        out_path = blocks_root / rec["stock_code"] / f"{rec['year']}_blocks.jsonl"
        prior = prior_index.get(doc_id)
        if not args.force and prior and prior.get("source_sha256") == digest \
                and prior.get("normalizer_version") == cfg["normalizer_version"] \
                and out_path.exists():
            log.info("未变化，跳过 %s", doc_id)
            continue

        blocks, stats = normalize_document(md_path, rec, cfg)
        write_blocks(blocks, out_path)

        record = {
            "doc_id": doc_id, "stock_code": rec["stock_code"], "year": rec["year"],
            "source_markdown_path": parsed["markdown_path"],
            "source_sha256": digest,
            "normalizer_version": cfg["normalizer_version"],
            "blocks_out": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "normalized_at": now_iso(),
            "status": "normalized",
        }
        record.update(stats)
        append_manifest(manifest_path, record)
        log.info("已规范化 %s | block=%s 表=%s(主表%s,降级BSISCF=%s,EQ=%s) 缝合=%s",
                 doc_id, stats["block_count"], stats["table_count"],
                 stats["main_table_count"], stats["degraded_main_bs_is_cf"],
                 stats["degraded_main_eq"], stats["sewn_prose_count"])


if __name__ == "__main__":
    main()
