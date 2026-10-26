# 金融企业主表文本流重建（保险/券商四大表无 pipe 语法，按科目/附注/数值物理行循环）
# - 按主表标题锚点切区间，多段（(续)/重印同标题）合并为一张逻辑表
# - 状态机识别 group/subitem/item/tail，4 行式（附注+两值）与 3 行式（两值）
# - 无法解析的行降级保留；整表数字行不足或无表尾则 degraded 并产出原文 residue
import re

import normalize_rules as R
from blocks_io import make_cell
from tables_markdown import STMT_CN

YEAR_EXTRACT_RE = re.compile(r"(20\d{2})")
TAIL_NOTE_TITLE = "财务报表附注"
MAX_NON_NUMERIC_RUN = 25


def _is_noise(text):
    # 区间内重印表头/元数据/签署语/主表标题本身/二级章节运行页眉
    if R.parse_statement(text):
        return True
    if R.HASH_SECTION_RE.match(text):
        return True
    return bool(
        R.NOTE_HEADER_RE.match(text) or R.DATE_HEADER_RE.match(text)
        or R.HALF_DATE_RE.match(text) or R.YEAR_LINE_RE.match(text)
        or R.SCOPE_LINE_RE.match(text) or R.UNIT_LINE_RE.search(text)
        or R.SIGN_OFF_RE.search(text)
    )


def _detect_year(texts):
    years = [int(y) for t in texts for y in YEAR_EXTRACT_RE.findall(t)]
    return max(years) if years else None


def _columns(kind, year):
    y = year or 2024
    if kind == "bs":
        return ["科目", "附注", f"{y}年12月31日", f"{y - 1}年12月31日"]
    if kind in ("is", "cf"):
        return ["科目", "附注", f"{y}年度", f"{y - 1}年度"]
    return ["科目", "附注", f"{y}年", f"{y - 1}年"]


def _group_anchors(anchors):
    # 同名(scope,kind)连续锚点归为一张表的多个段
    groups = []
    for a in anchors:
        scope = a["scope"] if a["scope"] in ("merge", "parent") else "merge"
        key = (scope, a["kind"])
        if groups and groups[-1]["key"] == key:
            groups[-1]["anchors"].append(a)
        else:
            groups.append({"key": key, "scope": scope, "kind": a["kind"],
                           "anchors": [a]})
    return groups


def _segment_bounds(group, first_anchor_idx, all_anchors, line_seq, gtext, gpage):
    # 每段 [start_gno, end_gno)；非末段以 下一锚点为界，末段跨组取下一首锚点，最后一张表按数字区收口
    bounds = []
    n = len(group["anchors"])
    next_group_idx = first_anchor_idx + n
    for k, a in enumerate(group["anchors"]):
        start = a["gno"]
        if k < n - 1:
            end = group["anchors"][k + 1]["gno"]
        elif next_group_idx < len(all_anchors):
            end = all_anchors[next_group_idx]["gno"]
        else:
            end = _find_table_end(start, line_seq, gtext)
        bounds.append((start, end, a))
    return bounds


def _find_table_end(start, line_seq, gtext):
    # 最后一张表收口：签署语/附注标题，或数字科目区结束后连续非数值行超阈值
    run = 0
    started = False
    for gno in line_seq:
        if gno <= start:
            continue
        t = gtext.get(gno, "")
        if R.SIGN_OFF_RE.search(t) or t.strip() == TAIL_NOTE_TITLE:
            return gno
        numeric = R.is_number_token(t)
        subject = R.classify_subject(t) if not _is_noise(t) else None
        if numeric or subject:
            started = True
            run = 0
        else:
            run += 1
            if started and run > MAX_NON_NUMERIC_RUN:
                return gno
    return line_seq[-1] + 1 if line_seq else start + 1


def _tokenize(start, end, line_seq, gmap, covered):
    # 区间行 -> subject/data token 序列；噪声行只计入覆盖、不入 token
    tokens = []
    residue_raw = []
    for gno in line_seq:
        if gno < start or gno >= end:
            continue
        e = gmap.get(gno)
        if e is None or e["kind"] != "line":
            continue
        covered.add(gno)
        text = e["text"]
        if not text:
            continue
        if _is_noise(text):
            continue
        residue_raw.append(text)
        cls = R.classify_subject(text)
        if cls:
            # 剥去 parser 给表内项目误加的 # 前缀（如 ### 一、营业收入），保留中文层级号
            subj_text = re.sub(r"^#+\s*", "", text)
            tokens.append({"kind": "subject", "cls": cls, "text": subj_text,
                           "page": e["page"], "gno": gno})
        elif R.is_number_token(text):
            tokens.append({"kind": "data", "text": text,
                           "page": e["page"], "gno": gno})
    return tokens, residue_raw


def _row_cells(subject, note, v1, v2):
    return [
        make_cell(subject),
        make_cell(note or "", note=bool(note)),
        make_cell(v1["text"] if v1 else "", value=R.normalize_number(v1["text"]) if v1 else None),
        make_cell(v2["text"] if v2 else "", value=R.normalize_number(v2["text"]) if v2 else None),
    ]


def _parse_tokens(tokens):
    # 状态机：subject 后收集至多 3 个连续 data，区分 4 行式/3 行式
    rows = []
    degraded_rows = 0
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        if tok["kind"] != "subject":
            i += 1
            continue
        cls, text, page = tok["cls"], tok["text"], tok["page"]
        if cls == "group":
            rows.append({"kind": "group", "page": page,
                         "cells": [make_cell(text), make_cell(""),
                                   make_cell(""), make_cell("")]})
            i += 1
            continue
        datas = []
        j = i + 1
        while j < n and tokens[j]["kind"] == "data" and len(datas) < 3:
            datas.append(tokens[j])
            j += 1
        note, v1, v2 = "", None, None
        ok = False
        if cls in ("item", "tail") and len(datas) >= 3 \
                and R.NOTE_TOKEN_RE.match(datas[0]["text"]) \
                and R.is_number_token(datas[1]["text"]):
            note, v1, v2 = datas[0]["text"], datas[1], datas[2]
            ok = True
        elif len(datas) >= 2 and R.is_number_token(datas[0]["text"]) \
                and R.is_number_token(datas[1]["text"]):
            v1, v2 = datas[0], datas[1]
            ok = True
        if ok:
            rows.append({"kind": cls, "page": page,
                         "cells": _row_cells(text, note, v1, v2)})
        else:
            rows.append({"kind": "degraded", "page": page,
                         "cells": [make_cell(text)] +
                                  [make_cell(d["text"]) for d in datas]})
            degraded_rows += 1
        i = j
    return rows, degraded_rows


def _dedup_groups(rows):
    # 跨段重印的同名分组标签（如续页顶部"负债"）相邻去重
    out = []
    for r in rows:
        if (r["kind"] == "group" and out and out[-1]["kind"] == "group"
                and out[-1]["cells"][0]["text"] == r["cells"][0]["text"]):
            continue
        out.append(r)
    return out


def parse_tables(elements, anchors, cfg):
    # 返回 (tables, covered_gnos, residues)
    gmap = {e["gno"]: e for e in elements}
    line_seq = [e["gno"] for e in elements if e["kind"] == "line"]
    line_seq.sort()
    gtext = {g: gmap[g]["text"] for g in line_seq}
    gpage = {g: gmap[g]["page"] for g in line_seq}

    covered = set()
    tables = []
    residues = []
    groups = _group_anchors(anchors)

    for gi, group in enumerate(groups):
        first_anchor_idx = sum(len(g["anchors"]) for g in groups[:gi])
        bounds = _segment_bounds(group, first_anchor_idx, anchors,
                                 line_seq, gtext, gpage)
        all_rows = []
        cont_anchors = []
        source_pages = []
        noise_texts = []
        residue_raw_all = []
        for k, (start, end, a) in enumerate(bounds):
            tokens, residue_raw = _tokenize(start, end, line_seq, gmap, covered)
            residue_raw_all.extend(residue_raw)
            rows, _ = _parse_tokens(tokens)
            all_rows.extend(rows)
            source_pages.append(a["page"])
            # 仅显式 (续) 锚点登记；同标题重印（无续标记）不登记
            if a["is_continuation"]:
                cont_anchors.append(f"{a['raw']}@p{a['page']}")
            # 收集噪声文本用于推断年度/单位
            for gno in line_seq:
                if start <= gno < end:
                    noise_texts.append(gtext.get(gno, ""))

        all_rows = _dedup_groups(all_rows)
        # source_pages 以实际解析行所在页为准（含无锚点的续页，如平安 BS 第157页）
        row_pages = sorted({r["page"] for r in all_rows} | set(source_pages))
        numeric_rows = [r for r in all_rows if r["kind"] in ("item", "tail", "subitem")
                        and any(c["value"] is not None for c in r["cells"][2:])]
        has_tail = any(r["kind"] == "tail" for r in all_rows)
        year = _detect_year(noise_texts)
        scope, kind = group["scope"], group["kind"]
        name = STMT_CN.get((scope, kind), f"{scope}_{kind}")
        cols = _columns(kind, year)
        degraded = len(numeric_rows) < cfg["textflow_min_numeric_rows"] or not has_tail

        table = {
            "name": name, "name_source": "statement_title", "source": "textflow",
            "is_main": True, "scope": scope, "kind": kind,
            "anchor_gno": group["anchors"][0]["gno"],
            "parse_status": "degraded" if degraded else "ok",
            "unit": "人民币" if kind == "bs" else "",
            "columns": cols, "header_rows": [cols], "multilevel_header": False,
            "rows": all_rows, "ncols": 4,
            "merged": len(bounds) > 1, "source_fragments": [],
            "source_pages": row_pages,
            "header_backfilled": len(bounds) > 1,
            "continuation_anchors": cont_anchors,
            "warnings": [] if not degraded else ["TEXTFLOW_LOW_CONFIDENCE"],
            "_order": group["anchors"][0]["gno"],
        }
        tables.append(table)
        if degraded:
            residues.append({
                "gno": group["anchors"][0]["gno"],
                "page": source_pages[0], "name": name, "lines": residue_raw_all,
            })

    tables.sort(key=lambda t: t["_order"])
    return tables, covered, residues
