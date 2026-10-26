# 工商企业 pipe 表格重组：碎片解析、表头判定、跨页逻辑表分组与列头回填
# 决策顺序：精确主表表头(项目+年度) → 强续表绑定在途表 → 普通表宽松表头独立成表
import re

import normalize_rules as R
from blocks_io import make_cell

# 主表标准中文名（未知主体按合并处理）
STMT_CN = {
    ("merge", "bs"): "合并资产负债表",
    ("parent", "bs"): "母公司资产负债表",
    ("merge", "is"): "合并利润表",
    ("parent", "is"): "母公司利润表",
    ("merge", "cf"): "合并现金流量表",
    ("parent", "cf"): "母公司现金流量表",
    ("merge", "eq"): "合并所有者权益变动表",
    ("parent", "eq"): "母公司所有者权益变动表",
}
HEADER_KEYWORDS = ("项目", "科目", "报表项目")
SEPARATOR_CELL_RE = re.compile(r"^:?-{2,}:?$")
YEAR_CELL_RE = re.compile(r"20\d{2}\s*年")


def _split_pipe(line):
    # | a | b | -> ["a","b"]
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _is_separator_row(cells):
    return bool(cells) and all(SEPARATOR_CELL_RE.match(c.replace(" ", "")) for c in cells if c != "")


def _clean_cell(c):
    # ColN 占位还原为空；含数字的单元格压缩内部空白，其余保留原文
    c = c.strip()
    if re.fullmatch(r"Col\d+", c):
        return ""
    return re.sub(r"\s+", "", c) if re.search(r"\d", c) else c


def parse_fragment(frag):
    # 去分隔行后的内容行 + 列数 + 单位串
    content_rows = []
    for line in frag["rows"]:
        cells = [_clean_cell(c) for c in _split_pipe(line)]
        if not _is_separator_row(cells):
            content_rows.append(cells)
    ncols = max((len(r) for r in content_rows), default=0)
    unit = ""
    for row in content_rows:
        for c in row:
            if re.search(r"单位[：:]", c):
                unit = c
                break
        if unit:
            break
    return {"frag_id": frag["frag_id"], "page": frag["page"], "gno": frag["gno"],
            "content_rows": content_rows, "ncols": ncols, "unit": unit}


def _row_all_text(row):
    nonempty = [c for c in row if c]
    return bool(nonempty) and all(R.normalize_number(c) is None for c in nonempty)


def _has_number_row(rows, start):
    return any(any(R.normalize_number(c) is not None for c in r)
               for r in rows[start + 1:])


def exact_header_idx(rows):
    # 主表式真表头：首格为项目类关键词且行内含年度列
    for i, row in enumerate(rows):
        c0 = row[0] if row else ""
        if c0 in HEADER_KEYWORDS and any(YEAR_CELL_RE.search(c) for c in row[1:]):
            return i
    return -1


def loose_header_idx(rows):
    # 普通表表头：整行无数值、非空格≥2、且下方存在数字数据行
    for i, row in enumerate(rows):
        nonempty = [c for c in row if c]
        if len(nonempty) >= 2 and _row_all_text(row) and _has_number_row(rows, i):
            return i
    return -1


def _classify_row(c0):
    t = (c0 or "").strip()
    if R.is_tail_subject(t):
        return "tail"
    if t in R.GROUP_EXACT or R.GROUP_COLON_RE.match(t):
        return "group"
    if R.SUBITEM_PREFIX_RE.match(t):
        return "subitem"
    if re.search(r"(合计|总计|小计)$", t):
        return "subtotal"
    return "item"


def _build_cells(text_cells, columns, ncols):
    # 对齐到 ncols；附注列打 note，数值列去千分位
    cells = []
    for i in range(ncols):
        raw = text_cells[i] if i < len(text_cells) else ""
        colname = columns[i] if i < len(columns) else ""
        note = ("附注" in colname) and bool(re.fullmatch(r"\d{1,3}", raw))
        value = None if note else R.normalize_number(raw)
        cells.append(make_cell(raw, value=value, note=note))
    return cells


def _new_table(name, name_source, parsed, header_rows, is_main, anchor, scope, kind):
    data_start = 0
    if header_rows:
        # header_rows 为内容行索引，数据行取最后一个表头行之后
        data_start = max(header_rows) + 1
    columns = list(parsed["content_rows"][header_rows[-1]]) if header_rows else \
        [f"col{i}" for i in range(parsed["ncols"])]
    table = {
        "name": name, "name_source": name_source, "source": "markdown",
        "is_main": is_main, "scope": scope, "kind": kind,
        "anchor_gno": anchor, "parse_status": "ok",
        "unit": parsed["unit"], "columns": columns,
        "header_rows": [list(parsed["content_rows"][i]) for i in header_rows],
        "multilevel_header": len(header_rows) > 1,
        "rows": [], "ncols": len(columns), "merged": False,
        "source_fragments": [parsed["frag_id"]], "source_pages": [parsed["page"]],
        "header_backfilled": False, "continuation_anchors": [],
        "warnings": [], "seen_tail": False, "subjects": set(),
        "_order": parsed["gno"],
    }
    _append_rows(table, parsed["content_rows"][data_start:], parsed["page"])
    return table


def _append_rows(table, data_rows, page):
    for text_cells in data_rows:
        c0 = text_cells[0] if text_cells else ""
        if not any(text_cells):
            continue
        kind = _classify_row(c0)
        table["rows"].append({
            "kind": kind, "page": page,
            "cells": _build_cells(text_cells, table["columns"], table["ncols"]),
        })
        if c0:
            table["subjects"].add(c0)
        if kind == "tail":
            table["seen_tail"] = True


def _bind_score(table, parsed, armed, cfg):
    # 续表碎片绑定在途表的打分
    score = 0
    if table["ncols"] == parsed["ncols"]:
        score += 2
    last_page = table["source_pages"][-1]
    if parsed["page"] > last_page:
        score += 1
    if 0 <= parsed["page"] - last_page <= cfg["max_continuation_page_gap"]:
        score += 1
    if not table["seen_tail"]:
        score += 2
    first_subject = next((r[0] for r in parsed["content_rows"] if r), "")
    if first_subject and first_subject not in table["subjects"]:
        score += 1
    if armed and table.get("kind") == armed[1] and table.get("scope") in (armed[0], "unknown"):
        score += 1
    return score


def rebuild(fragments, anchors, cfg):
    # 返回逻辑表列表（按起始位置排序）
    parsed_list = [parse_fragment(f) for f in fragments]
    anchor_idx = 0
    armed = None                            # (scope, kind, gno, page, consumed)
    tables = []
    active_idx = []

    def apply_anchors(before_gno):
        nonlocal armed, anchor_idx
        while anchor_idx < len(anchors) and anchors[anchor_idx]["gno"] < before_gno:
            a = anchors[anchor_idx]
            if not a["is_continuation"]:
                armed = (a["scope"], a["kind"], a["gno"], a["page"], False)
            anchor_idx += 1

    for parsed in parsed_list:
        apply_anchors(parsed["gno"])
        rows = parsed["content_rows"]
        exact = exact_header_idx(rows)

        if exact >= 0:
            # 精确主表表头：吸收其上方 col0 为空且无数值的多级表头行
            idxs = [exact]
            j = exact - 1
            # 向上吸收多级表头：col0 为空、整行无数值，且排除单位/币种伪行
            while j >= 0 and (not rows[j] or rows[j][0] == "") \
                    and _row_all_text(rows[j]) \
                    and not re.search(r"单位[：:]|币种", "".join(rows[j])):
                idxs.insert(0, j)
                j -= 1
            name, is_main, anchor_gno, scope, kind = None, False, None, None, None
            if armed and not armed[4] and \
                    0 <= parsed["page"] - armed[3] <= cfg["armed_title_ttl_pages"]:
                scope_key = armed[0] if armed[0] in ("merge", "parent") else "merge"
                name = STMT_CN[(scope_key, armed[1])]
                is_main, anchor_gno, scope, kind = True, armed[2], scope_key, armed[1]
                armed = (armed[0], armed[1], armed[2], armed[3], True)
            if not name:
                first_subject = rows[exact][0]
                name = first_subject if first_subject else f"表 {parsed['frag_id']}"
            new_idx = len(tables)
            tables.append(_new_table(name, "statement_title" if is_main else "caption",
                                     parsed, idxs, is_main, anchor_gno, scope, kind))
            if is_main:
                # 仅主表参与跨页续表绑定；新主表开启前关闭已见表尾的旧主表
                active_idx = [i for i in active_idx if not tables[i]["seen_tail"]]
                active_idx.append(new_idx)
            continue

        # 无精确表头：仅对在途主表做续表绑定（普通表默认独立，避免链式误并）
        best_i, best_s = None, cfg["bind_score_threshold"] - 1
        for i in active_idx:
            if not tables[i]["is_main"]:
                continue
            s = _bind_score(tables[i], parsed, armed, cfg)
            if s > best_s:
                best_s, best_i = s, i
        if best_i is not None:
            t = tables[best_i]
            t["source_fragments"].append(parsed["frag_id"])
            t["source_pages"].append(parsed["page"])
            t["merged"] = True
            t["header_backfilled"] = True
            _append_rows(t, rows, parsed["page"])
            continue

        # 独立普通表：宽松表头，表头缺失才占位
        loose = loose_header_idx(rows)
        if loose >= 0:
            name = rows[loose][0] or f"表 {parsed['frag_id']}"
            tables.append(_new_table(name, "caption", parsed, [loose],
                                     False, None, None, None))
        else:
            t = _new_table(f"表 {parsed['frag_id']}", "fragment_id", parsed, [],
                           False, None, None, None)
            t["warnings"].append("NO_HEADER_STANDALONE")
            tables.append(t)

    tables.sort(key=lambda t: t["_order"])
    return tables
