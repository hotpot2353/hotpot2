# 年报 Markdown 清洗与元素化
# - 分离 front matter，按 parser 页边界切页并保留全局行号（含空行作为段落边界）
# - 删除页眉页脚（具名规则 + 重复公司名频次兜底 + 紧邻家具的孤立印刷页码）
# - 剔除目录（茅台型点引导线、平安型短标题↔页码，均用前瞻正文信号保护真实标题）
# - 元素化：pipe 表格碎片 vs 文本行；同步扫描四大主表标题锚点
import re

import normalize_rules as R

FRONT_MATTER_RE = re.compile(r"^---\s*$")
CAPTION_RE = re.compile(r"^\*\*表\s*(\d+)-(\d+)\*\*\s*$")
PIPE_RE = re.compile(r"^\|.*\|\s*$")
WINDOW = 3                      # 页首/页末家具判定位置窗口
REPEAT_RATIO = 0.30             # 无数字公司名在页面重复出现的删除阈值
TOC_LOOKAHEAD = 3               # 目录短行向后探测正文信号的窗口
PROSE_SIGNAL_LEN = 30           # 长句正文信号的最小长度


def read_markdown(md_path):
    # 分离 YAML front matter 与正文，返回 (meta_dict, body)
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    meta = {}
    if lines and FRONT_MATTER_RE.match(lines[0].strip()):
        for i in range(1, len(lines)):
            if FRONT_MATTER_RE.match(lines[i].strip()):
                body = "\n".join(lines[i + 1:])
                for kv in lines[1:i]:
                    if ":" in kv:
                        k, v = kv.split(":", 1)
                        meta[k.strip()] = v.strip().strip('"')
                return meta, body
        return meta, text
    return meta, text


def split_pages(body):
    # 返回展平行列表 [(gno, page_no, text)]，text 为空串表示段落空行；页标记本身不产出
    out = []
    page_no = 1
    for gno, raw in enumerate(body.splitlines()):
        text = raw.strip()
        m = R.PAGE_MARK_RE.match(text)
        if m:
            page_no = int(m.group(1))
            continue
        out.append((gno, page_no, text))
    return out


def _page_index(lines):
    # 按页归集全局行号，返回 {page_no: [gno,...]}
    pages = {}
    for gno, pno, _ in lines:
        pages.setdefault(pno, []).append(gno)
    return pages


def strip_furniture(lines):
    # 标记页眉页脚行，返回 (drop_set, removed_count)
    drop = set()
    pages = _page_index(lines)
    gtext = {gno: text for gno, _, text in lines}

    # 第一遍：具名家具（与位置无关的高置信规则）
    for gno, pno, text in lines:
        if not text:
            continue
        if (R.FOOTER_SLASH_RE.match(text) or R.FOOTER_DASH_RE.match(text)
                or R.INNER_PAGE_RE.match(text)
                or R.PINGAN_BOOK_RE.match(text) or R.ANNUAL_HEADER_RE.match(text)):
            drop.add(gno)

    # 公司名行：页首/页末窗口内，结尾粘连印刷页码的高置信删除
    name_no_digit = {}
    for pno, gnolist in pages.items():
        head = set(gnolist[:WINDOW])
        tail = set(gnolist[-WINDOW:])
        for gno in gnolist:
            text = gtext.get(gno, "")
            if not text or gno in drop:
                continue
            if R.COMPANY_TAIL_RE.match(text) and (gno in head or gno in tail):
                if re.search(r"\d{1,4}$", text):
                    drop.add(gno)            # 平安：公司名+粘连页码
                else:
                    name_no_digit.setdefault(text, set()).add(pno)  # 中信：纯公司名，频次兜底

    # 第二遍：在足够多页面重复出现的无数字公司名视为运行书眉
    total_pages = len(pages)
    repeated_names = {
        name for name, pset in name_no_digit.items()
        if len(pset) / max(total_pages, 1) >= REPEAT_RATIO
    }
    for name in repeated_names:
        for pno, gnolist in pages.items():
            zone = set(gnolist[:WINDOW]) | set(gnolist[-WINDOW:])
            for gno in gnolist:
                if gno in zone and gtext.get(gno) == name:
                    drop.add(gno)

    # 第三遍：紧邻内部页码/书眉/页边界的孤立印刷页码（避免误删表内数字）
    for pno, gnolist in pages.items():
        gset = set(gnolist)
        for idx, gno in enumerate(gnolist):
            text = gtext.get(gno, "")
            if not text or gno in drop or not R.BARE_PAGE_NO_RE.match(text):
                continue
            nearby = gnolist[max(0, idx - 2): idx + 3]
            if any((n in drop) for n in nearby if n != gno):
                drop.add(gno)

    return drop, len(drop)


def _is_prose_signal(text):
    # 长句正文信号：足够长且含中文句读
    return len(text) >= PROSE_SIGNAL_LEN and bool(re.search(r"[，。；！？]", text))


def remove_toc(lines, drop):
    # 在 drop 上追加目录行；返回 (新增删除数, toc_titles 去重列表)
    # 单次顺序扫描，多个「目录」起点均可激活；用短行前瞻正文信号保护真实标题
    gno_order = [gno for gno, _, _ in lines]
    gtext = {gno: text for gno, _, text in lines}
    pno_for_gno = {gno: pno for gno, pno, _ in lines}
    toc_titles = []
    added = 0

    def mark(gno):
        nonlocal added
        if gno not in drop:
            drop.add(gno)
            added += 1

    def lookahead(idx, limit):
        # 取当前位置之后至多 limit 个未删除非空行文本（pipe 表格/caption 不算正文信号，跳过）
        out = []
        j = idx + 1
        while j < len(gno_order) and len(out) < limit:
            g = gno_order[j]
            t = gtext.get(g, "")
            if t and g not in drop and not PIPE_RE.match(t) and not CAPTION_RE.match(t):
                out.append(t)
            j += 1
        return out

    active = False
    for idx, gno in enumerate(gno_order):
        text = gtext.get(gno, "")
        if R.TOC_TITLE_RE.match(text or ""):
            mark(gno)
            active = True
            continue
        # 两行式目录标题：「目」独占一行，下一未删行为「录」
        if text == "目":
            nxt_gno = gno_order[idx + 1] if idx + 1 < len(gno_order) else None
            if nxt_gno and gtext.get(nxt_gno, "") == "录":
                mark(gno)
                mark(nxt_gno)
                active = True
                continue
        # 无「目录」标题但首行即长引导线条目（如美的/工商银行）
        # TOC_LEADER_RE 已要求 ≥30 个点，足以排除正文省略号
        if not active and R.TOC_LEADER_RE.search(text or ""):
            mark(gno)
            active = True
            continue
        if not active or not text or gno in drop:
            continue
        if PIPE_RE.match(text) or CAPTION_RE.match(text):
            continue                         # 目录页夹带的表格保留
        if text.startswith("###") or text.startswith("####"):
            active = False                  # 正文 hash 小节标题
            continue
        if _is_prose_signal(text):
            active = False                  # 长句正文
            continue
        if R.TOC_LEADER_RE.search(text):
            mark(gno)                       # 茅台型点引导线条目
            continue
        if R.TOC_PAGE_RE.match(text):
            mark(gno)                       # 孤立页码
            continue
        if R.TOC_PAGE_RANGE_RE.match(text):
            mark(gno)                       # 页码范围（如 1-6 / 25-180 / A-1）
            continue
        # 目录辅助单字行（页/次/附/件）
        if text in ("页", "次", "附", "件"):
            mark(gno)
            continue
        if len(text) <= R.TOC_TITLE_MAX_LEN and not re.search(r"[。！？]", text):
            nxt = lookahead(idx, TOC_LOOKAHEAD)
            if any(_is_prose_signal(t) or t.startswith("###") for t in nxt):
                active = False              # 正文标题/正文开始
                continue
            mark(gno)
            toc_titles.append(re.sub(r"[\s　]+", "", text))
            continue
        active = False                      # 其余较长非正文信号行，保守退出
    return added, sorted(set(toc_titles))


def elementize(lines, drop):
    # 未删除行 -> 有序元素；同步识别主表标题锚点
    # 元素：{"kind":"line","order","gno","page","text"}
    #       {"kind":"frag","order","gno","page","frag_id","rows":[pipe 行]}
    elements = []
    anchors = []
    order = 0
    i = 0
    live = [(gno, pno, text) for gno, pno, text in lines if gno not in drop]
    while i < len(live):
        gno, pno, text = live[i]
        if not text:
            i += 1
            continue
        cap = CAPTION_RE.match(text)
        if cap:
            # 向后找首个非空行：是 pipe 表则聚合为碎片，否则 caption 降级普通行
            j = i + 1
            while j < len(live) and not live[j][2]:
                j += 1
            if j < len(live) and PIPE_RE.match(live[j][2]):
                frag_id = f"{cap.group(1)}-{cap.group(2)}"
                rows = []
                k = j
                while k < len(live) and live[k][2] and PIPE_RE.match(live[k][2]):
                    rows.append(live[k][2])
                    k += 1
                elements.append({"kind": "frag", "order": order, "gno": gno,
                                 "page": pno, "frag_id": frag_id, "rows": rows})
                order += 1
                i = k
                continue
        if PIPE_RE.match(text):
            # 无 caption 的 pipe 表（兜底）
            rows = []
            k = i
            while k < len(live) and live[k][2] and PIPE_RE.match(live[k][2]):
                rows.append(live[k][2])
                k += 1
            elements.append({"kind": "frag", "order": order, "gno": gno,
                             "page": pno, "frag_id": f"p{pno}-{gno}", "rows": rows})
            order += 1
            i = k
            continue
        elements.append({"kind": "line", "order": order, "gno": gno,
                         "page": pno, "text": text})
        parsed = R.parse_statement(text)
        if parsed:
            scope, kind, is_cont = parsed
            anchors.append({"order": order, "gno": gno, "page": pno,
                            "scope": scope, "kind": kind,
                            "is_continuation": is_cont, "raw": text})
        order += 1
        i += 1
    return elements, anchors
