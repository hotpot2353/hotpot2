# 章节树重建：三种节标题形态归一、运行页眉去重、小节/子条识别、平安目录兜底
# - 输出 heading 记录与每条文本行的 section_path 栈快照
# - 表格遮罩（masked_gnos）内的行不参与标题识别，规避金融利润表内「### 一、营业收入」伪标题
import re

import normalize_rules as R

FINANCIAL_FALLBACK_PATH = ["财务报告"]
GENERIC_FALLBACK_PATH = ["（正文）"]


def _norm(s):
    return re.sub(r"[\s　#]+", "", s or "")


def _excluded_toc_word(word):
    # 主表名、附注、审计报告、备查/补充资料不作为平安式章节 L2
    if R.parse_statement(word):
        return True
    return any(bad in word for bad in
               ("附注", "审计报告", "补充资料", "备查文件"))


def analyze(elements, anchors, toc_titles, masked_gnos):
    # 返回 headings / path_map / prose_gnos / consumed_gnos / running_dropped
    line_elems = [e for e in elements if e["kind"] == "line"]
    masked = masked_gnos if masked_gnos is not None else set()

    # 标准「第X节」结构存在时禁用平安式目录兜底，避免污染茅台/中信
    has_standard = False
    for e in line_elems:
        t = e["text"]
        if R.SECTION_KEY_RE.match(t) or R.HASH_SECTION_RE.match(t):
            has_standard = True
            break
    toc_set = {w for w in (toc_titles or []) if not _excluded_toc_word(w)}
    enable_toc = (not has_standard) and bool(toc_set)

    def by_order(gno_current):
        # 取下一个未被遮罩的 line 元素
        for e in line_elems:
            if e["gno"] > gno_current and e["gno"] not in masked:
                return e
        return None

    headings = []
    consumed = set()           # 两行式标题的标题行被消费，不再产 prose
    accepted_l2 = {}           # num -> {"norm","head_idx"}，L2 去重与断行补全
    accepted_toc = set()       # 平安目录标题全局只接受首次作为章节

    stack = []                 # [(level, num, display, detected_by)]

    def push_h(level, num, display, detected_by):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, num, display, detected_by))

    def snapshot():
        return [d for _, _, d, _ in stack], (stack[0][3] if stack else "fallback")

    def register_section(num, title, detected_by, gno, order, page):
        # L2 统一登记：同节首次接受、重复页眉丢弃、更长标题（断行补全）就地替换
        nn = _norm(title)
        disp = f"{num} {title}"
        if num in accepted_l2:
            old = accepted_l2[num]
            if nn == old["norm"]:
                running_dropped[0] += 1
                return "drop"
            if nn.startswith(old["norm"]) and len(nn) > len(old["norm"]):
                headings[old["head_idx"]].update({"title": title, "text": disp})
                for k, item in enumerate(stack):
                    if item[0] == 2 and item[1] == num:
                        stack[k] = (2, num, disp, detected_by)
                old["norm"] = nn
                return "replace"
            running_dropped[0] += 1
            return "drop"
        push_h(2, num, disp, detected_by)
        old_idx = len(headings)
        headings.append({"gno": gno, "order": order, "page": page,
                         "level": 2, "num": num, "title": title,
                         "text": disp, "detected_by": detected_by})
        accepted_l2[num] = {"norm": nn, "head_idx": old_idx}
        return "new"

    running_dropped = [0]
    path_map = {}
    prose_gnos = []

    for e in line_elems:
        gno, order, page, text = e["gno"], e["order"], e["page"], e["text"]
        is_heading_gno = False

        if gno not in masked:
            # 形态A：两行式「第X节」+ 下一行标题（标题在 PDF 断行时顺拼下短行）
            m_key = R.SECTION_KEY_RE.match(text)
            if m_key:
                nxt = by_order(gno)
                if nxt and not R.SECTION_KEY_RE.match(nxt["text"]) \
                        and not nxt["text"].startswith("#") and len(nxt["text"]) <= 40:
                    num = m_key.group(1)
                    title = nxt["text"].strip()
                    consumed.add(nxt["gno"])
                    extra = by_order(nxt["gno"])
                    if title[-1:] in ("和", "及", "与", "或", "的") and extra \
                            and not extra["text"].startswith("#") \
                            and not R.SECTION_KEY_RE.match(extra["text"]) \
                            and len(extra["text"]) <= 16 \
                            and not re.search(r"[，。；：、]", extra["text"]):
                        title += extra["text"].strip()
                        consumed.add(extra["gno"])
                    status = register_section(num, title, "two_line", gno, order, page)
                    is_heading_gno = True

            # 形态B：单行 hash 节标题（统一按节号去重/补全）
            if R.HASH_SECTION_RE.match(text):
                bare = re.sub(r"^#+\s*", "", text).strip()
                inline = R.SECTION_INLINE_RE.match(bare)
                if inline:
                    status = register_section(
                        inline.group(1), inline.group(2).strip(),
                        "hash_heading", gno, order, page)
                    is_heading_gno = True

            # L3：### 一、xxx
            m3 = R.HASH_L3_RE.match(text)
            if not is_heading_gno and m3:
                push_h(3, None, f"{m3.group(1)}、{m3.group(2).strip()}",
                       "hash_heading")
                headings.append({"gno": gno, "order": order, "page": page,
                                 "level": 3, "num": None,
                                 "title": m3.group(2).strip(),
                                 "text": f"{m3.group(1)}、{m3.group(2).strip()}",
                                 "detected_by": "hash_heading"})
                is_heading_gno = True

            # L4：#### （一）xxx 或遮罩外独占的虚拟（一）标题
            if not is_heading_gno and (R.HASH_L4_RE.match(text)
                                       or R.VIRTUAL_L4_RE.match(text)):
                disp = re.sub(r"^#{1,6}\s*", "", text).strip()
                push_h(4, None, disp, "hash_heading")
                headings.append({"gno": gno, "order": order, "page": page,
                                 "level": 4, "num": None, "title": disp,
                                 "text": disp, "detected_by": "hash_heading"})
                is_heading_gno = True

            # 平安式目录标题兜底：全局只接受首次（章首页），重复出现保留为正文
            if not is_heading_gno and enable_toc and _norm(text) in toc_set \
                    and len(text) >= 2 and _norm(text) not in accepted_toc:
                accepted_toc.add(_norm(text))
                push_h(2, None, text.strip(), "toc_title")
                headings.append({"gno": gno, "order": order, "page": page,
                                 "level": 2, "num": None, "title": text.strip(),
                                 "text": text.strip(), "detected_by": "toc_title"})
                is_heading_gno = True

        path, top_detected = snapshot()
        path_map[gno] = (path, top_detected)

        if gno not in masked and not is_heading_gno and gno not in consumed:
            prose_gnos.append(gno)

    return {
        "headings": headings,
        "path_map": path_map,
        "prose_gnos": prose_gnos,
        "consumed_gnos": consumed,
        "running_dropped": running_dropped[0],
        "enable_toc": enable_toc,
        "financial_fallback": FINANCIAL_FALLBACK_PATH,
        "generic_fallback": GENERIC_FALLBACK_PATH,
    }


def path_for_anchor(analysis, anchor_gno):
    # 主表块取其标题锚点所在章节栈；无 L2 时落「财务报告」
    path, _ = analysis["path_map"].get(anchor_gno, ([], "fallback"))
    return path if path else list(FINANCIAL_FALLBACK_PATH)
