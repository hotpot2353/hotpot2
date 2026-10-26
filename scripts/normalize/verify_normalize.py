# 三样本规范化结果断言校验：茅台(markdown 续表)/平安(保险文本流)/中信(券商文本流)
# 任一断言失败退出码 1；--strict-main-tables 要求 BS/IS/CF 主表零降级（权益矩阵表除外）
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fetch"))
from common import PROJECT_ROOT, load_config  # noqa: E402

import clean_pages as C
import normalize_rules as R
from blocks_io import load_blocks, load_index

MAIN_NAMES_8 = (
    "合并资产负债表", "母公司资产负债表", "合并利润表", "母公司利润表",
    "合并现金流量表", "母公司现金流量表",
    "合并所有者权益变动表", "母公司所有者权益变动表",
)
MAIN_NAMES_FIN = (
    "合并资产负债表", "母公司资产负债表", "合并利润表", "母公司利润表",
    "合并现金流量表", "母公司现金流量表",
    "合并所有者权益变动表", "母公司所有者权益变动表",
)


class Report:
    def __init__(self):
        self.results = []

    def check(self, name, ok, detail=""):
        self.results.append((name, bool(ok), detail))

    @property
    def passed(self):
        return all(ok for _, ok, _ in self.results)


def find_table(blocks, name):
    for b in blocks:
        if b["type"] == "table" and b.get("name") == name:
            return b
    return None


def row_of(table, subject):
    for r in table.get("rows", []):
        if r["cells"] and r["cells"][0]["text"] == subject:
            return r
    return None


def cell_value(table, subject, col):
    r = row_of(table, subject)
    return r["cells"][col]["value"] if r else None


def load_case(code, year, blocks_root):
    blocks = load_blocks(blocks_root / code / f"{year}_blocks.jsonl")
    md = PROJECT_ROOT / "data" / "parsed" / "markdown" / code / f"{year}_annual.md"
    meta, _ = C.read_markdown(md)
    return blocks, int(meta.get("page_count", "0") or 0)


def check_common(rep, code, blocks, page_count):
    tag = f"[{code}]"
    # 家具残留：页脚精确为「N/总页数」；平安书眉仅在短块整段相等时判；内部页码标题
    footer_re = re.compile(rf"(?<!\d)\d{{1,3}}\s*/\s*{page_count}(?!\d)")
    furniture_hits = 0
    for b in blocks:
        if b["type"] not in ("heading", "prose") or b.get("degraded"):
            continue
        text = b.get("text", "")
        if footer_re.search(text):
            furniture_hits += 1
        if text.strip() == "二零二四年年报":
            furniture_hits += 1
        if b["type"] == "heading" and R.INNER_PAGE_RE.match(text):
            furniture_hits += 1
    rep.check(f"{tag} 页眉页脚残留=0", furniture_hits == 0, f"hits={furniture_hits}")

    # 目录块清零
    toc_hits = sum(
        1 for b in blocks
        if (b["type"] == "heading" and b.get("text") == "目录")
        or (b["type"] == "prose" and R.TOC_LEADER_RE.search(b.get("text", "")))
    )
    rep.check(f"{tag} 目录块=0", toc_hits == 0, f"hits={toc_hits}")

    # section_path 非空
    empty_path = sum(1 for b in blocks if not b.get("section_path"))
    rep.check(f"{tag} section_path 非空率100%", empty_path == 0, f"empty={empty_path}")

    # 页号合法
    bad_page = sum(1 for b in blocks
                   if b["page_start"] is None or b["page_end"] is None
                   or b["page_start"] > b["page_end"]
                   or b["page_start"] < 1 or b["page_end"] > page_count)
    rep.check(f"{tag} 页号合法", bad_page == 0, f"bad={bad_page}")

    # seq 连续、block_id 唯一
    seqs = [b["seq"] for b in blocks]
    ids = [b["block_id"] for b in blocks]
    rep.check(f"{tag} seq 连续且 block_id 唯一",
              seqs == list(range(len(blocks))) and len(set(ids)) == len(ids))


def check_moutai(rep, blocks):
    main_tables = [find_table(blocks, n) for n in MAIN_NAMES_8]
    rep.check("[600519] 8 张主表全部 ok",
              all(t and t["parse_status"] == "ok" for t in main_tables))

    bs = find_table(blocks, "合并资产负债表")
    rep.check("[600519] 合并BS碎片[58-1..61-1]",
              bs and bs["source_fragments"] == ["58-1", "59-1", "60-1", "61-1"],
              str(bs and bs["source_fragments"]))
    rep.check("[600519] 合并BS页58-61且回填列头",
              bs and bs["source_pages"] == [58, 59, 60, 61]
              and bs["header_backfilled"] and len(bs["header_rows"]) == 1)
    rep.check("[600519] 货币资金 59295822956.89/69070136376.12 附注1",
              cell_value(bs, "货币资金", 2) == "59295822956.89"
              and cell_value(bs, "货币资金", 3) == "69070136376.12"
              and row_of(bs, "货币资金")["cells"][1].get("note"))
    rep.check("[600519] 拆出资金(续页首行) 127187293298.17",
              cell_value(bs, "拆出资金", 2) == "127187293298.17")
    rep.check("[600519] 资产总计 298944579918.70",
              cell_value(bs, "资产总计", 2) == "298944579918.70")
    rep.check("[600519] 负债权益总计=资产总计",
              cell_value(bs, "负债和所有者权益总计", 2) == "298944579918.70"
              or any(r["cells"][0]["text"].startswith("负债和所有者权益")
                     and r["cells"][2]["value"] == "298944579918.70"
                     for r in bs["rows"]))

    pbs = find_table(blocks, "母公司资产负债表")
    rep.check("[600519] 母公司BS页61-63(61-1未误绑)",
              pbs and pbs["source_pages"] == [61, 62, 63],
              str(pbs and pbs["source_pages"]))

    sewn_text = "".join(b.get("text", "") for b in blocks
                        if b["type"] == "prose" and b.get("sewn"))
    rep.check("[600519] 跨页缝合串「直销和批发代理渠道进行销售」",
              "直销和批发代理渠道进行销售" in sewn_text)


def check_pingan(rep, blocks):
    bs = find_table(blocks, "合并资产负债表")
    rep.check("[601318] 合并BS为文本流/页155-157/一套列头",
              bs and bs["source"] == "textflow"
              and bs["source_pages"] == [155, 156, 157]
              and len(bs["header_rows"]) == 1
              and bs["continuation_anchors"] == [],
              str(bs and (bs["source"], bs["source_pages"])))
    rep.check("[601318] 货币资金 附注1 683433/577212",
              cell_value(bs, "货币资金", 2) == "683433"
              and row_of(bs, "货币资金")["cells"][1].get("note"))
    rep.check("[601318] 应收账款无附注 36006/35636",
              cell_value(bs, "应收账款", 2) == "36006"
              and cell_value(bs, "应收账款", 3) == "35636"
              and not row_of(bs, "应收账款")["cells"][1].get("note"))
    rep.check("[601318] 保险合同资产 空/3",
              cell_value(bs, "保险合同资产", 2) is None
              and cell_value(bs, "保险合同资产", 3) == "3")
    rep.check("[601318] 资产总计=负债股东权益总计 12957827",
              cell_value(bs, "资产总计", 2) == "12957827"
              and cell_value(bs, "负债和股东权益总计", 2) == "12957827")
    grp = row_of(bs, "金融投资：")
    rep.check("[601318] 金融投资为无值分组行",
              grp and grp["kind"] == "group"
              and all(c["value"] is None for c in grp["cells"][1:]))

    # 噪声 token 不得进入任何单元格科目
    noise = ("2024年12月31日", "附注八", "附注五", "二零二四年年报", "本集团")
    bad = []
    for b in blocks:
        if b["type"] != "table":
            continue
        for r in b.get("rows", []):
            for c in r["cells"]:
                if c["text"] in noise:
                    bad.append(c["text"])
    rep.check("[601318] 表内噪声token清零", not bad, f"bad={sorted(set(bad))}")

    headings_text = [b.get("text") for b in blocks if b["type"] == "heading"]
    rep.check("[601318] 「营业收入」未成为章节标题",
              not any(t in ("一、营业收入", "营业收入") for t in headings_text))


def check_citic(rep, blocks):
    bs = find_table(blocks, "合并资产负债表")
    rep.check("[600030] 合并BS页163-165/2个续锚/一套列头",
              bs and bs["source_pages"] == [163, 164, 165]
              and len(bs["continuation_anchors"]) == 2
              and len(bs["header_rows"]) == 1,
              str(bs and (bs["source_pages"], bs["continuation_anchors"])))
    rep.check("[600030] 货币资金 370646431306.03",
              cell_value(bs, "货币资金", 2) == "370646431306.03")
    rep.check("[600030] 客户资金存款子项 266597397073.76",
              (lambda r: r and r["kind"] == "subitem"
               and r["cells"][2]["value"] == "266597397073.76")(row_of(bs, "其中：客户资金存款")))
    rep.check("[600030] 资产总计 1710710828343.76",
              cell_value(bs, "资产总计", 2) == "1710710828343.76")
    rep.check("[600030] 负债合计 1411944156810.95",
              cell_value(bs, "负债合计", 2) == "1411944156810.95")
    rep.check("[600030] 负债股东权益总计=资产总计",
              cell_value(bs, "负债和股东权益总计", 2) == "1710710828343.76")

    # group 标签不重复（相邻同名已去重）
    groups = [r["cells"][0]["text"] for r in bs["rows"] if r["kind"] == "group"]
    rep.check("[600030] 相邻group标签不重复",
              all(groups[i] != groups[i + 1] for i in range(len(groups) - 1)),
              str(groups))

    h10 = [b for b in blocks if b["type"] == "heading"
           and b.get("num") == "第十节"]
    rep.check("[600030] 第十节运行页眉仅保留1处", len(h10) == 1, f"n={len(h10)}")
    inner = [b for b in blocks if b["type"] == "heading"
             and R.INNER_PAGE_RE.match(b.get("text", ""))]
    rep.check("[600030] 内部页码无残留", not inner)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", nargs="*",
                        default=["600519", "601318", "600030"])
    parser.add_argument("--years", nargs="*", type=int, default=[2024])
    parser.add_argument("--strict-main-tables", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    blocks_root = PROJECT_ROOT / cfg["paths"]["processed_blocks"]
    prior = load_index(blocks_root / "normalize_manifest.jsonl")
    rep = Report()

    for code in args.codes:
        for year in args.years:
            blocks, page_count = load_case(code, year, blocks_root)
            if not blocks:
                rep.check(f"[{code}] 产物存在", False, "blocks 为空，请先运行 normalize")
                continue
            check_common(rep, code, blocks, page_count)
            if code == "600519":
                check_moutai(rep, blocks)
            elif code == "601318":
                check_pingan(rep, blocks)
            elif code == "600030":
                check_citic(rep, blocks)

    if args.strict_main_tables:
        for doc_id, rec in prior.items():
            if rec.get("stock_code") in args.codes and rec.get("year") in args.years:
                rep.check(f"[{rec['stock_code']}] BS/IS/CF 主表零降级",
                          rec.get("degraded_main_bs_is_cf", 0) == 0,
                          f"degraded={rec.get('degraded_main_bs_is_cf')}")

    n_fail = 0
    for name, ok, detail in rep.results:
        mark = "PASS" if ok else "FAIL"
        line = f"[{mark}] {name}"
        if not ok and detail:
            line += f"  -> {detail}"
        print(line)
        if not ok:
            n_fail += 1
    print(f"\n共 {len(rep.results)} 条断言，失败 {n_fail} 条")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
