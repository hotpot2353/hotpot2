# PDF 解析脚本：年报 PDF -> Markdown
# - 按页保留边界标记（## 第 N 页），作为后续 evidence_ref 的定位锚点
# - 文本块按阅读顺序输出；表格用 PyMuPDF 内置识别转 Markdown，避免与正文重复
# - 解析结果登记 parsed_manifest.jsonl（doc_id -> 源文件哈希 -> 状态），内容未变则跳过
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fetch"))
from common import (  # noqa: E402
    PROJECT_ROOT, load_config, load_env, setup_logging,
)

PARSER_VERSION = "pymupdf-text-tables/1"

# 年报常见章节标题：第X节、一、二、（一）
HEADING_RULES = (
    (re.compile(r"^第[一二三四五六七八九十百0-9]+节[\s　].{0,40}$"), "## "),
    (re.compile(r"^[一二三四五六七八九十]+、.{1,40}$"), "### "),
    (re.compile(r"^（[一二三四五六七八九十]+）.{1,40}$"), "#### "),
)

#计算 PDF 文件的 SHA256 哈希值
#作为幂等键的一部分——如果 PDF 内容没变（哈希相同），就跳过解析
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

#读取下载阶段的`manifest.jsonl` ，返回`doc_id → record` 的字典
#逐行解析 JSON，用`doc_id` 作键,解析阶段要遍历所有已下载的文档，所以需要这个字典
def load_download_manifest(path: Path) -> dict:
    # 读取下载阶段 manifest，返回 doc_id -> record
    docs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                docs[rec["doc_id"]] = rec
    return docs


def load_parsed_index(path: Path) -> dict:
    # parsed_manifest 同一 doc_id 可能有多条记录，取最后一条为当前状态
    index = {}
    if not path.exists():
        return index
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                index[rec["doc_id"]] = rec
    return index

#PyMuPDF 提取文本块时，表格里的文字也会被当作文本块提出来。
# 如果不过滤，表格内容会出现两次——一次作为正文文本，一次作为表格。
def block_in_table(rect, table_rects, threshold: float = 0.5) -> bool:
    # 文本块与表格框重叠面积过半则视为表格内容，不再重复输出
    for tr in table_rects:
        inter = rect & tr
        if not inter.is_empty and (inter.get_area() / max(rect.get_area(), 1)) > threshold:
            return True
    return False

#给识别出的标题行加 Markdown 标题前缀（`##` /`###` /`####` ），不是标题就原样返回
def render_heading(line: str) -> str:
    line = line.strip()
    for pattern, prefix in HEADING_RULES:
        if pattern.match(line):
            return prefix + line
    return line


def render_table(df) -> str:
    # PyMuPDF 抽取的 DataFrame 转 Markdown 表格
    df = df.fillna("")
    headers = [str(c).replace("\n", " ").strip() for c in df.columns]
    rows = [[str(v).replace("\n", " ").strip() for v in row]
            for row in df.itertuples(index=False, name=None)]
    if not any(headers):
        return ""
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def parse_pdf(pdf_path: Path) -> tuple:
    # 返回 (markdown 正文, 页数, 识别到的表格数)
    doc = fitz.open(pdf_path)
    parts = []
    table_count = 0
    for pno in range(len(doc)):
        page = doc[pno]
        tables = page.find_tables()
        table_rects = [fitz.Rect(t.bbox) for t in tables]

        # 非表格文本块，按纵向再横向排序
        blocks = []
        for b in page.get_text("blocks"):
            x0, y0, x1, y1, text, _, block_type = b
            if block_type == 1 or not text.strip():
                continue
            rect = fitz.Rect(x0, y0, x1, y1)
            if block_in_table(rect, table_rects):
                continue
            blocks.append((y0, x0, text))
        blocks.sort(key=lambda x: (round(x[0] / 3), x[1]))

        parts.append(f"## 第 {pno + 1} 页\n")
        for _, _, text in blocks:
            for line in text.splitlines():
                line = line.strip()
                if line:
                    parts.append(render_heading(line))
            parts.append("")

        # 表格统一排在该页正文之后
        for idx, table in enumerate(tables, start=1):
            md = render_table(table.to_pandas())
            if md:
                parts.append(f"**表 {pno + 1}-{idx}**\n")
                parts.append(md)
                parts.append("")
                table_count += 1

    page_count = len(doc)
    doc.close()
    return "\n".join(parts).strip() + "\n", page_count, table_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", nargs="*", help="只处理指定股票代码")
    parser.add_argument("--years", nargs="*", type=int, help="只处理指定年份")
    args = parser.parse_args()

    load_env()
    cfg = load_config()
    log = setup_logging()

    dl_manifest_path = PROJECT_ROOT / cfg["paths"]["parsed_metadata"] / "manifest.jsonl"
    parsed_manifest_path = (
        PROJECT_ROOT / cfg["paths"]["parsed_metadata"] / "parsed_manifest.jsonl"
    )
    md_root = PROJECT_ROOT / cfg["paths"]["parsed_markdown"]

    docs = load_download_manifest(dl_manifest_path)
    parsed_index = load_parsed_index(parsed_manifest_path)

    for doc_id, rec in docs.items():
        if args.codes and rec["stock_code"] not in args.codes:
            continue
        if args.years and rec["year"] not in args.years:
            continue

        pdf_path = PROJECT_ROOT / rec["pdf_path"]
        if not pdf_path.exists():
            log.warning("PDF 缺失，跳过 %s: %s", doc_id, pdf_path)
            continue

        digest = sha256_file(pdf_path)
        prior = parsed_index.get(doc_id)
        out_path = md_root / rec["stock_code"] / f"{rec['year']}_annual.md"
        if prior and prior.get("sha256") == digest and out_path.exists():
            log.info("内容未变化，跳过 %s", doc_id)
            continue

        try:
            body, page_count, table_count = parse_pdf(pdf_path)
        except Exception as exc:
            log.error("解析失败 %s: %s", doc_id, exc)
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "---\n"
            f"doc_id: {doc_id}\n"
            f"stock_code: \"{rec['stock_code']}\"\n"
            f"year: {rec['year']}\n"
            f"title: {json.dumps(rec.get('title', ''), ensure_ascii=False)}\n"
            f"source_url: {rec['source_url']}\n"
            f"sha256: {digest}\n"
            f"parser: {PARSER_VERSION}\n"
            f"page_count: {page_count}\n"
            "---\n\n"
        )
        out_path.write_text(header + body, encoding="utf-8")

        record = {
            "doc_id": doc_id,
            "pdf_path": rec["pdf_path"],
            "sha256": digest,
            "markdown_path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "page_count": page_count,
            "table_count": table_count,
            "char_count": len(body),
            "parser": PARSER_VERSION,
            "status": "parsed",
        }
        with open(parsed_manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.info("已解析 %s | %s 页 | %s 表 | %s 字",
                 doc_id, page_count, table_count, len(body))


if __name__ == "__main__":
    main()
