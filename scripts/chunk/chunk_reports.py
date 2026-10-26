# 语义切块：blocks JSONL → 父子 chunks JSONL（离线预建管道核心工序）
# 输入：data/processed/blocks/<code>/<year>_blocks.jsonl（normalize 产物，block 三类 heading/prose/table）
# 输出：data/processed/chunks/<code>/<year>_chunks.jsonl + chunks_manifest.jsonl
# 幂等键：输入 blocks sha256 + normalizer_version + chunker_version，与 normalize 同一套惯例
# 算法（标题感知，长度仅兜底）：
#   1. 分组：按 section_path 元组取连续段为"小节组"——heading 的 path 末项=自身标题，
#      其引导的 prose 天然同组，heading 成为组内首行
#   2. 短小节合并：组文本 < short_section_chars 且与相邻组同父级（path[:-1] 相同）→ 并入前组
#   3. 子块生成：组内 prose 相邻聚合（句号处软断段，child_max_chars 兜底）；
#      表格整表成块（>child_max_chars 按行段切、每段复制表头）
#   4. 父块生成：连续子块拼接（parent_max_chars 兜底切分），子块永远完整落在其父块内
# 哈希配方：chunk_id = sha256("股票|年份|章节|角色|原文")，重跑产出同 ID（schemas.py 契约）
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "normalize"))

from blocks_io import append_manifest, load_blocks, write_blocks  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.logger import get_logger  # noqa: E402
from rag.config import get_rag_settings  # noqa: E402

logger = get_logger(__name__)

BLOCKS_DIR = PROJECT_ROOT / "data" / "processed" / "blocks"
CHUNKS_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"
MANIFEST_PATH = CHUNKS_DIR / "chunks_manifest.jsonl"


def chunk_id_of(code: str, year: int, section: str, role: str, text: str) -> str:
    # 内容哈希主键：重跑切块产出同一 ID，入库幂等由此保证
    raw = f"{code}|{year}|{section}|{role}|{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_to_markdown(block: dict) -> str:
    # 从 header_rows + rows 重建 markdown 表格文本（blocks 的 table.text 为空，内容在结构化字段）
    lines = []
    name = block.get("name") or ""
    if name:
        lines.append(f"【{name}】")
    unit = block.get("unit") or ""
    if unit:
        lines.append(unit)
    headers = block.get("header_rows") or []
    for h in headers:
        lines.append("| " + " | ".join(str(c) for c in h) + " |")
    if headers:
        ncols = max(len(h) for h in headers)
        lines.append("|" + "---|" * ncols)
    for row in block.get("rows") or []:
        cells = [c.get("text") or "" for c in row.get("cells") or []]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def split_table_markdown(block: dict, full_text: str, max_chars: int) -> list:
    # 超长表按行段切分，每段复制表头（保证每段的数字与列头对应，嵌入语义完整）
    name = block.get("name") or ""
    unit = block.get("unit") or ""
    head_lines = []
    if name:
        head_lines.append(f"【{name}】")
    if unit:
        head_lines.append(unit)
    headers = block.get("header_rows") or []
    header_block = [f"| {' | '.join(str(c) for c in h)} |" for h in headers]
    if headers:
        ncols = max(len(h) for h in headers)
        header_block.append("|" + "---|" * ncols)
    head_len = len("\n".join(head_lines + header_block))

    body = []
    for row in block.get("rows") or []:
        cells = [c.get("text") or "" for c in row.get("cells") or []]
        body.append("| " + " | ".join(cells) + " |")

    parts = []
    cur = []
    cur_len = head_len
    for line in body:
        add = len(line) + 1
        if cur and cur_len + add > max_chars:
            parts.append("\n".join(head_lines + header_block + cur))
            cur = []
            cur_len = head_len
        cur.append(line)
        cur_len += add
    if cur:
        parts.append("\n".join(head_lines + header_block + cur))
    return parts or [full_text]


def group_blocks(blocks: list, short_chars: int) -> list:
    # 按 section_path 元组取连续段为组；短小节（<short_chars）向同父级相邻组合并
    groups = []
    for b in blocks:
        path = tuple(b.get("section_path") or ["（正文）"])
        if groups and groups[-1]["path"] == path:
            groups[-1]["blocks"].append(b)
        else:
            groups.append({"path": path, "blocks": [b]})

    def group_text_len(g):
        n = 0
        for b in g["blocks"]:
            if b["type"] == "table":
                n += len(table_to_markdown(b))
            else:
                n += len(b.get("text") or "") + 1
        return n

    def tag_block(path, first_block):
        # 伪 prose 块：携带被合并组的章节标签，页码取该组首块（内容与标签都保留）
        return {
            "seq": first_block.get("seq"),
            "type": "prose",
            "page_start": first_block.get("page_start"),
            "page_end": first_block.get("page_end"),
            "section_path": list(path),
            "text": f"【{' / '.join(path)}】",
        }

    # 短组向前合并：同父级（path[:-1] 相同）
    merged = []
    for g in groups:
        if (merged and group_text_len(g) < short_chars
                and g["path"][:-1] == merged[-1]["path"][:-1]):
            merged[-1]["blocks"].append(tag_block(g["path"], g["blocks"][0]))
            merged[-1]["blocks"].extend(g["blocks"])
        else:
            merged.append(g)
    # 开头的短组（无前组）向后合并：与后组同父级则并入后组
    if len(merged) > 1 and group_text_len(merged[0]) < short_chars \
            and merged[0]["path"][:-1] == merged[1]["path"][:-1]:
        merged[1]["blocks"] = [tag_block(merged[0]["path"], merged[0]["blocks"][0])] \
            + merged[0]["blocks"] + merged[1]["blocks"]
        merged.pop(0)
    return merged


def build_children(code: str, year: int, group: dict, max_chars: int) -> list:
    # 组内生成子块元素：prose 相邻聚合（句号软断段）+ 表格整表成块
    section = " / ".join(group["path"])
    items = []
    para = []          # 当前聚合中的文本行
    para_pages = [None, None]

    def flush():
        if not para:
            return
        text = "\n".join(para)
        items.append({
            "text": text,
            "page_start": para_pages[0],
            "page_end": para_pages[1],
            "section": section,
        })
        para.clear()
        para_pages[0] = para_pages[1] = None

    for b in group["blocks"]:
        pages = (b.get("page_start"), b.get("page_end"))
        if b["type"] == "table":
            flush()
            md = table_to_markdown(b)
            parts = split_table_markdown(b, md, max_chars) if len(md) > max_chars else [md]
            for p in parts:
                items.append({
                    "text": p,
                    "page_start": pages[0],
                    "page_end": pages[1],
                    "section": section,
                })
            continue
        line = (b.get("text") or "").strip()
        if not line:
            continue
        if para_pages[0] is None:
            para_pages[0] = pages[0]
        para_pages[1] = pages[1]
        # 句号软断段：累计接近上限（0.75×max）且当前行以句末标点结尾 → 在句边界断段，
        # 保持段落语义完整；远离上限时持续聚合，避免子块过碎、语义密度不足
        if para and sum(len(x) for x in para) + len(line) > max_chars:
            flush()
            para_pages[0] = pages[0]
            para_pages[1] = pages[1]
        elif para and line.endswith(("。", "！", "？", "；")) \
                and sum(len(x) for x in para) >= max_chars * 3 // 4:
            para.append(line)
            flush()
            continue
        para.append(line)
    flush()
    return items


def chunk_document(blocks: list, code: str, year: int, cfg) -> list:
    # 单份年报 → 父子块记录列表（未含嵌入，嵌入在 Step 5）
    chunks = []
    # 文档级子块去重：跨页模板行（如审计报告续页头）会在不同组生成完全相同的子块，
    # 重复内容零信息增量，保留首个即可（去重前移到嵌入前，与 Step 4 原则一致）
    seen_child_texts = set()
    for group in group_blocks(blocks, cfg.short_section_chars):
        children = []
        for c in build_children(code, year, group, cfg.child_max_chars):
            key = (c["section"], c["text"])
            if key in seen_child_texts:
                continue
            seen_child_texts.add(key)
            children.append(c)
        if not children:
            continue
        # 父块 = 连续子块拼接（parent_max_chars 兜底切分）；子块永远完整落在其父块内
        parent_batches = []
        cur = []
        cur_len = 0
        for c in children:
            add = len(c["text"]) + 1
            if cur and cur_len + add > cfg.parent_max_chars:
                parent_batches.append(cur)
                cur = []
                cur_len = 0
            cur.append(c)
            cur_len += add
        if cur:
            parent_batches.append(cur)

        for batch in parent_batches:
            parent_text = "\n".join(c["text"] for c in batch)
            section = batch[0]["section"]
            parent_id = chunk_id_of(code, year, section, "parent", parent_text)
            pages = (
                min(c["page_start"] for c in batch if c["page_start"] is not None),
                max(c["page_end"] for c in batch if c["page_end"] is not None),
            )
            chunks.append({
                "chunk_id": parent_id,
                "doc_id": f"{code}_{year}_annual",
                "stock_code": code,
                "year": year,
                "section_path": list(group["path"]),
                "chunk_role": "parent",
                "parent_id": None,
                "content": parent_text,
                "page_start": pages[0],
                "page_end": pages[1],
            })
            for c in batch:
                child_id = chunk_id_of(code, year, c["section"], "child", c["text"])
                chunks.append({
                    "chunk_id": child_id,
                    "doc_id": f"{code}_{year}_annual",
                    "stock_code": code,
                    "year": year,
                    "section_path": list(group["path"]),
                    "chunk_role": "child",
                    "parent_id": parent_id,
                    "content": c["text"],
                    # 嵌入输入 = 上下文前缀 + 原文；前缀不落库（Step 6 入库时丢弃此字段）
                    "embed_input": f"{code} {year}年报 {' / '.join(group['path'])}\n{c['text']}",
                    "page_start": c["page_start"],
                    "page_end": c["page_end"],
                })
    return chunks


def main() -> None:
    rag = get_rag_settings()
    cfg = rag.chunking
    normalizer_version = get_settings().normalize.normalizer_version

    # 幂等索引：doc_id → 已登记记录（取最后一条为当前状态）
    done = {}
    if MANIFEST_PATH.exists():
        for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["doc_id"]] = rec

    total_parent = total_child = 0
    processed = skipped = failed = 0
    for blocks_file in sorted(BLOCKS_DIR.glob("*/*_blocks.jsonl")):
        code = blocks_file.parent.name
        year = int(blocks_file.stem.split("_")[0])
        doc_id = f"{code}_{year}_annual"

        input_sha = sha256_of_file(blocks_file)
        rec = done.get(doc_id)
        if rec and rec.get("input_sha256") == input_sha \
                and rec.get("normalizer_version") == normalizer_version \
                and rec.get("chunker_version") == cfg.chunker_version:
            skipped += 1
            continue

        try:
            blocks = load_blocks(blocks_file)
            chunks = chunk_document(blocks, code, year, cfg)
        except Exception as exc:
            failed += 1
            logger.error("切块失败 %s: %s", doc_id, exc)
            continue

        out_path = CHUNKS_DIR / code / f"{year}_chunks.jsonl"
        write_blocks(chunks, out_path)

        n_parent = sum(1 for c in chunks if c["chunk_role"] == "parent")
        n_child = len(chunks) - n_parent
        total_parent += n_parent
        total_child += n_child
        processed += 1
        append_manifest(MANIFEST_PATH, {
            "doc_id": doc_id,
            "input_sha256": input_sha,
            "normalizer_version": normalizer_version,
            "chunker_version": cfg.chunker_version,
            "n_parent": n_parent,
            "n_child": n_child,
            "out_path": str(out_path.relative_to(PROJECT_ROOT)),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        logger.info("切块完成 %s: parent=%s child=%s", doc_id, n_parent, n_child)

    logger.info(
        "全量切块结束: 处理=%s 跳过=%s 失败=%s | 父块合计=%s 子块合计=%s (chunker=%s)",
        processed, skipped, failed, total_parent, total_child, cfg.chunker_version)


if __name__ == "__main__":
    main()
