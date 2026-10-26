# 规范化 block 的结构工厂、JSONL 落盘与 normalize_manifest 读写
import json
import os
from pathlib import Path


def make_block(doc_id, seq, btype, page_start, page_end, section_path,
               payload, detected_by=""):
    # 统一公共字段；payload 提供类型相关字段
    block = {
        "block_id": f"{doc_id}#b{seq:05d}",
        "doc_id": doc_id,
        "seq": seq,
        "type": btype,
        "page_start": page_start,
        "page_end": page_end,
        "section_path": list(section_path or []),
        "section_detected_by": detected_by,
    }
    block.update(payload)
    return block


def make_cell(text, value=None, note=False):
    # 单元格：text 为规范化显示，value 为去千分位数值串（金额字符串存储，不用浮点）
    return {"text": text, "value": value, "note": bool(note)}


def write_blocks(blocks, out_path: Path) -> None:
    # 临时文件替换式覆盖写，保证重复执行结果确定、中途失败不破坏旧产物
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for block in blocks:
            f.write(json.dumps(block, ensure_ascii=False) + "\n")
    os.replace(tmp_path, out_path)


def load_blocks(in_path: Path):
    blocks = []
    if not in_path.exists():
        return blocks
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                blocks.append(json.loads(line))
    return blocks


def load_index(path: Path) -> dict:
    # 同一 doc_id 多条记录时取最后一条为当前状态（与 parse_reports 一致）
    index = {}
    if not path.exists():
        return index
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                index[rec["doc_id"]] = rec
    return index


def append_manifest(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
