# 切块产物断言验证：父子完整性 / 哈希配方 / 长度分布 / 表格行守恒 / 幂等确定性
# 用法：verify_chunk.py [code ...]（默认 3 样本 600519/601318/600030 的 2024）
import hashlib
import json
import re
import sys
from pathlib import Path

# markdown 表格分隔行：|---|---|（每格 ≥2 连字符，允许两侧空白）
_SEP_RE = re.compile(r"^\|(?:\s*-{2,}\s*\|)+$")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"
BLOCKS_DIR = PROJECT_ROOT / "data" / "processed" / "blocks"

passed = failed = 0


def check(label, cond, detail=""):
    global passed, failed
    print(("PASS" if cond else "FAIL"), "|", label, detail)
    passed, failed = passed + (1 if cond else 0), failed + (0 if cond else 1)


def verify_doc(code: str, year: int) -> None:
    print(f"\n===== {code} {year} =====")
    chunks_path = CHUNKS_DIR / code / f"{year}_chunks.jsonl"
    check("产物存在", chunks_path.exists())
    if not chunks_path.exists():
        return
    chunks = [json.loads(l) for l in chunks_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    parents = [c for c in chunks if c["chunk_role"] == "parent"]
    children = [c for c in chunks if c["chunk_role"] == "child"]
    print(f"父块 {len(parents)} / 子块 {len(children)}")

    # [1] ID 唯一性
    ids = [c["chunk_id"] for c in chunks]
    check("chunk_id 全局唯一", len(ids) == len(set(ids)))

    # [2] 哈希配方可复算（抽 3 条）
    import random
    random.seed(42)
    ok_hash = True
    for c in random.sample(chunks, min(3, len(chunks))):
        section = " / ".join(c["section_path"])
        expect = hashlib.sha256(
            f"{c['stock_code']}|{c['year']}|{section}|{c['chunk_role']}|{c['content']}".encode("utf-8")
        ).hexdigest()
        if expect != c["chunk_id"]:
            ok_hash = False
            break
    check("chunk_id = sha256(股票|年份|章节|角色|原文)", ok_hash)

    # [3] 父子完整性：子块 parent_id 必指向同文档父块；父块 parent_id 为空
    parent_ids = {p["chunk_id"] for p in parents}
    check("父块 parent_id 全空", all(p["parent_id"] is None for p in parents))
    check("子块 parent_id 全部命中本篇父块",
          all(c["parent_id"] in parent_ids for c in children))

    # [4] 子块内容完整落在父块内（Small-to-Big 回取语义正确性）
    parent_text = {p["chunk_id"]: p["content"] for p in parents}
    ok_contain = all(c["content"] in parent_text[c["parent_id"]] for c in children)
    check("子块 content 完整包含于父块 content", ok_contain)

    # [5] 角色字段与检索原料一致性（embed_input 仅 child 有）
    check("embed_input 仅子块携带",
          all(("embed_input" in c) == (c["chunk_role"] == "child") for c in chunks))

    # [6] 元数据三件套齐全
    check("元数据三件套非空",
          all(c["stock_code"] and c["year"] and c["section_path"] for c in chunks))

    # [7] 无空内容
    check("content 无空串", all(c["content"].strip() for c in chunks))

    # [8] 长度分布（长度仅兜底：允许个别超限，报告比例）
    over_child = [c for c in children if len(c["content"]) > 1200 * 1.5]
    over_parent = [p for p in parents if len(p["content"]) > 6000 * 1.5]
    check("子块超 1.5×上限占比 < 1%", len(over_child) < max(1, len(children) // 100),
          f"超限 {len(over_child)}/{len(children)}")
    check("父块超 1.5×上限占比 < 1%", len(over_parent) < max(1, len(parents) // 100),
          f"超限 {len(over_parent)}/{len(parents)}")

    # [9] 表格行守恒：输入 table 块的 rows 总数 = 输出表格子块内数据行总数
    # 每段恰含一份表头+分隔行：有分隔行的段只数分隔行后的数据行（复制表头不计）；
    # 无表头的表（header_rows 为空）无分隔行，全部管道行即数据行
    blocks = [json.loads(l) for l in
              (BLOCKS_DIR / code / f"{year}_blocks.jsonl").read_text(encoding="utf-8").splitlines()
              if l.strip()]
    n_rows_in = sum(len(b.get("rows") or []) for b in blocks if b["type"] == "table")
    n_rows_out = 0
    for c in children:
        lines = c["content"].splitlines()

        def is_sep(l):
            # 分隔行精确特征：|---|---| 模式（每格 ≥2 连字符）；
            # "-"占位数据行（空值单元格）形如 "|  |  | - |  |" 不误判
            return bool(_SEP_RE.match(l))

        pipe_lines = [i for i, l in enumerate(lines)
                      if l.startswith("|") and not is_sep(l)]
        if not pipe_lines:
            continue  # 非表格子块
        sep_idx = [i for i, l in enumerate(lines) if is_sep(l)]
        if sep_idx:
            first_sep = sep_idx[0]
            n_rows_out += sum(1 for i in pipe_lines if i > first_sep)
        else:
            n_rows_out += len(pipe_lines)
    check("表格数据行守恒（切分仅复制表头不丢行）",
          n_rows_in == n_rows_out, f"输入 {n_rows_in} / 输出 {n_rows_out}")

    # [10] 章节覆盖：输入 path 或出现在产物 section_path 前缀中，
    # 或（被短小节合并的组）以【标签行】形式保留在父块 content 中
    # 纯空内容块（text 空的孤立残留）无内容可保留，不纳入覆盖范围
    in_paths = {tuple(b.get("section_path") or []) for b in blocks
                if b["type"] == "table" or (b.get("text") or "").strip()}
    out_prefixes = {tuple(c["section_path"]) for c in chunks}
    all_parent_text = "\n".join(p["content"] for p in parents)
    ok_cov = True
    for p in in_paths:
        if any(op[:len(p)] == p for op in out_prefixes):
            continue
        label = f"【{' / '.join(p)}】"
        if label not in all_parent_text:
            ok_cov = False
            print("   未覆盖:", p)
            break
    check("输入章节路径全覆盖（字段或标签二选一保留）", ok_cov)


def main() -> None:
    codes = sys.argv[1:] or ["600519", "601318", "600030"]
    for code in codes:
        verify_doc(code, 2024)
    print(f"\n{passed} PASS / {failed} FAIL")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
