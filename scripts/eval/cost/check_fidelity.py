# 摘要数值保真检查（成本评测·质量护栏）：压缩是否丢改数字，纯程序零 LLM
# 逻辑：trace json 里取压缩摘要 + 证据 chunk_id 列表 → 回库取子块原文 →
#   摘要中每个数字必须在原文中找到（归一化逗号/百分号/尾零后子串匹配）
# 已知局限：四舍五入形式差异（20.3% vs 20.30%）已归一化处理；同比换算值（原文只给
#   两期绝对值、摘要写了百分比）会误报缺失——逐条人工看 missing 即可分辨
# 用法：python scripts/eval/cost/check_fidelity.py {trace_json路径|--latest}
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from db.connection import connect  # noqa: E402

TRACE_DIR = PROJECT_ROOT / "logs" / "trace"
NUM_RE = re.compile(r"\d[\d,，.]*\%?")


def norm(s: str) -> str:
    # 归一化：去逗号/百分号，去小数尾零（20.30 -> 20.3），末尾孤立点去掉
    s = s.replace(",", "").replace("，", "").rstrip("%")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def load_summary_and_ids():
    path = (max(TRACE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
            if sys.argv[1] == "--latest" else Path(sys.argv[1]))
    trace = json.loads(path.read_text(encoding="utf-8"))
    retr = trace.get("nodes", {}).get("fundamental", {}).get("retrieval") or {}
    summary = retr.get("summary") or ""
    if not summary:
        sys.exit(f"该 trace 无压缩摘要（压缩开关未开或摘要失败）: {path.name}")
    return path, summary, retr.get("chunk_ids") or []


def fetch_chunks(chunk_ids: list) -> str:
    if not chunk_ids:
        sys.exit("evidence_ref 为空，无原文可核对")
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT content FROM chunks WHERE chunk_id = ANY(%s)", (chunk_ids,))
            return "\n".join(r[0] for r in cur.fetchall())


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("用法: python check_fidelity.py {trace_json路径|--latest}")
    path, summary, ids = load_summary_and_ids()
    blob = norm(fetch_chunks(ids))

    numbers = sorted({norm(n) for n in NUM_RE.findall(summary) if norm(n)})
    missing = [n for n in numbers if n not in blob]
    print(f"trace: {path.name}")
    print(f"摘要数字 {len(numbers)} 个，未见于证据原文 {len(missing)} 个")
    for n in numbers:
        print(f"  [{'MISS' if n in missing else 'OK  '}] {n}")
    if missing:
        print("\n存在缺失数字：逐条核对是否为同比换算值（原文只给两期绝对值属正常误报），"
              "其余视为摘要丢改数字，需收紧 rag_compressor 提示词")
    else:
        print("\n保真通过：摘要所有数字均可在证据原文中找到")


if __name__ == "__main__":
    main()
