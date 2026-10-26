# 成本 A/B 评测：双开关（compression.enabled / hallucination.enabled）四组对照
#   A_baseline 双关 / B_compress 仅压缩 / C_halluc 仅标记 / D_full 全开
# 机制：get_settings 是 lru_cache 进程内只读一次，不能进程内改配置——本脚本逐组
#   改 config.yaml 开关 → 起子进程跑 main.py → finalize 落盘 logs/trace/{trace_id}.json
#   → 按 mtime 取新文件解析聚合为 results/ab_summary.csv；开关在 finally 还原
# 噪声提示：采集侧 akshare 偶发断连会引入组间噪声，看趋势用 --runs 3 取均值更稳
# 用法：python scripts/eval/cost/run_ab.py [--runs N] [--query "..."]
import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
TRACE_DIR = PROJECT_ROOT / "logs" / "trace"
OUT_CSV = Path(__file__).resolve().parent / "results" / "ab_summary.csv"

# 默认评测 query：与历次验收同款，保证与历史基线可比
QUERY = "研判贵州茅台600519的2024年基本面、技术面和舆情"

# 组名 -> (compression.enabled, hallucination.enabled)
GROUPS = {
    "A_baseline": (False, False),
    "B_compress": (True, False),
    "C_halluc": (False, True),
    "D_full": (True, True),
}


def set_switches(comp: bool, halu: bool) -> None:
    # 逐行扫描改开关：只动目标段下的 enabled 行，注释原样保留（yaml 重写会丢注释）
    target = {"compression": comp, "hallucination": halu}
    lines = CONFIG.read_text(encoding="utf-8").splitlines(keepends=True)
    section = None
    for i, line in enumerate(lines):
        m = re.match(r"^(\w+):", line)
        if m:
            section = m.group(1)
            continue
        if section in target and re.match(r"^  enabled: ", line):
            lines[i] = f"  enabled: {str(target[section]).lower()}\n"
    CONFIG.write_text("".join(lines), encoding="utf-8")


def newest_trace() -> Path:
    # 按 mtime 取最新 trace json（子进程每次跑完落一个新文件）
    files = sorted(TRACE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError("logs/trace/ 下无 trace json")
    return files[-1]


def run_once(query: str) -> dict:
    # 起子进程跑全链路（sys.executable 即 FIN 环境 python），返回最新 trace json 内容
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "main.py"), query],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"子进程失败 rc={proc.returncode}: {proc.stdout[-500:]}")
    return json.loads(newest_trace().read_text(encoding="utf-8"))


def flatten(trace: dict) -> dict:
    # trace json → 一行账单（与 finalize 汇总同口径，另拆关键节点耗时与检索体量）
    nodes = trace.get("nodes", {})
    tok = lambda n: (nodes.get(n, {}).get("tokens") or {})  # noqa: E731
    retr = nodes.get("fundamental", {}).get("retrieval") or {}
    row = {
        "trace_id": trace.get("trace_id", ""),
        "total_ms": sum(n.get("ms", 0) for n in nodes.values()),
        "prompt": sum(t.get("prompt", 0) for t in (tok(n) for n in nodes)),
        "completion": sum(t.get("completion", 0) for t in (tok(n) for n in nodes)),
        "evidence_refs": len(retr.get("chunk_ids") or []),
        "errors": len(trace.get("errors") or []),
        "fund_ms": nodes.get("fundamental", {}).get("ms", 0),
        "risk_ms": nodes.get("risk", {}).get("ms", 0),
        "fund_prompt": tok("fundamental").get("prompt", 0),
        "risk_prompt": tok("risk").get("prompt", 0),
        "summary_chars": retr.get("summary_chars", 0),
        "chunks_chars": retr.get("chunks_chars", 0),
    }
    ratio = (row["summary_chars"] / row["chunks_chars"]) if row["chunks_chars"] else 0
    row["compress_ratio"] = round(ratio, 4)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="成本 A/B 评测（四组对照）")
    parser.add_argument("--runs", type=int, default=1, help="每组重复次数（默认1，看趋势建议3）")
    parser.add_argument("--query", default=QUERY)
    args = parser.parse_args()

    rows = []
    original = CONFIG.read_text(encoding="utf-8")
    try:
        for group, (comp, halu) in GROUPS.items():
            set_switches(comp, halu)
            for r in range(args.runs):
                t0 = time.time()
                try:
                    trace = run_once(args.query)
                    row = {"group": group, **flatten(trace)}
                    rows.append(row)
                    print(f"[{group} #{r + 1}] {row['trace_id']} {row['total_ms']}ms "
                          f"prompt={row['prompt']} completion={row['completion']} "
                          f"refs={row['evidence_refs']} errors={row['errors']} "
                          f"({time.time() - t0:.0f}s)")
                except Exception as exc:  # 单组失败不拖垮整个评测
                    print(f"[{group} #{r + 1}] 失败: {exc}")
    finally:
        CONFIG.write_text(original, encoding="utf-8")  # 开关还原，评测不留痕

    if not rows:
        print("无有效结果")
        return

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 终端摘要：按组均值对比
    print(f"\n{'组':<12}{'次数':<6}{'平均ms':<10}{'平均prompt':<12}{'平均completion':<16}"
          f"{'压缩比':<10}{'平均errors':<12}")
    for group in GROUPS:
        g = [r for r in rows if r["group"] == group]
        if not g:
            continue
        n = len(g)
        print(f"{group:<12}{n:<6}{sum(r['total_ms'] for r in g) // n:<10}"
              f"{sum(r['prompt'] for r in g) // n:<12}"
              f"{sum(r['completion'] for r in g) // n:<16}"
              f"{sum(r['compress_ratio'] for r in g) / n:<10.4f}"
              f"{sum(r['errors'] for r in g) / n:<12.2f}")
    print(f"\n明细 CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
