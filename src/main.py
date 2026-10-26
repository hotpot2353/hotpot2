# 查询期 CLI 入口：接收 query → 驱动 Flow → 终端输出精简结论
# 运行：python src/main.py "对贵州茅台做完整研判"
import sys
from pathlib import Path

# src/ 入 sys.path：main.py 直接以脚本运行时，core/agents/llm 等按 src 根绝对导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.orchestrator.flow import run_flow  # noqa: E402
from core.config import get_settings  # noqa: E402
# 入口先触发 load_dotenv（挂在 core.config 导入时）：.env 的 CREWAI_TRACING_ENABLED=false
# 必须在 kickoff 前进 os.environ，否则 crewai 每次跑完弹 20s trace 分享询问，答 y 会静默上传
_ = get_settings


def main() -> None:
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print('用法: python src/main.py "你的研判问题"')
        return
    try:
        summary = run_flow(query)
    except ValueError as exc:
        # 可预期拒绝路径（意图非法/池外全维度不可支撑）：给用户可读提示而非堆栈
        print(f"无法完成研判：{exc}")
        return
    print(summary)


if __name__ == "__main__":
    main()
