# 结构性路径锚点：项目根 / configs / prompts / logs / data / reports
# 仅提供目录级锚点；数据子路径（raw/parsed/processed 各层）由 config.yaml paths 段定义、脚本层按需读取
from pathlib import Path

# src/core/paths.py 向上三级即项目根：core -> src -> fin_analyst_crew
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = PROJECT_ROOT / "configs"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_ROOT = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
