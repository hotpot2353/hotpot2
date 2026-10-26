# 日志：get_logger(name) 取 fin_analyst_crew.<name> 子 logger，首次调用自动按 configs/logging.yaml 初始化
# 与 scripts/fetch/common.py 的既有约定保持同一 logger 族与同一份配置，src 运行时不再各写各的
import logging
import logging.config
from pathlib import Path

import yaml

from core.paths import CONFIGS_DIR, LOGS_DIR

# 项目统一 logger 族名称；logging.yaml 按此名挂载控制台 + 轮转文件双 handler
ROOT_LOGGER_NAME = "fin_analyst_crew"

# 进程内只初始化一次
_configured = False


def setup_logging() -> None:
    # 幂等初始化：读 configs/logging.yaml 做 dictConfig
    # 关键修正：yaml 里 file handler 的 filename 是相对路径（logs/app.log），
    # 从非项目根目录启动时会落到错误位置，此处统一锚定到项目根 logs/ 目录
    global _configured
    if _configured:
        return
    cfg_path = CONFIGS_DIR / "logging.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for handler in cfg.get("handlers", {}).values():
            filename = handler.get("filename")
            if filename:
                handler["filename"] = str(LOGS_DIR / Path(filename).name)
        logging.config.dictConfig(cfg)
    else:
        # 兜底：配置缺失时保证日志仍可见
        logging.basicConfig(level=logging.INFO)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    # 取子 logger；传入裸名（如 "db"）自动挂到 fin_analyst_crew 命名空间下，
    # 传全名（如 __name__ 产出的 core.config）时去重前缀，保证日志 %(name)s 清爽
    setup_logging()
    if name.startswith(ROOT_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
