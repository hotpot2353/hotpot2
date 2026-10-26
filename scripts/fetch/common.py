# 采集脚本公共模块：配置加载、日志、限流请求、manifest 登记
import json
import logging
import logging.config
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env() -> None:
    # 读取项目根目录 .env（ARK_API_KEY、PG 凭据）
    load_dotenv(PROJECT_ROOT / ".env")
    # 所有数据源（巨潮/腾讯/东方财富/方舟）均为国内站点，绕过系统代理直连，
    # 避免 Clash 等代理软件未运行时 requests 读注册表代理导致连接失败
    os.environ.setdefault("NO_PROXY", "*")


def load_config() -> dict:
    with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging() -> logging.Logger:
    # 用 configs/logging.yaml 初始化，日志同时进控制台与 logs/app.log
    log_cfg = PROJECT_ROOT / "configs" / "logging.yaml"
    if log_cfg.exists():
        with open(log_cfg, "r", encoding="utf-8") as f:
            logging.config.dictConfig(yaml.safe_load(f))
    return logging.getLogger("fin_analyst_crew")


def read_stock_list(rel_path: str) -> list:
    # 读取 stock_list.txt，忽略空行与 # 注释，只取 6 位代码
    codes = []
    with open(PROJECT_ROOT / rel_path, "r", encoding="utf-8") as f:
        for line in f:
            code = line.split("#")[0].strip()
            if code:
                codes.append(code[:6])
    return codes


def market_of(code: str) -> str:
    # 巨潮查询的 column 参数：6/9 开头为沪市，其余为深市
    return "sse" if code[0] in ("6", "9") else "szse"


def tencent_symbol(code: str) -> str:
    return ("sh" if code[0] in ("6", "9") else "sz") + code


class RateLimitedSession:
    # 带限流与指数退避重试的会话，降低被数据源风控的概率
    def __init__(self, interval: float = 1.0, retries: int = 3,
                 timeout: int = 30, headers: dict = None):
        self.interval = interval  # 限流间隔，两次请求之间至少间隔 1 秒
        self.retries = retries  # 最大重试次数
        self.timeout = timeout  # 请求超时时间
        self.session = requests.Session()
        # User-Agent 伪装普通浏览器
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        # 如果用户传入了自定义请求头，合并到 session.headers 中
        if headers:
            self.session.headers.update(headers)
        self._last = 0.0  # 上次请求时间，用于限流计算
    def _throttle(self) -> None:
        # 计算当前请求与上次请求的时间间隔，若小于限流间隔，等待剩余时间
        wait = self.interval - (time.time() - self._last)
        if wait > 0:
            # 并随机添加 0-300ms 的抖动，避免被数据源识别为批量请求
            time.sleep(wait + random.uniform(0, 0.3))
        self._last = time.time()

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", self.timeout)  # 设置请求超时时间
        # 重试机制：最多重试 `retries` 次，每次间隔 2^attempt 秒
        # 若所有重试都失败，抛出异常
        for attempt in range(1, self.retries + 1):
            self._throttle()
            try:
                resp = self.session.request(method, url, **kwargs)
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                if attempt == self.retries:
                    raise
                logging.getLogger("fin_analyst_crew").warning(
                    "请求失败(%s/%s) %s: %s", attempt, self.retries, url, exc
                )
                time.sleep(2**attempt)
        raise RuntimeError("unreachable")

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)


class Manifest:
    # 解析登记表：逐行追加 JSONL，支持按 doc_id 断点去重
    def __init__(self, rel_path: str):
        self.path = PROJECT_ROOT / rel_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.existing = self._load()

    def _load(self) -> set:
        keys = set()
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        keys.add(json.loads(line)["doc_id"])
        return keys

    def has(self, doc_id: str) -> bool:
        return doc_id in self.existing

    def append(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.existing.add(record["doc_id"])


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
