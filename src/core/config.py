# 全局配置：yaml 加载工具 + get_settings() 冻结配置单例
# 约定：机密项（API Key、数据库密码）只从 .env 环境变量读取，不落 yaml；
#       参数项以 configs/config.yaml 为唯一来源，不设环境变量覆盖，保证配置可追溯
import os
from dataclasses import dataclass
from functools import lru_cache

import yaml
from dotenv import load_dotenv

from core.logger import get_logger
from core.paths import CONFIGS_DIR, PROJECT_ROOT

logger = get_logger(__name__)

# 项目根 .env 只加载一次；系统已有环境变量优先于 .env 文件
load_dotenv(PROJECT_ROOT / ".env")

# 方舟/巨潮等国内站点一律直连，绕过系统代理：
# 代理软件未运行时 requests 仍会读注册表代理配置导致连接失败
os.environ.setdefault("NO_PROXY", "*")


def load_yaml(name: str) -> dict:
    # 读取 configs/ 下的 yaml 文件
    # name: configs/ 下的文件名，如 "config.yaml"、"rag.yaml"
    # 空文件返回 {}；文件缺失或解析失败记日志后抛出原异常，由调用方决定兜底策略
    path = CONFIGS_DIR / name
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.error("配置文件不存在：%s", path)
        raise
    except yaml.YAMLError as e:
        logger.error("配置文件解析失败：%s，%s", path, e)
        raise


def get_env_var(key: str, default: str = "") -> str:
    # 从环境变量取值（含 .env 注入项），缺失时回退默认值
    return os.getenv(key, default)


@dataclass(frozen=True)
class AppConfig:
    # 应用标识
    name: str
    env: str
    timezone: str


@dataclass(frozen=True)
class ArkConfig:
    # 火山方舟（OpenAI 兼容接口）连接与模型参数
    api_key: str          # 密钥只从环境变量 ARK_API_KEY 读取，不落 yaml
    base_url: str
    llm_model: str
    timeout: int


@dataclass(frozen=True)
class VikingDBConfig:
    # VikingDB 数据面（知识库 rerank 接口）连接参数
    # AK/SK 只从环境变量 VIKINGDB_AK/VIKINGDB_SK 读取，不落 yaml
    host: str
    region: str
    scheme: str
    timeout: int


@dataclass(frozen=True)
class DbConfig:
    # PostgreSQL 连接参数；密码只从环境变量 POSTGRES_PASSWORD 读取
    host: str
    port: int
    dbname: str
    user: str
    password: str
    table_financials: str
    table_quotes: str


@dataclass(frozen=True)
class CninfoConfig:
    # 巨潮直连参数
    query_url: str
    pdf_base: str
    user_agent: str
    page_size: int
    report_season_start: str   # 年报披露窗口起始，如 "01-01"
    report_season_end: str     # 年报披露窗口截止，如 "04-30"


@dataclass(frozen=True)
class DatasourceConfig:
    # 数据源策略：巨潮直连为主源，akshare 为备用源
    primary: str
    fallback: str
    cninfo: CninfoConfig
    tencent_quote_url: str
    years: tuple                 # 采集年份，如 (2023, 2024)
    request_interval_sec: float  # 请求限流间隔（秒）
    max_retries: int
    timeout: int
    cache_enabled: bool


@dataclass(frozen=True)
class NormalizeConfig:
    # 年报规范化调参；规则变更后 bump normalizer_version 触发全量重算
    normalizer_version: str
    max_continuation_page_gap: int
    bind_score_threshold: int
    textflow_min_numeric_rows: int
    armed_title_ttl_pages: int


@dataclass(frozen=True)
class CompressionConfig:
    # 中间结果摘要压缩：RAG 检索命中子块的抽取式摘要开关与预算
    # enabled=false（debug 排障/评测）：检索子块直传全文不做摘要
    enabled: bool
    max_tokens: int     # 单次抽取式摘要输出 token 上限


@dataclass(frozen=True)
class HallucinationConfig:
    # 风控第3层 LLM 幻觉标记：候选仅标记不定案，等级由代码按型定
    # enabled=false（评测）：整体关闭，风控退回零 LLM 双层
    enabled: bool


@dataclass(frozen=True)
class Settings:
    # 全局配置聚合；冻结防篡改，进程内仅此一份
    app: AppConfig
    ark: ArkConfig
    vikingdb: VikingDBConfig
    db: DbConfig
    datasource: DatasourceConfig
    normalize: NormalizeConfig
    compression: CompressionConfig
    hallucination: HallucinationConfig
    # paths 段不在此定义：结构性锚点见 core.paths，数据子路径由脚本层按需读取


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # 加载并缓存全局配置；重复调用返回同一实例
    raw = load_yaml("config.yaml")

    app_raw = raw.get("app", {})
    ark_raw = raw.get("ark", {})
    vik_raw = raw.get("vikingdb", {})
    db_raw = raw.get("db", {})
    ds_raw = raw.get("datasource", {})
    cninfo_raw = ds_raw.get("cninfo", {})
    season_raw = cninfo_raw.get("report_season", {})
    norm_raw = raw.get("normalize", {})
    comp_raw = raw.get("compression", {})
    halu_raw = raw.get("hallucination", {})

    return Settings(
        app=AppConfig(
            name=app_raw.get("name", "fin_analyst_crew"),
            env=app_raw.get("env", "dev"),
            timezone=app_raw.get("timezone", "Asia/Shanghai"),
        ),
        ark=ArkConfig(
            api_key=get_env_var("ARK_API_KEY"),
            base_url=ark_raw.get("base_url", ""),
            llm_model=ark_raw.get("llm_model", ""),
            timeout=int(ark_raw.get("timeout", 120)),
        ),
        vikingdb=VikingDBConfig(
            host=vik_raw.get("host", "api-knowledgebase.mlp.cn-beijing.volces.com"),
            region=vik_raw.get("region", "cn-beijing"),
            scheme=vik_raw.get("scheme", "https"),
            timeout=int(vik_raw.get("timeout", 30)),
        ),
        db=DbConfig(
            host=db_raw.get("host", "localhost"),
            port=int(db_raw.get("port", 5432)),
            dbname=db_raw.get("dbname", "fin_rag"),
            user=db_raw.get("user", "fin"),
            # 本地开发默认密码兜底，与历史 connection.py 行为保持一致
            password=get_env_var("POSTGRES_PASSWORD", "fin_local_dev"),
            table_financials=db_raw.get("table_financials", "fin_financials"),
            table_quotes=db_raw.get("table_quotes", "fin_quotes"),
        ),
        datasource=DatasourceConfig(
            primary=ds_raw.get("primary", "cninfo_direct"),
            fallback=ds_raw.get("fallback", "akshare"),
            cninfo=CninfoConfig(
                query_url=cninfo_raw.get("query_url", ""),
                pdf_base=cninfo_raw.get("pdf_base", ""),
                user_agent=cninfo_raw.get("user_agent", "Mozilla/5.0"),
                page_size=int(cninfo_raw.get("page_size", 30)),
                report_season_start=str(season_raw.get("start", "01-01")),
                report_season_end=str(season_raw.get("end", "04-30")),
            ),
            tencent_quote_url=ds_raw.get("tencent_quote_url", ""),
            years=tuple(ds_raw.get("years", ())),
            request_interval_sec=float(ds_raw.get("request_interval_sec", 1.0)),
            max_retries=int(ds_raw.get("max_retries", 3)),
            timeout=int(ds_raw.get("timeout", 30)),
            cache_enabled=bool(ds_raw.get("cache_enabled", True)),
        ),
        normalize=NormalizeConfig(
            normalizer_version=norm_raw.get("normalizer_version", ""),
            max_continuation_page_gap=int(norm_raw.get("max_continuation_page_gap", 2)),
            bind_score_threshold=int(norm_raw.get("bind_score_threshold", 4)),
            textflow_min_numeric_rows=int(norm_raw.get("textflow_min_numeric_rows", 8)),
            armed_title_ttl_pages=int(norm_raw.get("armed_title_ttl_pages", 2)),
        ),
        compression=CompressionConfig(
            enabled=bool(comp_raw.get("enabled", True)),
            max_tokens=int(comp_raw.get("max_tokens", 600)),
        ),
        hallucination=HallucinationConfig(
            enabled=bool(halu_raw.get("enabled", True)),
        ),
    )


if __name__ == "__main__":
    # 直接运行时打印配置摘要自检；密钥只打印是否已配置，不回显内容
    s = get_settings()
    print("app:", s.app)
    print("ark:", s.ark.base_url, s.ark.llm_model, "api_key_set:", bool(s.ark.api_key))
    print("db:", s.db.host, s.db.port, s.db.dbname, s.db.user)
    print("datasource:", s.datasource.primary, s.datasource.years)
    print("normalize:", s.normalize.normalizer_version)
    print("compression:", s.compression.enabled, s.compression.max_tokens)
    print("hallucination:", s.hallucination.enabled)
