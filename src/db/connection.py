# PostgreSQL 连接：配置收敛至 core.config（config.yaml db 段 + .env 密码）
# 对外接口：connect / schema_sql / rag_schema_sql / load_settings（新增接口不破坏既有脚本）
import psycopg

from core.config import get_settings
from core.paths import PROJECT_ROOT


def load_settings() -> dict:
    # 兼容旧接口：返回 psycopg.connect 所需的连接参数字典
    db = get_settings().db
    return {
        "host": db.host,
        "port": db.port,
        "dbname": db.dbname,
        "user": db.user,
        "password": db.password,
    }


def connect():
    return psycopg.connect(**load_settings())


def schema_sql() -> str:
    return (PROJECT_ROOT / "src" / "db" / "schema.sql").read_text(encoding="utf-8")


def rag_schema_sql() -> str:
    # RAG 层建表 DDL（chunks 父子表 + pgvector 扩展激活）
    # 重索引 rag_indexes.sql 不在此列：由离线入库管道批量灌数后延迟执行
    return (PROJECT_ROOT / "src" / "db" / "rag_schema.sql").read_text(encoding="utf-8")
