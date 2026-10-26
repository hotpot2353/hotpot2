# 执行 src/db/schema.sql 建表（幂等：IF NOT EXISTS，可重复执行）
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from db.connection import connect, schema_sql, rag_schema_sql  # noqa: E402


def main() -> None:
    ddl = schema_sql()
    # RAG 检索层建表：chunks 切块父子表（含 pgvector 扩展激活）
    rag_ddl = rag_schema_sql()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
            # rag_indexes.sql（HNSW/GIN 重索引）刻意不在此时建：
            # 必须等离线管道批量灌完数据后由入库步骤执行，空表先建索引会让灌数慢数倍
            cur.execute(rag_ddl)
            cur.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname='public' "
                "ORDER BY tablename"
            )
            tables = [r[0] for r in cur.fetchall()]
    print("当前表:", ", ".join(tables))


if __name__ == "__main__":
    main()
