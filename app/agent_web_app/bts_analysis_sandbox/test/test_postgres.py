"""
PostgreSQL 连接测试脚本
用法: conda run -n pat-nlp-lifecycle python test\test_postgres.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from database_utils import PostgreSQLHelper


def test_connection():
    pg = PostgreSQLHelper.from_connection_string(
        "PORT=5432;DATABASE=devIPS;HOST=192.168.110.82;PASSWORD=123456;USER ID=postgres"
    )

    pg.connect()
    pg.get_current_database()
    tables, views = pg.list_tables_and_views()

    print(f"\n共 {len(tables)} 个表, {len(views)} 个视图")

    if tables:
        print(f"\n预览第一个表 {tables[0]}：")
        table_name = tables[0].split(".", 1)[-1]
        pg.get_dataframe_from_query(f'SELECT * FROM "{table_name}"', limit=5)

    pg.close_connection()


if __name__ == "__main__":
    test_connection()
