import pymssql
from utils import load_config
import pandas as pd

class SQLServerHelper:
    def __init__(self, server, port, database, username, password, tds_version="7.0"):
        self.server = server
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.tds_version = tds_version
        self.conn = None
        self.cursor = None

    def connect(self):
        """Connect to the SQL Server database."""
        try:
            self.conn = pymssql.connect(
                server=self.server,
                port=self.port,
                user=self.username,
                password=self.password,
                database=self.database,
                tds_version=self.tds_version,
                charset='cp936'
            )
            self.cursor = self.conn.cursor()
            print("连接成功！")
        except pymssql.DatabaseError as e:
            print(f"数据库连接失败: {e}")
            raise

    def get_current_database(self):
        """Retrieve the name of the current database."""
        self.cursor.execute("SELECT DB_NAME() AS CurrentDatabase")
        current_db = self.cursor.fetchone()[0]
        print(f"当前连接的数据库: {current_db}")
        return current_db

    def list_tables_and_views(self):
        """List all tables and views in the database."""
        self.cursor.execute("""
            SELECT 
                TABLE_SCHEMA,
                TABLE_NAME,
                TABLE_TYPE
            FROM INFORMATION_SCHEMA.TABLES 
            ORDER BY TABLE_TYPE, TABLE_SCHEMA, TABLE_NAME
        """)
        tables = []
        views = []
        for row in self.cursor.fetchall():
            schema = row[0]
            name = row[1]
            type_ = row[2]
            if type_ == 'BASE TABLE':
                tables.append(f"{schema}.{name}")
            else:
                views.append(f"{schema}.{name}")
        
        print("\n[表 (Tables)]:")
        for table in tables:
            print(f"  - {table}")

        print("\n[视图 (Views)]:")
        for view in views:
            print(f"  - {view}")
        return tables, views

    def close_connection(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("数据库连接已关闭。")
    
    def get_table_data_by_limit(self, table_name, limit=10):
        """Fetch data from a specified table."""
        query = f"SELECT TOP {limit} * FROM {table_name}"
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        for row in rows:
            print(row)
        return rows

    def get_dataframe_from_table_and_limit(self, table_name, limit=10):
        """Fetch data from a specified table and convert to pandas dataframe."""
        query = f"SELECT TOP {limit} * FROM {table_name}"
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        # 获取列名
        columns = [column[0] for column in self.cursor.description]
        # 转换为 DataFrame
        df = pd.DataFrame(rows, columns=columns)
        print(df.head())
        print(f'shape: {df.shape}')
        return df
    
    def get_dataframe_filter_by_module(self, table_name, module_name, start_time, end_time, limit=10):
        """Fetch data from a specified table and convert to pandas dataframe."""
        query = f"SELECT TOP {limit} * FROM {table_name} where [Logger]='{module_name}' AND [Date] >= '{start_time}' And [Date] < '{end_time}'"
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        # 获取列名
        columns = [column[0] for column in self.cursor.description]
        # 转换为 DataFrame
        df = pd.DataFrame(rows, columns=columns)
        print(df.head())
        print(f'shape: {df.shape}')
        return df


class PostgreSQLHelper:
    """PostgreSQL 数据库连接与查询辅助类。"""

    def __init__(self, host, port, database, user, password):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.conn = None
        self.cursor = None

    @classmethod
    def from_connection_string(cls, conn_str):
        """解析 ODBC 风格连接字符串，例如：
        PORT=5432;DATABASE=devBaseDB;HOST=192.168.110.82;PASSWORD=123456;USER ID=postgres
        """
        params = {}
        for part in conn_str.split(';'):
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip().upper().replace(' ', '_')
                params[key] = value.strip()
        return cls(
            host=params.get('HOST', 'localhost'),
            port=int(params.get('PORT', 5432)),
            database=params.get('DATABASE', ''),
            user=params.get('USER_ID', params.get('USER', 'postgres')),
            password=params.get('PASSWORD', ''),
        )

    def connect(self):
        """连接到 PostgreSQL 数据库。"""
        import psycopg2
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.database,
                user=self.user,
                password=self.password,
            )
            self.cursor = self.conn.cursor()
            print("PostgreSQL 连接成功！")
        except Exception as e:
            print(f"PostgreSQL 连接失败: {e}")
            raise

    def get_current_database(self):
        """获取当前数据库名称。"""
        self.cursor.execute("SELECT current_database()")
        current_db = self.cursor.fetchone()[0]
        print(f"当前连接的数据库: {current_db}")
        return current_db

    def list_tables_and_views(self):
        """列出数据库中所有用户表与视图（排除系统 schema）。"""
        self.cursor.execute("""
            SELECT table_schema, table_name, table_type
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_type, table_schema, table_name
        """)
        tables = []
        views = []
        for row in self.cursor.fetchall():
            schema = row[0]
            name = row[1]
            type_ = row[2]
            if type_ == 'BASE TABLE':
                tables.append(f"{schema}.{name}")
            else:
                views.append(f"{schema}.{name}")

        print("\n[表 (Tables)]:")
        for t in tables:
            print(f"  - {t}")

        print("\n[视图 (Views)]:")
        for v in views:
            print(f"  - {v}")
        return tables, views

    def close_connection(self):
        """关闭数据库连接。"""
        if self.conn:
            self.conn.close()
            print("PostgreSQL 连接已关闭。")

    def get_dataframe_from_query(self, query, params=None, limit=None):
        """执行查询并返回 pandas DataFrame。"""
        sql = query.rstrip(';')
        if limit is not None and 'LIMIT' not in sql.upper():
            sql += f' LIMIT {limit}'
        self.cursor.execute(sql, params)
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        print(df.head())
        print(f'shape: {df.shape}')
        return df