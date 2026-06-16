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
        
        print("\n📊 表 (Tables):")
        for table in tables:
            print(f"  - {table}")

        print("\n👁️ 视图 (Views):")
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