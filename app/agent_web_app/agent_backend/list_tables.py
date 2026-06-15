import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

from database_utils import SQLServerHelper
from utils import load_config

config = load_config('config.yaml')

for name, key in [('shly-rundata', 'database'), ('shly-basedb', 'basedatabase')]:
    db_config = config[key]
    print(f'\n=== {name} ({db_config["server"]}:{db_config["port"]}) ===', flush=True)
    db = SQLServerHelper(
        server=db_config['server'],
        port=db_config['port'],
        database=db_config['database'],
        username=db_config['username'],
        password=db_config['password'],
    )
    try:
        db.connect()
        print(f'Current DB: {db.get_current_database()}', flush=True)
        
        # Manually query without emoji
        db.cursor.execute("""
            SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE
            FROM INFORMATION_SCHEMA.TABLES
            ORDER BY TABLE_TYPE, TABLE_SCHEMA, TABLE_NAME
        """)
        tables = []
        views = []
        for row in db.cursor.fetchall():
            full_name = f"{row[0]}.{row[1]}"
            if row[2] == 'BASE TABLE':
                tables.append(full_name)
            else:
                views.append(full_name)
        
        print(f"Tables ({len(tables)}):", flush=True)
        for t in tables:
            print(f"  - {t}", flush=True)
        print(f"Views ({len(views)}):", flush=True)
        for v in views:
            print(f"  - {v}", flush=True)
        
        db.close_connection()
    except Exception as e:
        print(f'Error: {e}', flush=True)
