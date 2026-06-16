import sys
sys.path.insert(0, 'D:/code/pat-nlp-lifecycle/app/agent_web_app/bts_analysis_sandbox')
from database_utils import SQLServerHelper
from utils import load_config

cfg = load_config('D:/code/pat-nlp-lifecycle/app/agent_web_app/bts_analysis_sandbox/config.yaml')
db_cfg = cfg['basedatabase']
helper = SQLServerHelper(
    server=db_cfg['server'], port=db_cfg['port'],
    database=db_cfg['database'],
    username=db_cfg['username'], password=db_cfg['password']
)
helper.connect()

# Search for warp-related log messages
helper.cursor.execute("""
    SELECT TOP 50 Id, Date, Message, Exception
    FROM T_Log
    WHERE Message LIKE '%warp%' OR Message LIKE '%Warp%' OR Message LIKE '%弯翘%' OR Message LIKE '%WARP%'
    ORDER BY Date DESC
""")

rows = helper.cursor.fetchall()
print(f'找到 {len(rows)} 条相关日志\n')

for row in rows:
    print(f"Id={row[0]}  Date={row[1]}")
    print(f"  Message: {row[2][:200]}")
    if row[3]:
        print(f"  Exception: {row[3][:200]}")
    print()

helper.close_connection()
