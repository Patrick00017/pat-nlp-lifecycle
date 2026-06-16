import sys, json
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
helper.cursor.execute("""
    SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'T_Log' ORDER BY ORDINAL_POSITION
""")
for row in helper.cursor.fetchall():
    print(f'{row[0]:25s} {row[1]:15s} len={str(row[2]):8s} nullable={row[3]}')
helper.close_connection()
