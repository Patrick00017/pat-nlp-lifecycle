"""
测试 dev_base_glue_template.csv 模版匹配
用法: conda run -n pat-nlp-lifecycle python test\test_template.py [start_time] [end_time]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from log_parser import LogParser
from event_extractor import GlueEventExtractor
from database_utils import PostgreSQLHelper
import pandas as pd


def test_dev_base_glue_template(start_time, end_time):
    log_parser = LogParser("log_data/hnhy_glue_template.csv")
    extractor = GlueEventExtractor()

    pg = PostgreSQLHelper.from_connection_string(
        "PORT=5433;DATABASE=BaseDB;HOST=127.0.0.1;PASSWORD=123456;USER ID=postgres"
    )
    pg.connect()

    try:
        df_ips = pg.get_dataframe_from_query(
            'SELECT "Message", "Date" FROM "T_Log" WHERE "Logger" = %s AND "Date" >= %s AND "Date" < %s',
            ("BTS.Server.IPSMainCtrl", start_time, end_time),
        )
        df_ips = df_ips[~df_ips["Message"].str.contains("弯翘判定模块", na=False)]
        df_ips = df_ips[~df_ips["Message"].str.contains("服务端", na=False)]
        df_ips = df_ips[~df_ips["Message"].str.contains("WrapInfo", na=False)]

        df_glue = pg.get_dataframe_from_query(
            'SELECT "Message", "Date" FROM "T_Log" WHERE "Logger" = %s AND "Date" >= %s AND "Date" < %s',
            ("BTS.Server.GlueCtrl", start_time, end_time),
        )

        df = pd.concat([df_ips, df_glue], ignore_index=True)
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)

        df["Message"] = df["Message"].str.replace(
            '"Ip":"","Host":"","UserName":""',
            '"Ip":" ","Host":" ","UserName":" "',
            regex=False,
        )
        df["Message"] = df["Message"].str.replace(
            '"ExceptionInfo":""',
            '"ExceptionInfo":" "',
            regex=False,
        )
        df["Message"] = df["Message"].str.replace(
            "品牌LS0=,品牌MS1=,品牌LS1=,品牌MS2=,品牌LS2=,品牌MS3=,品牌LS3=",
            "品牌LS0= ,品牌MS1= ,品牌LS1= ,品牌MS2= ,品牌LS2= ,品牌MS3= ,品牌LS3=",
            regex=False,
        )

        parsed_df = log_parser.match_messages(df)
        for _, row in parsed_df.iterrows():
            extractor.process(row)

        total = len(parsed_df)
        matched = parsed_df["EventId"].notna().sum()
        unmatched = total - matched
        print(f"\n{'='*50}")
        print(f"  总日志行数: {total}")
        print(f"  匹配成功:   {matched}")
        print(f"  匹配失败:   {unmatched}")
        print(f"{'='*50}")

        print(f"\n  各模版匹配数量:")
        print(f"  {'EventId'} {'数量'}")
        print(f"  {'-'*16}")
        print(extractor.event_count_dict)
        # for eid in sorted(extractor.event_count_dict.keys()):
            # print(eid)
            # print(f"  {eid} {extractor.event_count_dict[eid]}")

        none_rows = parsed_df[parsed_df["EventId"].isna()]
        if len(none_rows) > 0:
            none_rows.to_csv("./none.csv", index=False)
            print(f"\n  未匹配消息已保存至 none.csv")
            print(f"  前 3 条未匹配:")
            for _, row in none_rows.head(3).iterrows():
                trunc = row["Message"][:120]
                print(f"    [{row['Date']}] {trunc}...")

        return extractor
    finally:
        pg.close_connection()


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        test_dev_base_glue_template(sys.argv[1], sys.argv[2])
    else:
        test_dev_base_glue_template("2026-06-05 15:00:00", "2026-06-05 17:00:00")
