from utils import load_config
from database_utils import SQLServerHelper, PostgreSQLHelper
import pandas as pd
from parse import parse
from event_extractor import (
    GlueEventExtractor,
    VacuumBlowerEventExtractor,
    SPTensionEventExtractor,
    PressrollMPEventExtractor,
    WarpEventExtractor,
)


class LogParser:
    def __init__(self, template_path):
        self.templates = self.load_templates(template_path)
        print(self.templates.head())

    def load_templates(self, template_path):
        """Load log templates from a CSV file."""
        templates = pd.read_csv(template_path, sep="->")
        return templates

    def match_message_to_template(self, message):
        """Match a message to the most appropriate template using parse."""
        for _, row in self.templates.iterrows():
            template = row["EventTemplate"].strip()
            # print(f"match: \n {message} \n {template}")
            result = parse(template, message)
            if result:
                return row["EventId"], template, result.named
        return None, None, None

    def match_messages(self, df):
        """Match all messages in a DataFrame to templates."""
        matched_results = []
        # print(df["Date"])
        # exit()
        for idx, row in df.iterrows():  # 使用 iterrows 同时获取索引和行数据
            message = row["Message"]
            date = row["Date"]  # 获取 Date 列

            event_id, template, parsed_values = self.match_message_to_template(message)
            matched_results.append(
                {
                    "Message": message,
                    "Date": date,  # 加入 Date
                    "EventId": event_id,
                    "MatchedTemplate": template,
                    "ParsedValues": parsed_values,
                }
            )
        return pd.DataFrame(matched_results)


def test_log_template():
    # 从 YAML 文件加载配置
    config = load_config("config.yaml")
    if "basedatabase" not in config:
        print("no database info in the config file.")
        exit()
    database_config = config["basedatabase"]
    log_parser = LogParser("log_template.csv")
    # 使用封装的类
    db_helper = SQLServerHelper(
        server=database_config["server"],
        port=database_config["port"],
        database=database_config["database"],
        username=database_config["username"],
        password=database_config["password"],
    )
    try:
        db_helper.connect()
        db_helper.get_current_database()
        db_helper.list_tables_and_views()
        tlog_df = db_helper.get_dataframe_from_table_and_limit("dbo.T_Log", 100)
        parsed_message_df = log_parser.match_messages(tlog_df)
        print(parsed_message_df.head())

    finally:
        db_helper.close_connection()


def test_gluecontrol_template():
    # 从 YAML 文件加载配置
    config = load_config("config.yaml")
    if "basedatabase" not in config:
        print("no database info in the config file.")
        exit()
    database_config = config["basedatabase"]
    log_parser = LogParser("log_data/gluecontrol_template.csv")
    # 使用封装的类
    db_helper = SQLServerHelper(
        server=database_config["server"],
        port=database_config["port"],
        database=database_config["database"],
        username=database_config["username"],
        password=database_config["password"],
    )
    try:
        db_helper.connect()
        db_helper.get_current_database()
        db_helper.list_tables_and_views()
        tlog_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.GlueCtrl",
            limit=100,
        )
        parsed_message_df = log_parser.match_messages(tlog_df)
        print(parsed_message_df.head())
        none_rows = parsed_message_df[parsed_message_df["EventId"].isna()]
        none_rows.to_csv("./none.csv")
        # print(none_rows)

    finally:
        db_helper.close_connection()


def test_ips_template():
    # 从 YAML 文件加载配置
    config = load_config("config.yaml")
    if "basedatabase" not in config:
        print("no database info in the config file.")
        exit()
    database_config = config["basedatabase"]
    log_parser = LogParser("log_data/ips_template.csv")
    # 使用封装的类
    db_helper = SQLServerHelper(
        server=database_config["server"],
        port=database_config["port"],
        database=database_config["database"],
        username=database_config["username"],
        password=database_config["password"],
    )
    try:
        db_helper.connect()
        db_helper.get_current_database()
        db_helper.list_tables_and_views()
        ips_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.IPSMainCtrl",
            limit=10000,
        )
        ips_df = ips_df[~ips_df["Message"].str.contains("弯翘判定模块", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("服务端", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("WrapInfo", na=False)]
        glue_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.GlueCtrl",
            limit=100,
        )
        df = pd.concat([ips_df, glue_df], ignore_index=True)
        print(df.head())
        parsed_df = log_parser.match_messages(df)
        print(parsed_df.head())
        none_rows = parsed_df[parsed_df["EventId"].isna()]
        none_rows.to_csv("./none.csv")
    finally:
        db_helper.close_connection()


def test_ips_and_glue_template(start_time, end_time):
    # 从 YAML 文件加载配置
    config = load_config("config.yaml")
    if "basedatabase" not in config:
        print("no database info in the config file.")
        exit()
    database_config = config["basedatabase"]
    log_parser = LogParser("log_data/ips_with_glue_template.csv")
    extractor = GlueEventExtractor()
    # 使用封装的类
    db_helper = SQLServerHelper(
        server=database_config["server"],
        port=database_config["port"],
        database=database_config["database"],
        username=database_config["username"],
        password=database_config["password"],
    )
    try:
        db_helper.connect()
        db_helper.get_current_database()
        db_helper.list_tables_and_views()
        ips_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.IPSMainCtrl",
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        ips_df = ips_df[~ips_df["Message"].str.contains("弯翘判定模块", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("服务端", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("WrapInfo", na=False)]
        glue_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.GlueCtrl",
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        df = pd.concat([ips_df, glue_df], ignore_index=True)

        # sort by time
        # 按time列排序（假设time是Date列的别名，或数据中有时time列）
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)

        parsed_message_df = log_parser.match_messages(df)
        print(parsed_message_df.head())
        none_rows = parsed_message_df[parsed_message_df["EventId"].isna()]
        none_rows.to_csv("./none.csv")
        for index, row in parsed_message_df.iterrows():
            # extractor.process_log_row(row)
            extractor.process(row)
        return extractor
        print(len(extractor.current_material_events))
        for event in extractor.current_material_events:
            # if event['part'] == 'ls0':
            #     print(event)
            print(event)
        # print(extractor.current_material_events)

    finally:
        db_helper.close_connection()


def test_ips_and_glue_template_pg(start_time, end_time):
    log_parser = LogParser("log_data/hnhy_glue_template.csv")
    extractor = GlueEventExtractor()

    pg = PostgreSQLHelper.from_connection_string(
        "PORT=5433;DATABASE=HNHY_BaseDB;HOST=127.0.0.1;PASSWORD=123456;USER ID=postgres"
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
        df_ips = df_ips[~df_ips["Message"].str.contains("弯翘判定模块", na=False)]
        df_ips = df_ips[~df_ips["Message"].str.contains("服务端", na=False)]
        df_ips = df_ips[~df_ips["Message"].str.contains("WrapInfo", na=False)]

        # df_glue = pg.get_dataframe_from_query(
        #     'SELECT "Message", "Date" FROM "T_Log" WHERE "Logger" = %s AND "Date" >= %s AND "Date" < %s LIMIT 1000',
        #     ("BTS.Server.Start.IPSBizs.NewCtrl.GlueCtrl", start_time, end_time),
        # )

        df = pd.concat([df_ips, df_glue], ignore_index=True)
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)
        df.to_csv("./pg_raw_messages.csv", index=False)

        # 查询每个订单首次出现的行（初始化数据）
        try:
            pg_ips = PostgreSQLHelper.from_connection_string(
                "PORT=5433;DATABASE=HNHY_IPS;HOST=127.0.0.1;PASSWORD=123456;USER ID=postgres"
            )
            pg_ips.connect()
            df_orders = pg_ips.get_dataframe_from_query(
                'SELECT DISTINCT ON ("F_OrderID") '
                '"F_CreateTime", "F_OrderID", "F_MachineID", '
                '"F_PaperCode", "F_Flute", "F_Width", '
                '"F_ErpPaperCode", "F_ErpWeight", "F_ErpWidth" '
                'FROM "T_IPS_HisRunningData_20260622" '
                'WHERE "F_CreateTime" >= %s AND "F_CreateTime" < %s '
                'ORDER BY "F_OrderID", "F_CreateTime"',
                (start_time, end_time),
            )
            extractor.order_init_data = df_orders.to_dict('records')
            # 查询各机台纸卷剩余量数据
            df_machine_run = pg_ips.get_dataframe_from_query(
                'SELECT "F_CreateTime", "F_OrderID", "F_MachineID", "F_Remainning_mm" '
                'FROM "T_IPS_HisRunningData_20260622" '
                'WHERE "F_MachineID" IN (%s, %s, %s, %s, %s, %s, %s) '
                'AND "F_CreateTime" >= %s AND "F_CreateTime" < %s '
                'ORDER BY "F_CreateTime"',
                ("LS0", "MS1", "LS1", "MS2", "LS2", "MS3", "LS3", start_time, end_time),
            )
            extractor.machine_run_data = df_machine_run.to_dict('records')
            pg_ips.close_connection()
        except Exception as e:
            print(f"Failed to query order init data: {e}")
            extractor.order_init_data = []

        # 预处理：将空字符串替换为空格，避免 parse 库空捕获 bug
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

        parsed_message_df = log_parser.match_messages(df)
        # print(parsed_message_df.head())
        none_rows = parsed_message_df[parsed_message_df["EventId"].isna()]
        if len(none_rows) > 0:
            none_rows.to_csv("./none.csv")
        for index, row in parsed_message_df.iterrows():
            extractor.process(row)
        extractor.process_orders()
        return extractor
    finally:
        pg.close_connection()

def test_from_csv():
    log_parser = LogParser("log_data/dev_base_glue_template.csv")
    extractor = GlueEventExtractor()

    # df = pd.read_csv("test_template_examples.csv")
    df = pd.read_csv("pg_raw_messages.csv")
    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)
    # 预处理：将空字符串替换为空格，避免 parse 库空捕获 bug
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
    parsed_message_df = log_parser.match_messages(df)
    print(parsed_message_df.head())
    none_rows = parsed_message_df[parsed_message_df["EventId"].isna()]
    if len(none_rows) > 0:
        none_rows.to_csv("./none.csv")
    for index, row in parsed_message_df.iterrows():
        extractor.process(row)
    print(extractor.event_count_dict)
    return extractor


def test_hotspray_template(start_time, end_time):
    # 从 YAML 文件加载配置
    config = load_config("config.yaml")
    if "basedatabase" not in config:
        print("no database info in the config file.")
        exit()
    database_config = config["basedatabase"]
    log_parser = LogParser("log_data/hotspray.csv")
    extractor = GlueEventExtractor()
    # 使用封装的类
    db_helper = SQLServerHelper(
        server=database_config["server"],
        port=database_config["port"],
        database=database_config["database"],
        username=database_config["username"],
        password=database_config["password"],
    )
    try:
        db_helper.connect()
        db_helper.get_current_database()
        db_helper.list_tables_and_views()
        ips_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.IPSMainCtrl",
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        ips_df = ips_df[~ips_df["Message"].str.contains("弯翘判定模块", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("服务端", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("WrapInfo", na=False)]
        glue_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.HotSprayCtrl",
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        df = pd.concat([ips_df, glue_df], ignore_index=True)

        # sort by time
        # 按time列排序（假设time是Date列的别名，或数据中有时time列）
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)

        parsed_message_df = log_parser.match_messages(df)
        print(parsed_message_df.head())
        none_rows = parsed_message_df[parsed_message_df["EventId"].isna()]
        none_rows.to_csv("./none.csv")
        for index, row in parsed_message_df.iterrows():
            # extractor.process_log_row(row)
            extractor.process(row)
        return extractor
        print(len(extractor.current_material_events))
        for event in extractor.current_material_events:
            # if event['part'] == 'ls0':
            #     print(event)
            print(event)
        # print(extractor.current_material_events)

    finally:
        db_helper.close_connection()


def test_bridge_tension_template(start_time, end_time):
    # 从 YAML 文件加载配置
    config = load_config("config.yaml")
    if "basedatabase" not in config:
        print("no database info in the config file.")
        exit()
    database_config = config["basedatabase"]
    log_parser = LogParser("log_data/bridge_tension.csv")
    extractor = GlueEventExtractor()
    # 使用封装的类
    db_helper = SQLServerHelper(
        server=database_config["server"],
        port=database_config["port"],
        database=database_config["database"],
        username=database_config["username"],
        password=database_config["password"],
    )
    try:
        db_helper.connect()
        db_helper.get_current_database()
        db_helper.list_tables_and_views()
        ips_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.IPSMainCtrl",
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        ips_df = ips_df[~ips_df["Message"].str.contains("弯翘判定模块", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("服务端", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("WrapInfo", na=False)]
        glue_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.BridgeTensionCtrl",
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        df = pd.concat([ips_df, glue_df], ignore_index=True)

        # sort by time
        # 按time列排序（假设time是Date列的别名，或数据中有时time列）
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)

        parsed_message_df = log_parser.match_messages(df)
        print(parsed_message_df.head())
        none_rows = parsed_message_df[parsed_message_df["EventId"].isna()]
        none_rows.to_csv("./none.csv")
        for index, row in parsed_message_df.iterrows():
            extractor.process(row)
        return extractor
        print(len(extractor.current_material_events))
        for event in extractor.current_material_events:
            # if event['part'] == 'ls0':
            #     print(event)
            print(event)
        # print(extractor.current_material_events)

    finally:
        db_helper.close_connection()


def test_coldplate_press_template(start_time, end_time):
    # 从 YAML 文件加载配置
    config = load_config("config.yaml")
    if "basedatabase" not in config:
        print("no database info in the config file.")
        exit()
    database_config = config["basedatabase"]
    log_parser = LogParser("log_data/coldplate.csv")
    extractor = GlueEventExtractor()
    # 使用封装的类
    db_helper = SQLServerHelper(
        server=database_config["server"],
        port=database_config["port"],
        database=database_config["database"],
        username=database_config["username"],
        password=database_config["password"],
    )
    try:
        db_helper.connect()
        db_helper.get_current_database()
        db_helper.list_tables_and_views()
        ips_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.IPSMainCtrl",
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        ips_df = ips_df[~ips_df["Message"].str.contains("弯翘判定模块", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("服务端", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("WrapInfo", na=False)]
        glue_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.ColdPlatePressCtrl",
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        df = pd.concat([ips_df, glue_df], ignore_index=True)

        # sort by time
        # 按time列排序（假设time是Date列的别名，或数据中有时time列）
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)

        parsed_message_df = log_parser.match_messages(df)
        print(parsed_message_df.head())
        none_rows = parsed_message_df[parsed_message_df["EventId"].isna()]
        none_rows.to_csv("./none.csv")
        for index, row in parsed_message_df.iterrows():
            extractor.process(row)
        return extractor
        print(len(extractor.current_material_events))
        for event in extractor.current_material_events:
            # if event['part'] == 'ls0':
            #     print(event)
            print(event)
        # print(extractor.current_material_events)

    finally:
        db_helper.close_connection()


def test_corrugated_roll_template(start_time, end_time):
    # 从 YAML 文件加载配置
    config = load_config("config.yaml")
    if "basedatabase" not in config:
        print("no database info in the config file.")
        exit()
    database_config = config["basedatabase"]
    log_parser = LogParser("log_data/corrugated_roll.csv")
    extractor = GlueEventExtractor()
    # 使用封装的类
    db_helper = SQLServerHelper(
        server=database_config["server"],
        port=database_config["port"],
        database=database_config["database"],
        username=database_config["username"],
        password=database_config["password"],
    )
    try:
        db_helper.connect()
        db_helper.get_current_database()
        db_helper.list_tables_and_views()
        ips_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.IPSMainCtrl",
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        ips_df = ips_df[~ips_df["Message"].str.contains("弯翘判定模块", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("服务端", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("WrapInfo", na=False)]
        glue_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.CorrugatedRollCtrl",
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        df = pd.concat([ips_df, glue_df], ignore_index=True)

        # sort by time
        # 按time列排序（假设time是Date列的别名，或数据中有时time列）
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)

        parsed_message_df = log_parser.match_messages(df)
        print(parsed_message_df.head())
        none_rows = parsed_message_df[parsed_message_df["EventId"].isna()]
        none_rows.to_csv("./none.csv")
        for index, row in parsed_message_df.iterrows():
            extractor.process(row)
        return extractor
        print(len(extractor.current_material_events))
        for event in extractor.current_material_events:
            # if event['part'] == 'ls0':
            #     print(event)
            print(event)
        # print(extractor.current_material_events)

    finally:
        db_helper.close_connection()


def test_hotloadgroup_template(start_time, end_time):
    # 从 YAML 文件加载配置
    config = load_config("config.yaml")
    if "basedatabase" not in config:
        print("no database info in the config file.")
        exit()
    database_config = config["basedatabase"]
    log_parser = LogParser("log_data/hot_load_group.csv")
    extractor = GlueEventExtractor()
    # 使用封装的类
    db_helper = SQLServerHelper(
        server=database_config["server"],
        port=database_config["port"],
        database=database_config["database"],
        username=database_config["username"],
        password=database_config["password"],
    )
    try:
        db_helper.connect()
        db_helper.get_current_database()
        db_helper.list_tables_and_views()
        ips_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.IPSMainCtrl",
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        ips_df = ips_df[~ips_df["Message"].str.contains("弯翘判定模块", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("服务端", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("WrapInfo", na=False)]
        glue_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.HotLoadGroupQtyCtrl",
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        df = pd.concat([ips_df, glue_df], ignore_index=True)

        # sort by time
        # 按time列排序（假设time是Date列的别名，或数据中有时time列）
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)

        parsed_message_df = log_parser.match_messages(df)
        print(parsed_message_df.head())
        none_rows = parsed_message_df[parsed_message_df["EventId"].isna()]
        none_rows.to_csv("./none.csv")
        for index, row in parsed_message_df.iterrows():
            extractor.process(row)
        return extractor
        print(len(extractor.current_material_events))
        for event in extractor.current_material_events:
            # if event['part'] == 'ls0':
            #     print(event)
            print(event)
        # print(extractor.current_material_events)

    finally:
        db_helper.close_connection()


def test_hotplate_press_template(start_time, end_time):
    # 从 YAML 文件加载配置
    config = load_config("config.yaml")
    if "basedatabase" not in config:
        print("no database info in the config file.")
        exit()
    database_config = config["basedatabase"]
    log_parser = LogParser("log_data/hot_plate_press.csv")
    extractor = GlueEventExtractor()
    # 使用封装的类
    db_helper = SQLServerHelper(
        server=database_config["server"],
        port=database_config["port"],
        database=database_config["database"],
        username=database_config["username"],
        password=database_config["password"],
    )
    try:
        db_helper.connect()
        db_helper.get_current_database()
        db_helper.list_tables_and_views()
        ips_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.IPSMainCtrl",
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        ips_df = ips_df[~ips_df["Message"].str.contains("弯翘判定模块", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("服务端", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("WrapInfo", na=False)]
        glue_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.HotPlatePressCtrl",
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        df = pd.concat([ips_df, glue_df], ignore_index=True)

        # sort by time
        # 按time列排序（假设time是Date列的别名，或数据中有时time列）
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)

        parsed_message_df = log_parser.match_messages(df)
        print(parsed_message_df.head())
        none_rows = parsed_message_df[parsed_message_df["EventId"].isna()]
        none_rows.to_csv("./none.csv")
        for index, row in parsed_message_df.iterrows():
            extractor.process(row)
        return extractor
        print(len(extractor.current_material_events))
        for event in extractor.current_material_events:
            # if event['part'] == 'ls0':
            #     print(event)
            print(event)
        # print(extractor.current_material_events)

    finally:
        db_helper.close_connection()


def test_pressroll_mp_template(start_time, end_time):
    # 从 YAML 文件加载配置
    config = load_config("config.yaml")
    if "basedatabase" not in config:
        print("no database info in the config file.")
        exit()
    database_config = config["basedatabase"]
    log_parser = LogParser("log_data/pressroll_mp.csv")
    extractor = PressrollMPEventExtractor()
    # 使用封装的类
    db_helper = SQLServerHelper(
        server=database_config["server"],
        port=database_config["port"],
        database=database_config["database"],
        username=database_config["username"],
        password=database_config["password"],
    )
    try:
        db_helper.connect()
        db_helper.get_current_database()
        db_helper.list_tables_and_views()
        ips_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.IPSMainCtrl",
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        ips_df = ips_df[~ips_df["Message"].str.contains("弯翘判定模块", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("服务端", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("WrapInfo", na=False)]
        glue_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.PressRollMPCtrl",
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        df = pd.concat([ips_df, glue_df], ignore_index=True)

        # sort by time
        # 按time列排序（假设time是Date列的别名，或数据中有时time列）
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)

        parsed_message_df = log_parser.match_messages(df)
        print(parsed_message_df.head())
        none_rows = parsed_message_df[parsed_message_df["EventId"].isna()]
        none_rows.to_csv("./none.csv")
        for index, row in parsed_message_df.iterrows():
            extractor.process(row)
        return extractor
        print(len(extractor.current_material_events))
        for event in extractor.current_material_events:
            # if event['part'] == 'ls0':
            #     print(event)
            print(event)
        # print(extractor.current_material_events)

    finally:
        db_helper.close_connection()


def test_sptension_template(start_time, end_time):
    # 从 YAML 文件加载配置
    config = load_config("config.yaml")
    if "basedatabase" not in config:
        print("no database info in the config file.")
        exit()
    database_config = config["basedatabase"]
    log_parser = LogParser("log_data/sptension.csv")
    extractor = SPTensionEventExtractor()
    # 使用封装的类
    db_helper = SQLServerHelper(
        server=database_config["server"],
        port=database_config["port"],
        database=database_config["database"],
        username=database_config["username"],
        password=database_config["password"],
    )
    try:
        db_helper.connect()
        db_helper.get_current_database()
        db_helper.list_tables_and_views()
        ips_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.IPSMainCtrl",
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        ips_df = ips_df[~ips_df["Message"].str.contains("弯翘判定模块", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("服务端", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("WrapInfo", na=False)]
        glue_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.SPTensionCtrl",
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        df = pd.concat([ips_df, glue_df], ignore_index=True)

        # sort by time
        # 按time列排序（假设time是Date列的别名，或数据中有时time列）
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)

        parsed_message_df = log_parser.match_messages(df)
        print(parsed_message_df.head())
        none_rows = parsed_message_df[parsed_message_df["EventId"].isna()]
        none_rows.to_csv("./none.csv")
        for index, row in parsed_message_df.iterrows():
            extractor.process(row)
        return extractor
        print(len(extractor.current_material_events))
        for event in extractor.current_material_events:
            # if event['part'] == 'ls0':
            #     print(event)
            print(event)
        # print(extractor.current_material_events)

    finally:
        db_helper.close_connection()


def test_vacuum_blower_template(start_time, end_time):
    # 从 YAML 文件加载配置
    config = load_config("config.yaml")
    if "basedatabase" not in config:
        print("no database info in the config file.")
        exit()
    database_config = config["basedatabase"]
    log_parser = LogParser("log_data/vacuum_blower.csv")
    extractor = VacuumBlowerEventExtractor()
    # 使用封装的类
    db_helper = SQLServerHelper(
        server=database_config["server"],
        port=database_config["port"],
        database=database_config["database"],
        username=database_config["username"],
        password=database_config["password"],
    )
    try:
        db_helper.connect()
        db_helper.get_current_database()
        db_helper.list_tables_and_views()
        ips_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.IPSMainCtrl",
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        ips_df = ips_df[~ips_df["Message"].str.contains("弯翘判定模块", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("服务端", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("WrapInfo", na=False)]
        glue_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.VacuumBlowerCtrl",
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        df = pd.concat([ips_df, glue_df], ignore_index=True)

        # sort by time
        # 按time列排序（假设time是Date列的别名，或数据中有时time列）
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)

        parsed_message_df = log_parser.match_messages(df)
        print(parsed_message_df.head())
        none_rows = parsed_message_df[parsed_message_df["EventId"].isna()]
        none_rows.to_csv("./none.csv")
        for index, row in parsed_message_df.iterrows():
            extractor.process(row)
        return extractor
        print(len(extractor.current_material_events))
        for event in extractor.current_material_events:
            # if event['part'] == 'ls0':
            #     print(event)
            print(event)
        # print(extractor.current_material_events)

    finally:
        db_helper.close_connection()


def test_wrap_template(start_time, end_time):
    config = load_config("config.yaml")
    if "basedatabase" not in config:
        print("no database info in the config file.")
        exit()
    database_config = config["basedatabase"]
    log_parser = LogParser("log_data/warp_template.csv")
    extractor = WarpEventExtractor()
    db_helper = SQLServerHelper(
        server=database_config["server"],
        port=database_config["port"],
        database=database_config["database"],
        username=database_config["username"],
        password=database_config["password"],
    )
    try:
        db_helper.connect()
        db_helper.get_current_database()
        db_helper.list_tables_and_views()
        db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.IPSMainCtrl",
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        warp_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.PMSDataService",
            start_time=start_time,
            end_time=end_time,
            limit=5000,
        )
        warp_df = warp_df[
            warp_df["Message"].str.contains("弯翘判定模块", na=False)
        ].copy()
        if "Date" in warp_df.columns:
            warp_df = warp_df.sort_values("Date").reset_index(drop=True)
        print(f"shape: {warp_df.shape}")
        parsed_message_df = log_parser.match_messages(warp_df)
        print(parsed_message_df.head())
        none_rows = parsed_message_df[parsed_message_df["EventId"].isna()]
        if len(none_rows) > 0:
            none_rows.to_csv("./none.csv")
        for index, row in parsed_message_df.iterrows():
            extractor.process(row)
        return extractor
    finally:
        db_helper.close_connection()


def test_riding_roll_template(start_time, end_time):
    # 从 YAML 文件加载配置
    config = load_config("config.yaml")
    if "basedatabase" not in config:
        print("no database info in the config file.")
        exit()
    database_config = config["basedatabase"]
    log_parser = LogParser("log_data/riding_roll.csv")
    extractor = GlueEventExtractor()
    # 使用封装的类
    db_helper = SQLServerHelper(
        server=database_config["server"],
        port=database_config["port"],
        database=database_config["database"],
        username=database_config["username"],
        password=database_config["password"],
    )
    try:
        db_helper.connect()
        db_helper.get_current_database()
        db_helper.list_tables_and_views()
        ips_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.NewCtrl.IPSMainCtrl",
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        ips_df = ips_df[~ips_df["Message"].str.contains("弯翘判定模块", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("服务端", na=False)]
        ips_df = ips_df[~ips_df["Message"].str.contains("WrapInfo", na=False)]
        glue_df = db_helper.get_dataframe_filter_by_module(
            table_name="dbo.T_Log",
            module_name="BTS.Server.Start.IPSBizs.RidingRollCtrl",
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        df = pd.concat([ips_df, glue_df], ignore_index=True)

        # sort by time
        # 按time列排序（假设time是Date列的别名，或数据中有时time列）
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)

        parsed_message_df = log_parser.match_messages(df)
        print(parsed_message_df.head())
        none_rows = parsed_message_df[parsed_message_df["EventId"].isna()]
        none_rows.to_csv("./none.csv")
        for index, row in parsed_message_df.iterrows():
            extractor.process(row)
        return extractor
        print(len(extractor.current_material_events))
        for event in extractor.current_material_events:
            # if event['part'] == 'ls0':
            #     print(event)
            print(event)
        # print(extractor.current_material_events)

    finally:
        db_helper.close_connection()


if __name__ == "__main__":
    # test_log_template()
    # test_gluecontrol_template()
    # test_ips_template()
    # test_hotspray_template(start_time="2026-01-08 14:03:50.690", end_time="2026-01-09 15:03:50.690")
    # test_bridge_tension_template(start_time="2026-01-08 14:03:50.690", end_time="2026-01-09 15:03:50.690")
    # test_coldplate_press_template(start_time="2026-01-08 14:03:50.690", end_time="2026-01-09 15:03:50.690")
    # test_corrugated_roll_template(start_time="2026-01-08 14:03:50.690", end_time="2026-01-09 15:03:50.690")
    # test_hotloadgroup_template(start_time="2026-01-08 14:03:50.690", end_time="2026-01-09 15:03:50.690")
    # test_hotplate_press_template(start_time="2026-01-08 14:03:50.690", end_time="2026-01-09 15:03:50.690")
    # test_pressroll_mp_template(start_time="2026-01-08 14:03:50.690", end_time="2026-01-09 15:03:50.690")
    # test_sptension_template(start_time="2026-01-08 14:03:50.690", end_time="2026-01-09 15:03:50.690")
    # test_vacuum_blower_template(start_time="2026-01-08 14:03:50.690", end_time="2026-01-09 15:03:50.690")
    # test_wrap_template(start_time="2026-01-08 14:03:50.690", end_time="2026-01-09 15:03:50.690") # 存在问题, 上中下层无法匹配
    # test_riding_roll_template(start_time="2026-01-08 14:03:50.690", end_time="2026-01-09 15:03:50.690") # 存在问题，上中下层无法匹配

    # 胶水测试
    # extractor: GlueEventExtractor = test_ips_and_glue_template(start_time="2026-01-08 14:03:50.690", end_time="2026-01-08 15:03:50.690")
    # results = extractor.get_glue_set_function_full_event()
    # print(results)

    # 胶水测试baseDEV
    extractor: GlueEventExtractor = test_ips_and_glue_template_pg(
        start_time="2026-06-22 00:00:00.000", end_time="2026-06-22 01:00:00.000"
    )
    # extractor: GlueEventExtractor = test_from_csv()
    # results = extractor.get_glue_set_function_full_event()
    results = extractor.get_all_events()
    # print(results)
    from fsm_engine import GlueGapDiagnosticFSM

    fsm = GlueGapDiagnosticFSM(extractor)
    fsm.run()
    fsm.get_results()
    fsm.save_results(filepath='./environments/fsm_results.json')
    # fsm_data = fsm.generate_json()
    # print(f"\nFSM 引擎运行完成：{len(fsm_data.get('cycles', []))} 个周期")
    # for c in fsm_data.get("cycles", []):
    #     print(f"  {c['position']}#{c['index']} {c['status']['id']} {c['trigger']['label']} mat={c['material']}")
    #     if c.get("errors"):
    #         for e in c["errors"]:
    #             print(f"    错误: {e['label']}: {e['detail']}")
    #     if c.get("warnings"):
    #         print(f"    警告: {', '.join(set(c['warnings']))}")
    # fsm_report = fsm.generate_report()
    # with open("fsm_report.md", "w", encoding="utf-8") as f:
    #     f.write(fsm_report)
    # print("FSM 报告已保存到 fsm_report.md")

    # 弯翘测试
    # extractor: WarpEventExtractor = test_wrap_template(start_time="2026-01-08 14:00:00.000", end_time="2026-01-08 15:00:00.000")
    # print("\n=== 弯翘事件统计 ===")
    # print(extractor.get_summary())

    # 真空泵测试
    # extractor: VacuumBlowerEventExtractor = test_vacuum_blower_template(start_time="2026-01-08 14:03:50.690", end_time="2026-01-09 15:03:50.690")
    # print(extractor.set_func_call_events)
    # results, markdown_results = extractor.get_vacuum_blower_set_function_full_event()
    # print(results)
    # print(markdown_results)
    # with open('setglue_sf2_report.md', 'w', encoding='utf-8') as f:
    #     f.write(markdown_results)

    # 接纸机张力测试
    # extractor: SPTensionEventExtractor = test_sptension_template(start_time="2026-01-10 14:03:50.690", end_time="2026-01-11 15:03:50.690")
    # print(extractor.set_func_call_events)
    # results, markdown_results = extractor.get_vacuum_blower_set_function_full_event()
    # print(results)
    # print(markdown_results)
    # with open('setglue_sf2_report.md', 'w', encoding='utf-8') as f:
    #     f.write(markdown_results)

    # MP压力辊
    # extractor: PressrollMPEventExtractor = test_pressroll_mp_template(start_time="2026-01-10 14:03:50.690", end_time="2026-01-11 15:03:50.690")
    # print("material change event")
    # print(extractor.material_events)
    # print("set func call event")
    # print(extractor.set_func_call_events)
    # results, markdown_results = extractor.get_vacuum_blower_set_function_full_event()
    # print(results)
    # print(markdown_results)
    # with open('setglue_sf2_report.md', 'w', encoding='utf-8') as f:
    #     f.write(markdown_results)
