from transitions import Machine

# from statemachine import State
# from statemachine import StateChart
import pandas as pd
from datetime import datetime
import numpy as np
from constant import handle_func_to_splicer_part
from utils import material_part_count


class SplicerLogStateMachine:
    """State machine for processing logs."""

    states = [
        "normal",  # 正常状态
        "paperChange",  # 换卷与延迟
        "Error",  # 错误状态 可能不存在
    ]

    def __init__(self):
        self.machine = Machine(
            model=self, states=SplicerLogStateMachine.states, initial="normal"
        )

        # 定义状态转换
        self.machine.add_transition(
            trigger="change_paper", source="normal", dest="paperChange"
        )
        self.machine.add_transition(
            trigger="change_paper_done", source="paperChange", dest="normal"
        )


class SplicerLogStateMachineWrapper:
    def __init__(self):
        self.fsm = SplicerLogStateMachine()
        self.former_change_paper_row = None
        self.former_row = None  # 用于可能需要的

    def process_log_row(self, row):
        print(row)
        if isinstance(self.former_row, pd.Series) and self.former_row.empty:
            # 状态机在最初始阶段
            self.former_row = row
            return
        if self.fsm.state == "normal":
            if row["F_IsChangePaperRoll"] == 1:
                # 拿到换卷信号
                self.former_change_paper_row = row
                self.fsm.change_paper()
        if self.fsm.state == "paperChange":
            # 对比时间，换卷完成后需要延迟多少秒回到normal状态，此处举例为5秒
            before_time = self.former_change_paper_row["F_CreateTime"]
            present_time = row["F_CreateTime"]
            # 解析字符串为datetime对象
            dt1 = datetime.strptime(before_time, "%Y-%m-%d %H:%M:%S")
            dt2 = datetime.strptime(present_time, "%Y-%m-%d %H:%M:%S")
            time_diff = dt2 - dt1
            seconds = time_diff.total_seconds()
            if seconds > 5:
                self.fsm.change_paper_done()
        self.former_row = row

    def get_state(self):
        return self.fsm.state


# class GlueRunParStateMachine(StateChart):
#     qdm_factor_vec = np.zeros(6, dtype=np.float32) # [sf1_qdm_factor, sf2_qdm_factor, sf3_qdm_factor, gu1_qdm_factor, gu2_qdm_factor, gu3_qdm_factor]
#     glue_ui_factor_vec = np.zeros(6, dtype=np.float32) # [sf1_ui_factor, sf2_ui_factor, sf3_ui_factor, gu1_ui_factor, gu2_ui_factor, gu3_ui_factor]
#     glue_switch_vec = np.zeros(6, dtype=np.int8) # [sf1_switch, sf2_switch, sf3_switch, gu1_switch, gu2_switch, gu3_switch] --- 0: disable 1: enable
#     glue_set_value_vec = np.zeros(6, dtype=np.float32) # [sf1_set_value, sf2_set_value, sf3_set_value, gu1_set_value, gu2_set_value, gu3_set_value]
#     glue_value_vec = np.zeros(6, dtype=np.float32) # [sf1_value, sf2_value, sf3_value, gu1_value, gu2_value, gu3_value]

#     normal = State("normal", initial=True)

# class GlueRunParStateMachineWrapper:
#     def __init__(self):
#         self.fsm = GlueRunParStateMachine()

#     def process_log_row(self, row):
#         pass


class KeyEventExtractor:
    def __init__(self):
        """
        material change event
        event maybe like: {
            part: aaaaa
            change: '(L,2500,B) -> (P,2600,3B)',
            time: '2026-03-02 15:39:32.530'
        }
        """

        # material change events
        self.material_events = []  # will be changed when I8, I12 is triggered
        # set func call events
        self.set_func_call_events = []  # will be changed when G12, G5 is triggered
        # splicer current state
        self.splicer_state = {
            "ls0": {
                "material": "-",
                "width": 0,
                "flute_type": "-",
                "next_batch": {
                    "material": "-",
                    "width": 0,
                    "flute_type": "-",
                },
                "change_time": "2026-03-02 15:39:32.530",
            },
            "ms1": {
                "material": "-",
                "width": 0,
                "flute_type": "-",
                "next_batch": {
                    "material": "-",
                    "width": 0,
                    "flute_type": "-",
                },
                "change_time": "2026-03-02 15:39:32.530",
            },
            "ls1": {
                "material": "-",
                "width": 0,
                "flute_type": "-",
                "next_batch": {
                    "material": "-",
                    "width": 0,
                    "flute_type": "-",
                },
                "change_time": "2026-03-02 15:39:32.530",
            },
            "ms2": {
                "material": "-",
                "width": 0,
                "flute_type": "-",
                "next_batch": {
                    "material": "-",
                    "width": 0,
                    "flute_type": "-",
                },
                "change_time": "2026-03-02 15:39:32.530",
            },
            "ls2": {
                "material": "-",
                "width": 0,
                "flute_type": "-",
                "next_batch": {
                    "material": "-",
                    "width": 0,
                    "flute_type": "-",
                },
                "change_time": "2026-03-02 15:39:32.530",
            },
        }
        self.df_state = {
            "material": "-.-.-.-.-",
            "width": 0,
            "flute_type": "-",
            "next_batch": {
                "material": "-.-.-.-.-",
                "width": 0,
                "flute_type": "-",
            },
            "change_time": "2026-03-02 15:39:32.530",
        }

        # glue part
        # gu set value
        self.gu_value_state = {}
        # sf value state
        self.sf_value_state = {}

    def process_log_row(self, row):
        """
        row: {
                "Message": message,
                "Date": date,  # 加入 Date
                "EventId": event_id,
                "MatchedTemplate": template,
                "ParsedValues": parsed_values
            }
        """
        # check log eventid
        if row["EventId"] == "I7":
            # get change paper ready event, and next material.
            # save the next batch based on log info
            # for sf material change ready
            parsed_values = row["ParsedValues"]
            handle_func_name = parsed_values["handle_func_name"]
            part = handle_func_to_splicer_part[handle_func_name]
            # assign next material batch
            self.splicer_state[part]["next_batch"] = {
                "material": parsed_values["next_material"],
                "width": int(parsed_values["width"]),
                "flute_type": parsed_values["flute_type"],
            }
        elif row["EventId"] == "I8":
            # Change paper is checked by the system.
            # use next batch to simulate the change material event
            # this function will change the splicer state part material and the material event
            # for sf material change check
            parsed_values = row["ParsedValues"]
            handle_func_name = parsed_values["handle_func_name"]
            part = handle_func_to_splicer_part[handle_func_name]
            # save material for event generate
            prev_material_batch = {
                "material": self.splicer_state[part]["material"],
                "width": self.splicer_state[part]["width"],
                "flute_type": self.splicer_state[part]["flute_type"],
            }
            current_material_batch = self.splicer_state[part]["next_batch"]
            # update the state
            self.splicer_state[part] = {
                "material": current_material_batch["material"],
                "width": current_material_batch["width"],
                "flute_type": current_material_batch["flute_type"],
                "next_batch": {
                    "material": "-",
                    "width": 0,
                    "flute_type": "-",
                },
                "change_time": str(row["Date"]),
            }
            # generate the event
            event = {
                "part": part,
                "msg": f"({prev_material_batch['material']},{prev_material_batch['width']},{prev_material_batch['flute_type']}) -> ({current_material_batch['material']},{current_material_batch['width']},{current_material_batch['flute_type']})",
                "time": str(row["Date"]),
            }
            self.material_events.append(event)
        elif row["EventId"] == "I11":
            # df is ready to change paper, get information
            # for df change material ready
            parsed_values = row["ParsedValues"]
            # print(parsed_values) # {'module': '换材判定模块', 'ip': '172.32.64.10', 'host': 'BTS-SHLY-SVR', 'username': 'null', 'prev_material': 'T.-.-.7.T', 'prev_flute_type': '3B', 'prev_width': '2400', 'material': 'P.-.-.8.J', 'flute_type': '3B', 'width': '2350', 'next_material': 'P.-.-.8.J'}
            self.df_state["next_batch"] = {
                "material": parsed_values["material"],
                "width": int(parsed_values["width"]),
                "flute_type": parsed_values["flute_type"],
            }
        elif row["EventId"] == "I12":
            # df is changed paper, generate event
            # for df change material check
            parsed_values = row["ParsedValues"]
            # print(parsed_values) # {'module': '换材判定模块', 'ip': '172.32.64.10', 'host': 'BTS-SHLY-SVR', 'username': 'null', 'handle_func_name': 'HandleGuChangePaper'}
            # save material for event generate
            prev_material_batch = {
                "material": self.df_state["material"],
                "width": self.df_state["width"],
                "flute_type": self.df_state["flute_type"],
            }
            current_material_batch = self.df_state["next_batch"]
            # update the state
            self.df_state = {
                "material": current_material_batch["material"],
                "width": current_material_batch["width"],
                "flute_type": current_material_batch["flute_type"],
                "next_batch": {
                    "material": "-",
                    "width": 0,
                    "flute_type": "-",
                },
                "change_time": str(row["Date"]),
            }
            # generate the event
            event = {
                "part": "df",
                "msg": f"({prev_material_batch['material']},{prev_material_batch['width']},{prev_material_batch['flute_type']}) -> ({current_material_batch['material']},{current_material_batch['width']},{current_material_batch['flute_type']})",
                "time": str(row["Date"]),
            }
            self.material_events.append(event)
        elif row["EventId"] == "G12":
            # setgluegu set gu value
            # based on self.gu_value_state, so G14 is triggered before G12
            parsed_values = row["ParsedValues"]
            # generate func event
            event = {
                "func": parsed_values["set_func_name"],
                "material": parsed_values["material"],
                "flute_type": parsed_values["flute_type"],
                "set_values": self.gu_value_state,
                "time": str(row["Date"]),
            }
            # clear gu value state
            self.gu_value_state = {}
            self.set_func_call_events.append(event)
        elif row["EventId"] == "G5":
            # setgluesf1/2 set gu value
            # based on self.sf_value_state, so G4 is triggered before G5
            parsed_values = row["ParsedValues"]
            glue_part = parsed_values["glue_part"]
            # generate func event
            event = {
                "func": parsed_values["set_func_name"],
                "part": parsed_values["glue_part"],
                "material": parsed_values["material"],
                "flute_type": parsed_values["flute_type"],
                "set_values": {
                    glue_part: self.sf_value_state[glue_part]
                },  # align the sf glue set function and gu glue set function
                "time": str(row["Date"]),
            }
            self.sf_value_state[glue_part] = {}
            self.set_func_call_events.append(event)
        elif row["EventId"] == "G4":  # SF calculate value
            parsed_values = row["ParsedValues"]
            glue_part = parsed_values["glue_part"]
            # 创建副本并删除指定字段
            filtered_data = parsed_values.copy()
            remove_fields = ["module", "ip", "host", "username", "glue_part"]

            for field in remove_fields:
                if field in filtered_data:
                    del filtered_data[field]

            # convert to simple format
            data = {
                "columns": [
                    "speed",
                    "min_glue",
                    "max_glue",
                    "min_weight",
                    "max_weight",
                    "current_glue_weight",
                    "speed_factor",
                    "min_speed",
                    "qdm_factor",
                    "ui_factor",
                    "offset",
                    "value",
                ],
                "data": [],
            }
            for i in range(1, 9, 1):
                temp = [
                    filtered_data[f"speed{i}"],
                    filtered_data[f"min_glue{i}"],
                    filtered_data[f"max_glue{i}"],
                    filtered_data[f"min_weight{i}"],
                    filtered_data[f"max_weight{i}"],
                    filtered_data[f"current_glue_weight{i}"],
                    filtered_data[f"speed_factor{i}"],
                    filtered_data[f"min_speed{i}"],
                    filtered_data[f"qdm_factor{i}"],
                    filtered_data[f"ui_factor{i}"],
                    filtered_data[f"offset{i}"],
                    filtered_data[f"value{i}"],
                ]
                data["data"].append(temp)

            # add to sf value state
            self.sf_value_state[glue_part] = data
        elif row["EventId"] == "G14":  # GU calculate value
            parsed_values = row["ParsedValues"]
            glue_part = parsed_values["glue_part"]
            # 创建副本并删除指定字段
            filtered_data = parsed_values.copy()
            remove_fields = ["module", "ip", "host", "username", "glue_part"]

            for field in remove_fields:
                if field in filtered_data:
                    del filtered_data[field]

            # convert to simple format
            data = {
                "columns": [
                    "speed",
                    "min_glue",
                    "max_glue",
                    "min_weight",
                    "max_weight",
                    "current_glue_weight",
                    "speed_factor",
                    "min_speed",
                    "qdm_factor",
                    "ui_factor",
                    "value",
                ],
                "data": [],
            }
            for i in range(1, 9, 1):
                temp = [
                    filtered_data[f"speed{i}"],
                    filtered_data[f"min_glue{i}"],
                    filtered_data[f"max_glue{i}"],
                    filtered_data[f"min_weight{i}"],
                    filtered_data[f"max_weight{i}"],
                    filtered_data[f"current_glue_weight{i}"],
                    filtered_data[f"speed_factor{i}"],
                    filtered_data[f"min_speed{i}"],
                    filtered_data[f"qdm_factor{i}"],
                    filtered_data[f"ui_factor{i}"],
                    filtered_data[f"value{i}"],
                ]
                data["data"].append(temp)

            # add gu values to data
            self.gu_value_state[glue_part] = data

    def track_machine_material_lifecycle(self, events, index):
        set_func_event = events[index]
        machine = set_func_event["part"]
        ms_ls = ("ms1", "ls1") if machine == "SF1" else ("ms2", "ls2")
        material_lifecycle = {
            ms_ls[0]: {"msg": "", "time": ""},
            ms_ls[1]: {"msg": "", "time": ""},
            "set_func": {"name": "", "time": ""},
        }
        # set func
        material_lifecycle["set_func"]["name"] = set_func_event["func"]
        material_lifecycle["set_func"]["time"] = set_func_event["time"]
        # prepare part assign
        part_count = material_part_count(set_func_event["material"])
        is_set_part = [0, 0, 1]
        part_2_idx = {
            ms_ls[0]: 0,
            ms_ls[1]: 1,
        }
        for i in range(index - 1, -1, -1):
            if (
                sum(is_set_part) >= part_count + 2
            ):  # df and set func is two, and we need another part to match the material
                return material_lifecycle
            event = events[i]
            if "func" in event:
                continue
            part = event["part"]
            if part not in ms_ls:
                continue
            material_lifecycle[part]["msg"] = event["msg"]
            material_lifecycle[part]["time"] = event["time"]
            is_set_part[part_2_idx[part]] = 1
        return material_lifecycle

    def track_material_lifecycle(self, events, index):
        material_lifecycle = {
            "ls0": {"msg": "", "time": ""},
            "ms1": {"msg": "", "time": ""},
            "ls1": {"msg": "", "time": ""},
            "ms2": {"msg": "", "time": ""},
            "ls2": {"msg": "", "time": ""},
            "df": {"msg": "", "time": ""},
            "set_func": {"name": "", "time": ""},
        }
        set_func_event = events[index]
        material_lifecycle["set_func"]["name"] = set_func_event["func"]
        material_lifecycle["set_func"]["time"] = set_func_event["time"]
        part_count = material_part_count(set_func_event["material"])
        is_set_part = [0, 0, 0, 0, 0, 0, 1]
        part_2_idx = {
            "ls0": 0,
            "ms1": 1,
            "ls1": 2,
            "ms2": 3,
            "ls2": 4,
            "df": 5,
        }
        for i in range(index - 1, -1, -1):
            if (
                sum(is_set_part) >= part_count + 2
            ):  # df and set func is two, and we need another part to match the material
                return material_lifecycle
            event = events[i]
            if "func" in event:
                continue
            part = event["part"]
            material_lifecycle[part]["msg"] = event["msg"]
            material_lifecycle[part]["time"] = event["time"]
            is_set_part[part_2_idx[part]] = 1
        return material_lifecycle

    def get_glue_set_function_full_event(self):
        # return all set funcs, and with lifecycle based on the machine material change event
        all_events = self.material_events + self.set_func_call_events
        all_events.sort(key=lambda x: x["time"])
        set_func_index = -1
        for i in range(len(all_events) - 1, -1, -1):
            if "material" in all_events[i]:
                # identify material type 8/J or P.-.-.8.J
                material = all_events[i]["material"]
                # get change material events
                lifecycle = (
                    self.track_machine_material_lifecycle(all_events, i)
                    if "/" in material
                    else self.track_material_lifecycle(all_events, i)
                )
                # print(f"{self.set_func_call_events[set_func_index]['material']} -> {lifecycle}")
                self.set_func_call_events[set_func_index]["lifecycle"] = lifecycle
                set_func_index -= 1
        return self.set_func_call_events

    def analysis(self, material):
        # todo: check material format
        # try to track the material lifecycle
        all_events = self.material_events + self.set_func_call_events
        all_events.sort(key=lambda x: x["time"])
        # find set func for material, check started from the latest row
        material_lifecycles = []
        for i in range(len(all_events) - 1, -1, -1):
            if "material" in all_events[i] and all_events[i]["material"] == material:
                # material maybe not appear only once
                lifecycle = self.track_material_lifecycle(all_events, i)
                material_lifecycles.append(lifecycle)
        return material_lifecycles


# L.-.-.8.L -> ls0, ms1, ls1, ms2, ls2

# DF 换材
# ""DFChangePaper--已进入换材处理:糊机同材剩余-横切到糊机的距离=32.975,糊机到横切距离设定值=34"",""ExceptionInfo"":null}",
# ""DFChangePaper--糊机判定为换材\r\n上次使用的：材质=A.-.-.0.A，楞型=3B，门幅=2250\r\n即将使用的：材质=A.-.-.0.G，楞型=3B，门幅=2250;下批材质=A.-.-.0.G\r\n准备进入HandleGuChangePaper具体执行函数\r\n"",""ExceptionInfo"":null}",
# ""HandleGuChangePaper 已发送糊机换材消息，通知各执行类"",""ExceptionInfo"":null}",

# 接纸机换材
# ""LS0 判定为换卷，准备进入 HandleChangeRollLS0 函数"",""ExceptionInfo"":null}",
# ""进入到换卷处理函数 HandleChangeRollLS0\r\n"",""ExceptionInfo"":null}",
# ""HandleChangeRollLS0--已经进入换材准备中状态了，用下批理论材质进行赋值操作\r\n当前：材质=A，门幅=2300，楞型=3B\r\n下批理论：材质=A，门幅=2250，楞型=3B，下批订单全材质=A.-.-.0.A\r\n"",""ExceptionInfo"":null}",
# ""HandleChangePaperLS0 准备发送LS0换材消息(正常换卷换材)，通知各执行类"",""ExceptionInfo"":null}",
# ""拿到了 LS0 实际材质=A,实际门幅=2250.00\r\n当前 LS0 材质=A,门幅=2250\r\n材质一样，不需要赋值\r\n"",""ExceptionInfo"":null}",

# 胶水赋值
# "进入 SetGlueGu 准备点位赋值，材质=D.-.-.8.A,楞型=3B"
# "GU2糊间隙计算结果：车速值=30;最小糊间隙=10;最大糊间隙=35;最小克重=200;最大克重=500;当前胶水克重=330;车速系数=1.80;车速限制的最小值=30;QDM系数=1.00;界面系数=1.15;计算结果：43.12\r\nGU2糊间隙计算结果：车速值=60;最小糊间隙=10;最大糊间隙=35;最小克重=200;最大克重=500;当前胶水克重=330;车速系数=1.60;车速限制的最小值=25;QDM系数=1.00;界面系数=1.15;计算结果：38.33\r\nGU2糊间隙计算结果：车速值=90;最小糊间隙=10;最大糊间隙=35;最小克重=200;最大克重=500;当前胶水克重=330;车速系数=1.40;车速限制的最小值=20;QDM系数=1.00;界面系数=1.15;计算结果：33.54\r\nGU2糊间隙计算结果：车速值=120;最小糊间隙=10;最大糊间隙=35;最小克重=200;最大克重=500;当前胶水克重=330;车速系数=1.20;车速限制的最小值=17;QDM系数=1.00;界面系数=1.15;计算结果：28.75\r\nGU2糊间隙计算结果：车速值=140;最小糊间隙=10;最大糊间隙=35;最小克重=200;最大克重=500;当前胶水克重=330;车速系数=1.10;车速限制的最小值=15;QDM系数=1.00;界面系数=1.15;计算结果：26.35\r\nGU2糊间隙计算结果：车速值=200;最小糊间隙=10;最大糊间隙=35;最小克重=200;最大克重=500;当前胶水克重=330;车速系数=1.00;车速限制的最小值=10;QDM系数=1.00;界面系数=1.15;计算结果：23.96\r\nGU2糊间隙计算结果：车速值=240;最小糊间隙=10;最大糊间隙=35;最小克重=200;最大克重=500;当前胶水克重=330;车速系数=0.90;车速限制的最小值=10;QDM系数=1.00;界面系数=1.15;计算结果：21.56\r\nGU2糊间隙计算结果：车速值=260;最小糊间隙=10;最大糊间隙=35;最小克重=200;最大克重=500;当前胶水克重=330;车速系数=0.85;车速限制的最小值=10;QDM系数=1.00;界面系数=1.15;计算结果：20.36\r\n"
# "SetGlueGu--糊机糊间隙设备写值完成,材质=A.-.-.0.A,楞型=3B"
# 存在其他点位的胶水信息赋值，比如GU，总体需要在SetGlue*方法写入的log处查看材质是否错位

# 横切换材
