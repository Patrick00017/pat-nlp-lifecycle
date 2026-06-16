import pandas as pd
from datetime import datetime
import numpy as np
from constant import handle_func_to_splicer_part
from utils import material_part_count, is_material_equal

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
        self.material_events = [] # will be changed when I8, I12 is triggered
        
        # splicer current state
        self.splicer_state = {
            'ls0': {
                'material': '-',
                'width': 0,
                'flute_type': '-',
                'next_batch': {
                    'material': '-',
                    'width': 0,
                    'flute_type': '-',
                },
                'change_time': '2026-03-02 15:39:32.530'
            },
            'ms1': {
                'material': '-',
                'width': 0,
                'flute_type': '-',
                'next_batch': {
                    'material': '-',
                    'width': 0,
                    'flute_type': '-',
                },
                'change_time': '2026-03-02 15:39:32.530'
            },
            'ls1': {
                'material': '-',
                'width': 0,
                'flute_type': '-',
                'next_batch': {
                    'material': '-',
                    'width': 0,
                    'flute_type': '-',
                },
                'change_time': '2026-03-02 15:39:32.530'
            },
            'ms2': {
                'material': '-',
                'width': 0,
                'flute_type': '-',
                'next_batch': {
                    'material': '-',
                    'width': 0,
                    'flute_type': '-',
                },
                'change_time': '2026-03-02 15:39:32.530'
            },
            'ls2': {
                'material': '-',
                'width': 0,
                'flute_type': '-',
                'next_batch': {
                    'material': '-',
                    'width': 0,
                    'flute_type': '-',
                },
                'change_time': '2026-03-02 15:39:32.530'
            }
        }
        self.df_state = {
            'material': '-.-.-.-.-',
            'width': 0,
            'flute_type': '-',
            'next_batch': {
                'material': '-.-.-.-.-',
                'width': 0,
                'flute_type': '-',
            },
            'change_time': '2026-03-02 15:39:32.530'
        }
        

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
                'material': parsed_values["next_material"],
                'width': int(parsed_values["width"]),
                'flute_type': parsed_values["flute_type"]
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
                'material': self.splicer_state[part]["material"],
                'width': self.splicer_state[part]["width"],
                'flute_type': self.splicer_state[part]["flute_type"]
            }
            current_material_batch = self.splicer_state[part]["next_batch"]
            # update the state
            self.splicer_state[part] = {
                'material': current_material_batch['material'],
                'width': current_material_batch['width'],
                'flute_type': current_material_batch['flute_type'],
                'next_batch': {
                    'material': '-',
                    'width': 0,
                    'flute_type': '-',
                },
                'change_time': str(row['Date'])
            }
            # generate the event
            event = {
                'part': part,
                'msg': f"({prev_material_batch['material']},{prev_material_batch['width']},{prev_material_batch['flute_type']}) -> ({current_material_batch['material']},{current_material_batch['width']},{current_material_batch['flute_type']})",
                'time': str(row['Date'])
            }
            self.material_events.append(event)
        elif row['EventId'] == 'I11':
            # df is ready to change paper, get information
            # for df change material ready
            parsed_values = row["ParsedValues"]
            # print(parsed_values) # {'module': '换材判定模块', 'ip': '172.32.64.10', 'host': 'BTS-SHLY-SVR', 'username': 'null', 'prev_material': 'T.-.-.7.T', 'prev_flute_type': '3B', 'prev_width': '2400', 'material': 'P.-.-.8.J', 'flute_type': '3B', 'width': '2350', 'next_material': 'P.-.-.8.J'}
            self.df_state['next_batch'] = {
                'material': parsed_values['material'],
                'width': int(parsed_values['width']),
                'flute_type': parsed_values['flute_type'],
            }
        elif row['EventId'] == 'I12':
            # df is changed paper, generate event
            # for df change material check
            parsed_values = row["ParsedValues"]
            # print(parsed_values) # {'module': '换材判定模块', 'ip': '172.32.64.10', 'host': 'BTS-SHLY-SVR', 'username': 'null', 'handle_func_name': 'HandleGuChangePaper'}
            # save material for event generate
            prev_material_batch = {
                'material': self.df_state["material"],
                'width': self.df_state["width"],
                'flute_type': self.df_state["flute_type"]
            }
            current_material_batch = self.df_state["next_batch"]
            # update the state
            self.df_state = {
                'material': current_material_batch['material'],
                'width': current_material_batch['width'],
                'flute_type': current_material_batch['flute_type'],
                'next_batch': {
                    'material': '-',
                    'width': 0,
                    'flute_type': '-',
                },
                'change_time': str(row['Date'])
            }
            # generate the event
            event = {
                'part': 'df',
                'msg': f"({prev_material_batch['material']},{prev_material_batch['width']},{prev_material_batch['flute_type']}) -> ({current_material_batch['material']},{current_material_batch['width']},{current_material_batch['flute_type']})",
                'time': str(row['Date'])
            }
            self.material_events.append(event)
    
    
                

class GlueEventExtractor(KeyEventExtractor):
    def __init__(self):
        # 调用父类初始化方法
        super().__init__()
        # glue part
        # set func call events
        self.set_func_call_events = [] # will be changed when G12, G5 is triggered
        # gu set value
        self.gu_value_state = {}
        # sf value state
        self.sf_value_state = {}
        # raw parsed rows for diagnostic analysis
        self.raw_parsed_rows = []

    def process(self, row):
        # print(row['EventId'])
        self.process_log_row(row) # for material change event
        # store raw parsed row for diagnostic analysis
        if pd.notna(row.get('EventId')):
            self.raw_parsed_rows.append(row.to_dict())
        # for glue event
        if row['EventId'] == 'G12':
            # setgluegu set gu value
            # based on self.gu_value_state, so G14 is triggered before G12
            parsed_values = row["ParsedValues"]
            # generate func event
            event = {
                'func': parsed_values['set_func_name'],
                'part': 'DF',
                'material': parsed_values['material'],
                'flute_type': parsed_values['flute_type'],
                'set_values': self.gu_value_state, 
                'time': str(row['Date'])
            }
            # clear gu value state
            self.gu_value_state = {}
            self.set_func_call_events.append(event)
        elif row['EventId'] == 'G5':
            # setgluesf1/2 set gu value
            # based on self.sf_value_state, so G4 is triggered before G5
            parsed_values = row["ParsedValues"]
            glue_part = parsed_values['glue_part']
            # generate func event
            event = {
                'func': parsed_values['set_func_name'],
                'part': parsed_values['glue_part'],
                'material': parsed_values['material'],
                'flute_type': parsed_values['flute_type'],
                'set_values': {glue_part : self.sf_value_state[glue_part]}, # align the sf glue set function and gu glue set function
                'time': str(row['Date'])
            }
            self.sf_value_state[glue_part] = {}
            self.set_func_call_events.append(event)
        elif row['EventId'] == 'G4': # SF calculate value
            parsed_values = row["ParsedValues"]
            glue_part = parsed_values['glue_part']
            # 创建副本并删除指定字段
            filtered_data = parsed_values.copy()
            remove_fields = ['module', 'ip', 'host', 'username', 'glue_part']

            for field in remove_fields:
                if field in filtered_data:
                    del filtered_data[field]

            # convert to simple format
            data = {
                'columns': ['speed', 'min_glue', 'max_glue', 'min_weight', 'max_weight', 'current_glue_weight', 'speed_factor', 'min_speed', 'qdm_factor', 'ui_factor', 'offset', 'value'],
                'data': []
            }
            for i in range(1, 9, 1):
                temp = [filtered_data[f'speed{i}'], filtered_data[f'min_glue{i}'], filtered_data[f'max_glue{i}'], filtered_data[f'min_weight{i}'], filtered_data[f'max_weight{i}'], filtered_data[f'current_glue_weight{i}'], filtered_data[f'speed_factor{i}'], filtered_data[f'min_speed{i}'], filtered_data[f'qdm_factor{i}'], filtered_data[f'ui_factor{i}'], filtered_data[f'offset{i}'], filtered_data[f'value{i}']]
                data['data'].append(temp)

            # add to sf value state
            self.sf_value_state[glue_part] = data
        elif row['EventId'] == 'G14': # GU calculate value
            parsed_values = row["ParsedValues"]
            glue_part = parsed_values['glue_part']
            # 创建副本并删除指定字段
            filtered_data = parsed_values.copy()
            remove_fields = ['module', 'ip', 'host', 'username', 'glue_part']

            for field in remove_fields:
                if field in filtered_data:
                    del filtered_data[field]

            # convert to simple format
            data = {
                'columns': ['speed', 'min_glue', 'max_glue', 'min_weight', 'max_weight', 'current_glue_weight', 'speed_factor', 'min_speed', 'qdm_factor', 'ui_factor', 'value'],
                'data': []
            }
            for i in range(1, 9, 1):
                temp = [filtered_data[f'speed{i}'], filtered_data[f'min_glue{i}'], filtered_data[f'max_glue{i}'], filtered_data[f'min_weight{i}'], filtered_data[f'max_weight{i}'], filtered_data[f'current_glue_weight{i}'], filtered_data[f'speed_factor{i}'], filtered_data[f'min_speed{i}'], filtered_data[f'qdm_factor{i}'], filtered_data[f'ui_factor{i}'], filtered_data[f'value{i}']]
                data['data'].append(temp)

            # add gu values to data
            self.gu_value_state[glue_part] = data
        elif row['EventId'] == 'G15':
            # GU pre-write cancellation: values were calculated but never written to device
            self.gu_value_state = {}

    def track_machine_material_lifecycle(self, events, index):
        set_func_event = events[index]
        machine = set_func_event['part']
        ms_ls = ('ms1', 'ls1') if machine == 'SF1' else ('ms2', 'ls2')
        material_lifecycle = {
            ms_ls[0]: {
                'msg': '',
                'time': ''
            },
            ms_ls[1]: {
                'msg': '',
                'time': ''
            },
            'set_func': {
                'name': '',
                'time': ''
            }
        }
        # set func
        material_lifecycle['set_func']['name'] = set_func_event['func']
        material_lifecycle['set_func']['time'] = set_func_event['time']
        # prepare part assign
        part_count = material_part_count(set_func_event['material'])
        is_set_part = [0, 0, 1]
        part_2_idx = {
            ms_ls[0]: 0,
            ms_ls[1]: 1,
        }
        for i in range(index-1, -1, -1):
            if sum(is_set_part) >= part_count + 2: # df and set func is two, and we need another part to match the material
                return material_lifecycle
            event = events[i]
            if 'func' in event:
                continue
            part = event['part']
            if part not in ms_ls:
                continue
            material_lifecycle[part]['msg'] = event['msg']
            material_lifecycle[part]['time'] = event['time']
            is_set_part[part_2_idx[part]] = 1
        return material_lifecycle

    def track_material_lifecycle(self, events, index):
        material_lifecycle = {
            'ls0': {
                'msg': '',
                'time': ''
            },
            'ms1': {
                'msg': '',
                'time': ''
            },
            'ls1': {
                'msg': '',
                'time': ''
            },
            'ms2': {
                'msg': '',
                'time': ''
            },
            'ls2': {
                'msg': '',
                'time': ''
            },
            'df': {
                'msg': '',
                'time': ''
            },
            'set_func': {
                'name': '',
                'time': ''
            }
        }
        set_func_event = events[index]
        material_lifecycle['set_func']['name'] = set_func_event['func']
        material_lifecycle['set_func']['time'] = set_func_event['time']
        part_count = material_part_count(set_func_event['material'])
        is_set_part = [0, 0, 0, 0, 0, 0, 1]
        part_2_idx = {
            'ls0': 0,
            'ms1': 1,
            'ls1': 2,
            'ms2': 3,
            'ls2': 4,
            'df': 5,
        }
        for i in range(index-1, -1, -1):
            if sum(is_set_part) >= part_count + 2: # df and set func is two, and we need another part to match the material
                return material_lifecycle
            event = events[i]
            if 'func' in event:
                continue
            part = event['part']
            material_lifecycle[part]['msg'] = event['msg']
            material_lifecycle[part]['time'] = event['time']
            is_set_part[part_2_idx[part]] = 1
        return material_lifecycle

    def get_glue_set_function_full_event(self):
        # return all set funcs, and with lifecycle based on the machine material change event
        print(f"material len: {len(self.material_events)}, setfunc len: {len(self.set_func_call_events)}")
        all_events = self.material_events + self.set_func_call_events
        all_events.sort(key=lambda x: x['time'])
        set_func_index = -1
        for i in range(len(all_events)-1, -1, -1):
            if 'material' in all_events[i]:
                # identify material type 8/J or P.-.-.8.J
                material = all_events[i]['material']
                # get change material events
                lifecycle = self.track_machine_material_lifecycle(all_events, i) if '/' in material else self.track_material_lifecycle(all_events, i)
                # print(f"{self.set_func_call_events[set_func_index]['material']} -> {lifecycle}")
                self.set_func_call_events[set_func_index]['lifecycle'] = lifecycle
                set_func_index -= 1
        return self.set_func_call_events

    def analysis(self, material):
        # todo: check material format
        # try to track the material lifecycle
        all_events = self.material_events + self.set_func_call_events
        all_events.sort(key=lambda x: x['time'])
        # find set func for material, check started from the latest row
        material_lifecycles = []
        for i in range(len(all_events)-1, -1, -1):
            if 'material' in all_events[i] and all_events[i]['material'] == material:
                # material maybe not appear only once 
                lifecycle = self.track_material_lifecycle(all_events, i)
                material_lifecycles.append(lifecycle)
        return material_lifecycles
    
    def convert_glue_func_to_markdown(self, data, desire_material=None):
        print(data)
        """
        将SetGlueSF*数据转换为Markdown格式
        """
        markdown = []
        
        # 1. 标题和基本信息
        markdown.append("# 📊 函数调用记录\n")
        
        # 基本信息表格
        markdown.append("## 📋 基本信息\n")
        markdown.append("| 字段 | 值 |")
        markdown.append("|------|-----|")
        markdown.append(f"| **函数** | `{data.get('func', 'N/A')}` |")
        markdown.append(f"| **部件** | `{data.get('part', 'DF')}` |")
        markdown.append(f"| **材料** | `{data.get('material', 'N/A')}` |")
        markdown.append(f"| **瓦楞类型** | `{data.get('flute_type', 'N/A')}` |")
        markdown.append(f"| **时间** | `{data.get('time', 'N/A')}` |\n")
        
        # 2. 生命周期信息
        if 'lifecycle' in data:
            markdown.append("## 🔄 生命周期\n")
            markdown.append("| 阶段 | 信息 | 时间 |")
            markdown.append("|------|------|------|")
            
            lifecycle = data['lifecycle']
            if 'ls0' in lifecycle:
                markdown.append(f"| **LS0** | `{lifecycle['ls0'].get('msg', 'N/A')}` | {lifecycle['ls0'].get('time', 'N/A')} |")
            if 'ms1' in lifecycle:
                markdown.append(f"| **MS1** | `{lifecycle['ms1'].get('msg', 'N/A')}` | {lifecycle['ms1'].get('time', 'N/A')} |")
            if 'ls1' in lifecycle:
                markdown.append(f"| **LS1** | `{lifecycle['ls1'].get('msg', 'N/A')}` | {lifecycle['ls1'].get('time', 'N/A')} |")
            if 'ms2' in lifecycle:
                markdown.append(f"| **MS2** | `{lifecycle['ms2'].get('msg', 'N/A')}` | {lifecycle['ms2'].get('time', 'N/A')} |")
            if 'ls2' in lifecycle:
                markdown.append(f"| **LS2** | `{lifecycle['ls2'].get('msg', 'N/A')}` | {lifecycle['ls2'].get('time', 'N/A')} |")
            if 'df' in lifecycle:
                markdown.append(f"| **DF** | `{lifecycle['df'].get('msg', 'N/A')}` | {lifecycle['df'].get('time', 'N/A')} |")
            if 'set_func' in lifecycle:
                markdown.append(f"| **Set Function** | `{lifecycle['set_func'].get('name', 'N/A')}` | {lifecycle['set_func'].get('time', 'N/A')} |")
            markdown.append("")
        
        # 3. 设置值表格
        if 'set_values' in data:
            markdown.append("## ⚙️ 设置值\n")
            
            set_values = data['set_values']
            for key, value in set_values.items():
                markdown.append(f"### 部位: {key}")
                columns = value.get('columns', [])
                data_rows = value.get('data', [])
                
                if columns and data_rows:
                    # 创建表格头
                    header = "| " + " | ".join([col.replace('_', ' ').title() for col in columns]) + " |"
                    separator = "|" + "|".join(["---" for _ in columns]) + "|"
                    
                    markdown.append(header)
                    markdown.append(separator)
                    
                    # 添加数据行
                    for row in data_rows:
                        markdown.append("| " + " | ".join(row) + " |")
                    
                    markdown.append("")
        
        # # 4. 数据统计
        # if 'set_values' in data and 'data' in data['set_values']:
        #     markdown.append("## 📈 数据统计\n")
            
        #     data_rows = data['set_values']['data']
        #     speeds = [int(row[0]) for row in data_rows]
        #     values = [str(row[-1]) for row in data_rows]
            
        #     markdown.append("| 统计项 | 数值 |")
        #     markdown.append("|--------|------|")
        #     markdown.append(f"| **速度范围** | `{min(speeds)} - {max(speeds)}` |")
        #     markdown.append(f"| **数值范围** | `{min(values):.2f} - {max(values):.2f}` |")
        #     markdown.append(f"| **数据点数** | `{len(data_rows)}` |\n")
        
        # 5. 完整数据（可折叠）
        # markdown.append("## 📦 完整数据\n")
        # markdown.append("<details>")
        # markdown.append("<summary><b>点击查看完整JSON</b></summary>\n")
        # markdown.append("```json")
        # markdown.append(json.dumps(data, indent=2, ensure_ascii=False))
        # markdown.append("```")
        # markdown.append("</details>")

        if desire_material: # for example P.-.-.8.J
            markdown.append("## 材质匹配情况 \n")
            # get the wrong material event
            material = data.get('material', 'N/A')
            part = data.get('part', 'N/A')
            if material == 'N/A' or part == 'N/A':
                return "\n".join(markdown)
            # check material is 5 parts or 2 parts
            if is_material_equal(desire_material, material, part=part):
                markdown.append(f"{desire_material} <---> {material}: 材质匹配成功 \n")
            else:
                markdown.append(f"{desire_material} <---> {material}: 材质匹配失败 \n")

        markdown.append("\n --- \n")
        return "\n".join(markdown)
    
class VacuumBlowerEventExtractor(KeyEventExtractor):
    def __init__(self):
        # 调用父类初始化方法
        super().__init__()
        # glue part
        # set func call events
        self.set_func_call_events = [] # will be changed when vb5 is triggered

    def process(self, row):
        # print(row['EventId'])
        self.process_log_row(row) # for material change event
        # for glue event
        if row['EventId'] == 'vb5':
            # setgluegu set gu value
            # based on self.gu_value_state, so G14 is triggered before G12
            parsed_values = row["ParsedValues"]
            self.set_func_call_events.append({
                'func': parsed_values['set_func_name'],
                'part': parsed_values['part'], 
                'base_value_multiply_width_factor_result': parsed_values['base_value_multiply_width_factor_result'], 
                'ms_material': parsed_values['ms_material'], 
                'ls_material': parsed_values['ls_material'], 
                'width': parsed_values['width'], 
                'flute_type': parsed_values['flute_type'],
                'time': str(row['Date'])
            })

    def get_vacuum_blower_set_function_full_event(self):
        # return all set funcs, and with lifecycle based on the machine material change event
        print(f"material len: {len(self.material_events)}, setfunc len: {len(self.set_func_call_events)}")
        print(self.material_events)
        all_events = self.material_events + self.set_func_call_events
        all_events.sort(key=lambda x: x['time'])
        set_func_index = -1
        for i in range(len(all_events)-1, -1, -1):
            if 'ms_material' in all_events[i]: # select vacuum blower set event
                # ms_material and ls_material is single material like "8", "P"
                # get change material events
                lifecycle = self.track_machine_material_lifecycle(all_events, i)
                # print(f"{self.set_func_call_events[set_func_index]['material']} -> {lifecycle}")
                self.set_func_call_events[set_func_index]['lifecycle'] = lifecycle
                # print(self.set_func_call_events[set_func_index]['lifecycle'])
                set_func_index -= 1
        # return self.set_func_call_events

        markdown_results = ''
        for result in self.set_func_call_events:
            md = self.convert_vacuum_blower_func_to_markdown(result)
            markdown_results += md
            # markdown_results += "\n---\n"

        return self.set_func_call_events, markdown_results
    
    def track_machine_material_lifecycle(self, events, idx):
        set_func_event = events[idx]
        machine = set_func_event['part']
        ms_ls = ('ms1', 'ls1') if machine == 'SF1' else ('ms2', 'ls2')
        material_lifecycle = {
            ms_ls[0]: {
                'msg': '',
                'time': ''
            },
            ms_ls[1]: {
                'msg': '',
                'time': ''
            },
            'set_func': {
                'name': '',
                'time': ''
            }
        }
        # set func
        material_lifecycle['set_func']['name'] = set_func_event['func']
        material_lifecycle['set_func']['time'] = set_func_event['time']
        # prepare part assign
        is_set_part = [0, 0, 1]
        part_2_idx = {
            ms_ls[0]: 0,
            ms_ls[1]: 1,
        }
        for i in range(idx-1, -1, -1):
            if sum(is_set_part) >= 3: # df and set func is two, and we need another part to match the material
                return material_lifecycle
            event = events[i]
            if 'func' in event:
                continue
            part = event['part']
            if part not in ms_ls:
                continue
            material_lifecycle[part]['msg'] = event['msg']
            material_lifecycle[part]['time'] = event['time']
            is_set_part[part_2_idx[part]] = 1
        return material_lifecycle

    def convert_vacuum_blower_func_to_markdown(self, result):
        """
        将数据转换为Markdown格式
        """
        markdown = []
        
        # 1. 标题和基本信息
        markdown.append("# 📊 函数调用记录\n")
        
        # 基本信息表格
        markdown.append("## 📋 基本信息\n")
        markdown.append("| 字段 | 值 |")
        markdown.append("|------|-----|")
        markdown.append(f"| **函数** | `{result.get('func', 'N/A')}` |")
        markdown.append(f"| **部件** | `{result.get('part', 'N/A')}` |")
        markdown.append(f"| **基础值 x 门幅系数** | `{result.get('part', 'N/A')}` |")
        markdown.append(f"| **芯纸材质** | `{result.get('ms_material', 'N/A')}` |")
        markdown.append(f"| **里纸材质** | `{result.get('ls_material', 'N/A')}` |")
        markdown.append(f"| **瓦楞类型** | `{result.get('flute_type', 'N/A')}` |")
        markdown.append(f"| **时间** | `{result.get('time', 'N/A')}` |\n")
        
        # 2. 生命周期信息
        if 'lifecycle' in result:
            markdown.append("## 🔄 生命周期\n")
            markdown.append("| 阶段 | 信息 | 时间 |")
            markdown.append("|------|------|------|")
            
            lifecycle = result['lifecycle']
            if 'ls0' in lifecycle:
                markdown.append(f"| **LS0** | `{lifecycle['ls0'].get('msg', 'N/A')}` | {lifecycle['ls0'].get('time', 'N/A')} |")
            if 'ms1' in lifecycle:
                markdown.append(f"| **MS1** | `{lifecycle['ms1'].get('msg', 'N/A')}` | {lifecycle['ms1'].get('time', 'N/A')} |")
            if 'ls1' in lifecycle:
                markdown.append(f"| **LS1** | `{lifecycle['ls1'].get('msg', 'N/A')}` | {lifecycle['ls1'].get('time', 'N/A')} |")
            if 'ms2' in lifecycle:
                markdown.append(f"| **MS2** | `{lifecycle['ms2'].get('msg', 'N/A')}` | {lifecycle['ms2'].get('time', 'N/A')} |")
            if 'ls2' in lifecycle:
                markdown.append(f"| **LS2** | `{lifecycle['ls2'].get('msg', 'N/A')}` | {lifecycle['ls2'].get('time', 'N/A')} |")
            if 'df' in lifecycle:
                markdown.append(f"| **DF** | `{lifecycle['df'].get('msg', 'N/A')}` | {lifecycle['df'].get('time', 'N/A')} |")
            if 'set_func' in lifecycle:
                markdown.append(f"| **Set Function** | `{lifecycle['set_func'].get('name', 'N/A')}` | {lifecycle['set_func'].get('time', 'N/A')} |")
            markdown.append("")
        
        # 3. 设置值表格
        if 'set_values' in result:
            markdown.append("## ⚙️ 设置值\n")
            
            set_values = result['set_values']
            for key, value in set_values.items():
                markdown.append(f"### 部位: {key}")
                columns = value.get('columns', [])
                data_rows = value.get('data', [])
                
                if columns and data_rows:
                    # 创建表格头
                    header = "| " + " | ".join([col.replace('_', ' ').title() for col in columns]) + " |"
                    separator = "|" + "|".join(["---" for _ in columns]) + "|"
                    
                    markdown.append(header)
                    markdown.append(separator)
                    
                    # 添加数据行
                    for row in data_rows:
                        markdown.append("| " + " | ".join(row) + " |")
                    
                    markdown.append("")
        
        # # 4. 数据统计
        # if 'set_values' in data and 'data' in data['set_values']:
        #     markdown.append("## 📈 数据统计\n")
            
        #     data_rows = data['set_values']['data']
        #     speeds = [int(row[0]) for row in data_rows]
        #     values = [str(row[-1]) for row in data_rows]
            
        #     markdown.append("| 统计项 | 数值 |")
        #     markdown.append("|--------|------|")
        #     markdown.append(f"| **速度范围** | `{min(speeds)} - {max(speeds)}` |")
        #     markdown.append(f"| **数值范围** | `{min(values):.2f} - {max(values):.2f}` |")
        #     markdown.append(f"| **数据点数** | `{len(data_rows)}` |\n")
        
        # 5. 完整数据（可折叠）
        # markdown.append("## 📦 完整数据\n")
        # markdown.append("<details>")
        # markdown.append("<summary><b>点击查看完整JSON</b></summary>\n")
        # markdown.append("```json")
        # markdown.append(json.dumps(data, indent=2, ensure_ascii=False))
        # markdown.append("```")
        # markdown.append("</details>")
        markdown.append("\n\n --- \n")

        return "\n".join(markdown)
        
class SPTensionEventExtractor(KeyEventExtractor):
    def __init__(self):
        # 调用父类初始化方法
        super().__init__()
        # glue part
        # set func call events
        self.set_func_call_events = [] # will be changed when sp3 is triggered

    def process(self, row):
        # print(row['EventId'])
        self.process_log_row(row) # for material change event
        # for glue event
        if row['EventId'] == 'spt3':
            # setgluegu set gu value
            # based on self.gu_value_state, so G14 is triggered before G12
            parsed_values = row["ParsedValues"]
            self.set_func_call_events.append({
                'func': parsed_values['set_func_name'],
                # 'part': parsed_values['part'], 
                'material': parsed_values['material'], # 8/D 
                'width': parsed_values['width'], 
                # 'flute_type': parsed_values['flute_type'],
                'time': str(row['Date'])
            })

    def get_sptension_set_function_full_event(self):
        pass
        # return all set funcs, and with lifecycle based on the machine material change event
        # print(f"material len: {len(self.material_events)}, setfunc len: {len(self.set_func_call_events)}")
        # print(self.material_events)
        # all_events = self.material_events + self.set_func_call_events
        # all_events.sort(key=lambda x: x['time'])
        # set_func_index = -1
        # for i in range(len(all_events)-1, -1, -1):
        #     if 'ms_material' in all_events[i]: # select vacuum blower set event
        #         # ms_material and ls_material is single material like "8", "P"
        #         # get change material events
        #         lifecycle = self.track_machine_material_lifecycle(all_events, i)
        #         # print(f"{self.set_func_call_events[set_func_index]['material']} -> {lifecycle}")
        #         self.set_func_call_events[set_func_index]['lifecycle'] = lifecycle
        #         # print(self.set_func_call_events[set_func_index]['lifecycle'])
        #         set_func_index -= 1
        # return self.set_func_call_events

        # markdown_results = ''
        # for result in self.set_func_call_events:
        #     md = self.convert_vacuum_blower_func_to_markdown(result)
        #     markdown_results += md
        #     # markdown_results += "\n---\n"

        # return self.set_func_call_events, markdown_results
    
    def track_machine_material_lifecycle(self, events, idx):
        pass

        # set_func_event = events[idx]
        # machine = set_func_event['part']
        # ms_ls = ('ms1', 'ls1') if machine == 'SF1' else ('ms2', 'ls2')
        # material_lifecycle = {
        #     ms_ls[0]: {
        #         'msg': '',
        #         'time': ''
        #     },
        #     ms_ls[1]: {
        #         'msg': '',
        #         'time': ''
        #     },
        #     'set_func': {
        #         'name': '',
        #         'time': ''
        #     }
        # }
        # # set func
        # material_lifecycle['set_func']['name'] = set_func_event['func']
        # material_lifecycle['set_func']['time'] = set_func_event['time']
        # # prepare part assign
        # is_set_part = [0, 0, 1]
        # part_2_idx = {
        #     ms_ls[0]: 0,
        #     ms_ls[1]: 1,
        # }
        # for i in range(idx-1, -1, -1):
        #     if sum(is_set_part) >= 3: # df and set func is two, and we need another part to match the material
        #         return material_lifecycle
        #     event = events[i]
        #     if 'func' in event:
        #         continue
        #     part = event['part']
        #     if part not in ms_ls:
        #         continue
        #     material_lifecycle[part]['msg'] = event['msg']
        #     material_lifecycle[part]['time'] = event['time']
        #     is_set_part[part_2_idx[part]] = 1
        # return material_lifecycle

    def convert_sptension_func_to_markdown(self, result):
        """
        将数据转换为Markdown格式
        """
        markdown = []
        
        # 1. 标题和基本信息
        markdown.append("# 📊 函数调用记录\n")
        
        # 基本信息表格
        markdown.append("## 📋 基本信息\n")
        markdown.append("| 字段 | 值 |")
        markdown.append("|------|-----|")
        markdown.append(f"| **函数** | `{result.get('func', 'N/A')}` |")
        markdown.append(f"| **部件** | `{result.get('part', 'N/A')}` |")
        markdown.append(f"| **基础值 x 门幅系数** | `{result.get('part', 'N/A')}` |")
        markdown.append(f"| **芯纸材质** | `{result.get('ms_material', 'N/A')}` |")
        markdown.append(f"| **里纸材质** | `{result.get('ls_material', 'N/A')}` |")
        markdown.append(f"| **瓦楞类型** | `{result.get('flute_type', 'N/A')}` |")
        markdown.append(f"| **时间** | `{result.get('time', 'N/A')}` |\n")
        
        # 2. 生命周期信息
        if 'lifecycle' in result:
            markdown.append("## 🔄 生命周期\n")
            markdown.append("| 阶段 | 信息 | 时间 |")
            markdown.append("|------|------|------|")
            
            lifecycle = result['lifecycle']
            if 'ls0' in lifecycle:
                markdown.append(f"| **LS0** | `{lifecycle['ls0'].get('msg', 'N/A')}` | {lifecycle['ls0'].get('time', 'N/A')} |")
            if 'ms1' in lifecycle:
                markdown.append(f"| **MS1** | `{lifecycle['ms1'].get('msg', 'N/A')}` | {lifecycle['ms1'].get('time', 'N/A')} |")
            if 'ls1' in lifecycle:
                markdown.append(f"| **LS1** | `{lifecycle['ls1'].get('msg', 'N/A')}` | {lifecycle['ls1'].get('time', 'N/A')} |")
            if 'ms2' in lifecycle:
                markdown.append(f"| **MS2** | `{lifecycle['ms2'].get('msg', 'N/A')}` | {lifecycle['ms2'].get('time', 'N/A')} |")
            if 'ls2' in lifecycle:
                markdown.append(f"| **LS2** | `{lifecycle['ls2'].get('msg', 'N/A')}` | {lifecycle['ls2'].get('time', 'N/A')} |")
            if 'df' in lifecycle:
                markdown.append(f"| **DF** | `{lifecycle['df'].get('msg', 'N/A')}` | {lifecycle['df'].get('time', 'N/A')} |")
            if 'set_func' in lifecycle:
                markdown.append(f"| **Set Function** | `{lifecycle['set_func'].get('name', 'N/A')}` | {lifecycle['set_func'].get('time', 'N/A')} |")
            markdown.append("")
        
        # 3. 设置值表格
        if 'set_values' in result:
            markdown.append("## ⚙️ 设置值\n")
            
            set_values = result['set_values']
            for key, value in set_values.items():
                markdown.append(f"### 部位: {key}")
                columns = value.get('columns', [])
                data_rows = value.get('data', [])
                
                if columns and data_rows:
                    # 创建表格头
                    header = "| " + " | ".join([col.replace('_', ' ').title() for col in columns]) + " |"
                    separator = "|" + "|".join(["---" for _ in columns]) + "|"
                    
                    markdown.append(header)
                    markdown.append(separator)
                    
                    # 添加数据行
                    for row in data_rows:
                        markdown.append("| " + " | ".join(row) + " |")
                    
                    markdown.append("")
        
        # # 4. 数据统计
        # if 'set_values' in data and 'data' in data['set_values']:
        #     markdown.append("## 📈 数据统计\n")
            
        #     data_rows = data['set_values']['data']
        #     speeds = [int(row[0]) for row in data_rows]
        #     values = [str(row[-1]) for row in data_rows]
            
        #     markdown.append("| 统计项 | 数值 |")
        #     markdown.append("|--------|------|")
        #     markdown.append(f"| **速度范围** | `{min(speeds)} - {max(speeds)}` |")
        #     markdown.append(f"| **数值范围** | `{min(values):.2f} - {max(values):.2f}` |")
        #     markdown.append(f"| **数据点数** | `{len(data_rows)}` |\n")
        
        # 5. 完整数据（可折叠）
        # markdown.append("## 📦 完整数据\n")
        # markdown.append("<details>")
        # markdown.append("<summary><b>点击查看完整JSON</b></summary>\n")
        # markdown.append("```json")
        # markdown.append(json.dumps(data, indent=2, ensure_ascii=False))
        # markdown.append("```")
        # markdown.append("</details>")
        markdown.append("\n\n --- \n")

        return "\n".join(markdown)
        
class PressrollMPEventExtractor(KeyEventExtractor):
    def __init__(self):
        # 调用父类初始化方法
        super().__init__()
        # glue part
        # set func call events
        self.set_func_call_events = [] # will be changed when MP4 is triggered

    def process(self, row):
        # print(row['EventId'])
        self.process_log_row(row) # for material change event
        # for glue event
        if row['EventId'] == 'MP4':
            parsed_values = row["ParsedValues"]
            # print(parsed_values)
            # 'set_func_name': 'SetPressRollSF2', 'ms_material': '8', 'ls_material': 'T', 'width': '2250', 'flute_type': 'B', 'base_value': '21'
            self.set_func_call_events.append({
                'func': parsed_values['set_func_name'], # SetPressRollSF2
                # 'part': parsed_values['part'], 
                'ms_material': parsed_values['ms_material'], # 8
                'ls_material': parsed_values['ls_material'], # T 
                'width': parsed_values['width'], 
                'flute_type': parsed_values['flute_type'],
                'base_value': parsed_values['base_value'],
                'time': str(row['Date'])
            })

    def get_pressroll_mp_set_function_full_event(self):
        pass
        # return all set funcs, and with lifecycle based on the machine material change event
        # print(f"material len: {len(self.material_events)}, setfunc len: {len(self.set_func_call_events)}")
        # print(self.material_events)
        # all_events = self.material_events + self.set_func_call_events
        # all_events.sort(key=lambda x: x['time'])
        # set_func_index = -1
        # for i in range(len(all_events)-1, -1, -1):
        #     if 'ms_material' in all_events[i]: # select vacuum blower set event
        #         # ms_material and ls_material is single material like "8", "P"
        #         # get change material events
        #         lifecycle = self.track_machine_material_lifecycle(all_events, i)
        #         # print(f"{self.set_func_call_events[set_func_index]['material']} -> {lifecycle}")
        #         self.set_func_call_events[set_func_index]['lifecycle'] = lifecycle
        #         # print(self.set_func_call_events[set_func_index]['lifecycle'])
        #         set_func_index -= 1
        # return self.set_func_call_events

        # markdown_results = ''
        # for result in self.set_func_call_events:
        #     md = self.convert_vacuum_blower_func_to_markdown(result)
        #     markdown_results += md
        #     # markdown_results += "\n---\n"

        # return self.set_func_call_events, markdown_results
    
    def track_machine_material_lifecycle(self, events, idx):
        pass

        # set_func_event = events[idx]
        # machine = set_func_event['part']
        # ms_ls = ('ms1', 'ls1') if machine == 'SF1' else ('ms2', 'ls2')
        # material_lifecycle = {
        #     ms_ls[0]: {
        #         'msg': '',
        #         'time': ''
        #     },
        #     ms_ls[1]: {
        #         'msg': '',
        #         'time': ''
        #     },
        #     'set_func': {
        #         'name': '',
        #         'time': ''
        #     }
        # }
        # # set func
        # material_lifecycle['set_func']['name'] = set_func_event['func']
        # material_lifecycle['set_func']['time'] = set_func_event['time']
        # # prepare part assign
        # is_set_part = [0, 0, 1]
        # part_2_idx = {
        #     ms_ls[0]: 0,
        #     ms_ls[1]: 1,
        # }
        # for i in range(idx-1, -1, -1):
        #     if sum(is_set_part) >= 3: # df and set func is two, and we need another part to match the material
        #         return material_lifecycle
        #     event = events[i]
        #     if 'func' in event:
        #         continue
        #     part = event['part']
        #     if part not in ms_ls:
        #         continue
        #     material_lifecycle[part]['msg'] = event['msg']
        #     material_lifecycle[part]['time'] = event['time']
        #     is_set_part[part_2_idx[part]] = 1
        # return material_lifecycle

    def convert_pressroll_mp_func_to_markdown(self, result):
        """
        将数据转换为Markdown格式
        """
        markdown = []
        
        # 1. 标题和基本信息
        markdown.append("# 📊 函数调用记录\n")
        
        # 基本信息表格
        markdown.append("## 📋 基本信息\n")
        markdown.append("| 字段 | 值 |")
        markdown.append("|------|-----|")
        markdown.append(f"| **函数** | `{result.get('func', 'N/A')}` |")
        markdown.append(f"| **部件** | `{result.get('part', 'N/A')}` |")
        markdown.append(f"| **基础值 x 门幅系数** | `{result.get('part', 'N/A')}` |")
        markdown.append(f"| **芯纸材质** | `{result.get('ms_material', 'N/A')}` |")
        markdown.append(f"| **里纸材质** | `{result.get('ls_material', 'N/A')}` |")
        markdown.append(f"| **瓦楞类型** | `{result.get('flute_type', 'N/A')}` |")
        markdown.append(f"| **时间** | `{result.get('time', 'N/A')}` |\n")
        
        # 2. 生命周期信息
        if 'lifecycle' in result:
            markdown.append("## 🔄 生命周期\n")
            markdown.append("| 阶段 | 信息 | 时间 |")
            markdown.append("|------|------|------|")
            
            lifecycle = result['lifecycle']
            if 'ls0' in lifecycle:
                markdown.append(f"| **LS0** | `{lifecycle['ls0'].get('msg', 'N/A')}` | {lifecycle['ls0'].get('time', 'N/A')} |")
            if 'ms1' in lifecycle:
                markdown.append(f"| **MS1** | `{lifecycle['ms1'].get('msg', 'N/A')}` | {lifecycle['ms1'].get('time', 'N/A')} |")
            if 'ls1' in lifecycle:
                markdown.append(f"| **LS1** | `{lifecycle['ls1'].get('msg', 'N/A')}` | {lifecycle['ls1'].get('time', 'N/A')} |")
            if 'ms2' in lifecycle:
                markdown.append(f"| **MS2** | `{lifecycle['ms2'].get('msg', 'N/A')}` | {lifecycle['ms2'].get('time', 'N/A')} |")
            if 'ls2' in lifecycle:
                markdown.append(f"| **LS2** | `{lifecycle['ls2'].get('msg', 'N/A')}` | {lifecycle['ls2'].get('time', 'N/A')} |")
            if 'df' in lifecycle:
                markdown.append(f"| **DF** | `{lifecycle['df'].get('msg', 'N/A')}` | {lifecycle['df'].get('time', 'N/A')} |")
            if 'set_func' in lifecycle:
                markdown.append(f"| **Set Function** | `{lifecycle['set_func'].get('name', 'N/A')}` | {lifecycle['set_func'].get('time', 'N/A')} |")
            markdown.append("")
        
        # 3. 设置值表格
        if 'set_values' in result:
            markdown.append("## ⚙️ 设置值\n")
            
            set_values = result['set_values']
            for key, value in set_values.items():
                markdown.append(f"### 部位: {key}")
                columns = value.get('columns', [])
                data_rows = value.get('data', [])
                
                if columns and data_rows:
                    # 创建表格头
                    header = "| " + " | ".join([col.replace('_', ' ').title() for col in columns]) + " |"
                    separator = "|" + "|".join(["---" for _ in columns]) + "|"
                    
                    markdown.append(header)
                    markdown.append(separator)
                    
                    # 添加数据行
                    for row in data_rows:
                        markdown.append("| " + " | ".join(row) + " |")
                    
                    markdown.append("")
        
        # # 4. 数据统计
        # if 'set_values' in data and 'data' in data['set_values']:
        #     markdown.append("## 📈 数据统计\n")
            
        #     data_rows = data['set_values']['data']
        #     speeds = [int(row[0]) for row in data_rows]
        #     values = [str(row[-1]) for row in data_rows]
            
        #     markdown.append("| 统计项 | 数值 |")
        #     markdown.append("|--------|------|")
        #     markdown.append(f"| **速度范围** | `{min(speeds)} - {max(speeds)}` |")
        #     markdown.append(f"| **数值范围** | `{min(values):.2f} - {max(values):.2f}` |")
        #     markdown.append(f"| **数据点数** | `{len(data_rows)}` |\n")
        
        # 5. 完整数据（可折叠）
        # markdown.append("## 📦 完整数据\n")
        # markdown.append("<details>")
        # markdown.append("<summary><b>点击查看完整JSON</b></summary>\n")
        # markdown.append("```json")
        # markdown.append(json.dumps(data, indent=2, ensure_ascii=False))
        # markdown.append("```")
        # markdown.append("</details>")
        markdown.append("\n\n --- \n")

        return "\n".join(markdown)