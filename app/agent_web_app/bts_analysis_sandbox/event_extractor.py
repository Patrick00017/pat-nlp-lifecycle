import pandas as pd
from datetime import datetime
import numpy as np
from constant import handle_func_to_splicer_part
from utils import material_part_count, is_material_equal
import uuid
from parse import parse

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
        # all event count
        self.event_count_dict = {}
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
            },
            'df': {
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
        }
        # raw order list
        # {'F_CreateTime': Timestamp('2026-06-22 00:00:00'), 'F_OrderID': '9102', 'F_MachineID': 'MS1', 'F_PaperCode': 'HD.07.07.07.B9', 'F_Flute': '5BA', 'F_Width': 2150, 'F_ErpPaperCode': '07', 'F_ErpWeight': Decimal('170.00'), 'F_ErpWidth': Decimal('2150.00')}
        self.order_init_data = []
        self.order_events = []
        
    def process_orders(self):
        for row in self.order_init_data:
            event = {
                'id': uuid.uuid1(),
                'type': 'order',
                'order_id': str(row['F_OrderID']),
                'time': str(row['F_CreateTime']),
                'machine': row['F_MachineID'],
                'paper_code': row['F_PaperCode'],
                'flute': row['F_Flute'],
                'width': int(row['F_Width']) if row.get('F_Width') else 0,
                'erp_paper_code': str(row.get('F_ErpPaperCode') or ''),
                'erp_weight': float(row['F_ErpWeight'] or 0),
                'erp_width': float(row['F_ErpWidth'] or 0),
            }
            self.order_events.append(event)
    
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
        if row["EventId"] in self.event_count_dict.keys():
            self.event_count_dict[row["EventId"]] = self.event_count_dict[row["EventId"]] + 1
        else:
            self.event_count_dict[row["EventId"]] = 1
                    
        # check log eventid
        if row["EventId"] == 'I1':
            content = row['ParsedValues']['content'] + ' '
            print(f"I1 -> {content}")
            # LS0横切换材了。上笔材质=B，门幅=3150；本批材质=B，门幅=3050\r\nLS0材质校准：当前正在用的材质=B，门幅=3150，校准后的材质=B，门幅=3050\r\n
            ls0_template1 = "{idk0}LS0材质校准：当前正在用的材质={material}，门幅={width}，校准后的材质={next_material}，门幅={next_width}\\r\\n{idk1}"
            ls0_template2 = "LS0横切换材了。上笔材质={material}，门幅={width}；本批材质={next_material}，门幅={next_width}\\r\\n{idk1}"
            ms1_template1 = "{idk0}MS1材质校准：当前正在用的材质={material}，门幅={width}，校准后的材质={next_material}，门幅={next_width}\\r\\n{idk1}"
            ms1_template2 = "{idk0}MS1横切换材了。上笔材质={material}，门幅={width}；本批材质={next_material}，门幅={next_width}\\r\\n{idk1}"
            ms1_template3 = "MS1横切换材了。上笔材质={material}，门幅={width}；本批材质={next_material}，门幅={next_width}\\r\\n{idk1}"
            ls1_template1 = "{idk0}LS1材质校准：当前正在用的材质={material}，门幅={width}，校准后的材质={next_material}，门幅={next_width}\\r\\n{idk1}"
            ls1_template2 = "{idk0}LS1横切换材了。上笔材质={material}，门幅={width}；本批材质={next_material}，门幅={next_width}\\r\\n{idk1}"
            ls1_template3 = "LS1横切换材了。上笔材质={material}，门幅={width}；本批材质={next_material}，门幅={next_width}\\r\\n{idk1}"
            ms2_template1 = "{idk0}MS2材质校准：当前正在用的材质={material}，门幅={width}，校准后的材质={next_material}，门幅={next_width}\\r\\n{idk1}"
            ms2_template2 = "{idk0}MS2横切换材了。上笔材质={material}，门幅={width}；本批材质={next_material}，门幅={next_width}\\r\\n{idk1}"
            ms2_template3 = "MS2横切换材了。上笔材质={material}，门幅={width}；本批材质={next_material}，门幅={next_width}\\r\\n{idk1}"
            ls2_template1 = "{idk0}LS2材质校准：当前正在用的材质={material}，门幅={width}，校准后的材质={next_material}，门幅={next_width}\\r\\n{idk1}"
            ls2_template2 = "{idk0}LS2横切换材了。上笔材质={material}，门幅={width}；本批材质={next_material}，门幅={next_width}\\r\\n{idk1}"
            ls2_template3 = "LS2横切换材了。上笔材质={material}，门幅={width}；本批材质={next_material}，门幅={next_width}\\r\\n{idk1}"
            
            part2templates = {
                "ls0": [ls0_template1, ls0_template2],
                "ms1": [ms1_template1, ms1_template2, ms1_template3],
                "ls1": [ls1_template1, ls1_template2, ls1_template3],
                "ms2": [ms2_template1, ms2_template2, ms2_template3],
                "ls2": [ls2_template1, ls2_template2, ls2_template3]
            }
            
            for k, v in part2templates.items():
                results = None
                for template in v:
                    results = parse(template, content)
                    if results != None:
                        break
                if results:
                    # update the state
                    prev_info = {
                        'material': self.splicer_state[k]['material'],
                        'width': self.splicer_state[k]['width'],
                        'flute_type': self.splicer_state[k]['flute_type']
                    }
                    current_info = {
                        'material': results['next_material'],
                        'width': results['next_width'],
                        'flute_type': self.splicer_state[k]['flute_type'] if self.splicer_state[k]['flute_type'] != '-' else '-'
                    }
                    self.splicer_state[k] = {
                        'material': current_info['material'],
                        'width': current_info['width'],
                        'flute_type': current_info['flute_type'],
                        'next_batch': {
                            'material': '-',
                            'width': 0,
                            'flute_type': '-',
                        },
                        'change_time': str(row['Date'])
                    }
                    # generate the event
                    event = {
                        'id': uuid.uuid1(),
                        'part': k,
                        'type': 'material',
                        'msg': f"({prev_info['material']},{prev_info['width']},{prev_info['flute_type']}) -> ({current_info['material']},{current_info['width']},{current_info['flute_type']})",
                        'time': str(row['Date']),
                        'reason': 'hq'
                    }
                    print(event)
                    self.material_events.append(event)
        
        elif row["EventId"] == "I7":
            # get change paper ready event, and next material.
            # save the next batch based on log info
            # for sf material change ready
            parsed_values = row["ParsedValues"]
            handle_func_name = parsed_values["handle_func_name"]
            part = handle_func_to_splicer_part[handle_func_name]
            # assign next material batch
            self.splicer_state[part]["next_batch"] = {
                'material': parsed_values["next_material"],
                'width': int(parsed_values["next_width"]),
                'flute_type': parsed_values["next_flute_type"].split('，')[0]
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
            if current_material_batch['material'] == '-':
                return
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
                'id': uuid.uuid1(),
                'part': part,
                'type': 'material',
                'msg': f"({prev_material_batch['material']},{prev_material_batch['width']},{prev_material_batch['flute_type']}) -> ({current_material_batch['material']},{current_material_batch['width']},{current_material_batch['flute_type']})",
                'time': str(row['Date']),
                'reason': 'normal'
            }
            self.material_events.append(event)
        elif row['EventId'] == 'I11':
            # df is ready to change paper, get information
            # for df change material ready
            parsed_values = row["ParsedValues"]
            # print(f"I11 -> {parsed_values}")
            # print(parsed_values) # {'module': '换材判定模块', 'ip': '172.32.64.10', 'host': 'BTS-SHLY-SVR', 'username': 'null', 'prev_material': 'T.-.-.7.T', 'prev_flute_type': '3B', 'prev_width': '2400', 'material': 'P.-.-.8.J', 'flute_type': '3B', 'width': '2350', 'next_material': 'P.-.-.8.J'}
            self.splicer_state['df']['next_batch'] = {
                'material': parsed_values['material'],
                'width': int(parsed_values['width']),
                'flute_type': parsed_values['flute_type'],
            }
        elif row['EventId'] == 'I12':
            # df is changed paper, generate event
            # for df change material check
            parsed_values = row["ParsedValues"]
            # print(f"I12 -> {parsed_values}")
            # print(parsed_values) # {'module': '换材判定模块', 'ip': '172.32.64.10', 'host': 'BTS-SHLY-SVR', 'username': 'null', 'handle_func_name': 'HandleGuChangePaper'}
            # save material for event generate
            prev_material_batch = {
                'material': self.splicer_state['df']["material"],
                'width': self.splicer_state['df']["width"],
                'flute_type': self.splicer_state['df']["flute_type"]
            }
            current_material_batch = self.splicer_state['df']["next_batch"]
            # update the state
            self.splicer_state['df'] = {
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
                'id': uuid.uuid1(),
                'part': 'df',
                'type': 'material',
                'msg': f"({prev_material_batch['material']},{prev_material_batch['width']},{prev_material_batch['flute_type']}) -> ({current_material_batch['material']},{current_material_batch['width']},{current_material_batch['flute_type']})",
                'time': str(row['Date']),
                'reason': 'normal'
            }
            self.material_events.append(event)
        
        elif row['EventId'] == 'I13':
            # 实际材质，等于直接换
            parsed_values = row["ParsedValues"]
            # 糊机实材={material},楞型={flute_type},门幅={width}
            current_material_batch = {
                'material': parsed_values['material'],
                'flute_type': parsed_values['flute_type'],
                'width': parsed_values['width']
            }
            prev_material_batch = {
                'material': self.splicer_state['df']["material"],
                'width': self.splicer_state['df']["width"],
                'flute_type': self.splicer_state['df']["flute_type"]
            }
            # update the df state
            self.splicer_state['df'] = {
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
            # generate the df event
            event = {
                'id': uuid.uuid1(),
                'part': 'df',
                'type': 'material',
                'msg': f"({prev_material_batch['material']},{prev_material_batch['width']},{prev_material_batch['flute_type']}) -> ({current_material_batch['material']},{current_material_batch['width']},{current_material_batch['flute_type']})",
                'time': str(row['Date']),
                'reason': '实际材质'
            }
            self.material_events.append(event)
        
        elif row['EventId'] == 'I4':
            # 实际材质，等于直接换
            parsed_values = row["ParsedValues"]
            # print(f"I4 -> {parsed_values}")
            part2part = {
                'LS0': 'ls0',
                'MS1': 'ms1',
                'LS1': 'ls1',
                'MS2': 'ms2',
                'LS2': 'ls2',
                'MS3': 'ms3',
                'LS3': 'ls3'
            }
            part = part2part[parsed_values['splicer_part']]
            real_material = parsed_values['real_material']
            real_width = int(float(parsed_values['real_width']))
            prev_material_batch = {
                'material': self.splicer_state[part]["material"],
                'width': self.splicer_state[part]["width"],
                'flute_type': self.splicer_state[part]["flute_type"]
            }
            # update the state
            self.splicer_state[part] = {
                'material': real_material,
                'width': real_width,
                'flute_type': prev_material_batch['flute_type'],
                'next_batch': {
                    'material': '-',
                    'width': 0,
                    'flute_type': '-',
                },
                'change_time': str(row['Date'])
            }
            # generate the event
            event = {
                'id': uuid.uuid1(),
                'part': part,
                'type': 'material',
                'msg': f"({prev_material_batch['material']},{prev_material_batch['width']},{prev_material_batch['flute_type']}) -> ({real_material},{real_width},{prev_material_batch['flute_type']})",
                'time': str(row['Date']),
                'reason': '实际材质'
            }
            self.material_events.append(event)
        
        # I16: InitInfos 包含所有部位当前材质（PG 特有），初始化或者重置时触发
        elif row['EventId'] == 'I16':
            parsed_values = row['ParsedValues']
            # print(f"I16 -> {parsed_values}")
            # todo: ms3, ls3
            for part in ['df', 'ls0', 'ms1', 'ls1', 'ms2', 'ls2']:
                prev_material_batch = {
                    'material': self.splicer_state[part]["material"],
                    'width': self.splicer_state[part]["width"],
                    'flute_type': self.splicer_state[part]["flute_type"]
                }
                current_material_batch = {
                    'material': parsed_values[f"{part}_material"],
                    'width': parsed_values[f"{part}_width"],
                    'flute_type': parsed_values[f"{part}_flute"]
                }
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
                    'id': uuid.uuid1(),
                    'part': part,
                    'type': 'material',
                    'msg': f"({prev_material_batch['material']},{prev_material_batch['width']},{prev_material_batch['flute_type']}) -> ({current_material_batch['material']},{current_material_batch['width']},{current_material_batch['flute_type']})",
                    'time': str(row['Date']),
                    'reason': 'reset'
                }
                self.material_events.append(event)
            
    
    
                

class GlueEventExtractor(KeyEventExtractor):
    """
    目前的事件路径：
    1. 常规GU赋值 G7(start) -> G14(calculate done) -> G12(complete)
    2. 由于用户没有启用的GU赋值 G7(start) -> G14(calculate done) -> G13(disable)
    3. 常规SF赋值 G2(start) -> G4(calculate done) -> G5(complete)
    4. 由于用户没有启用的SF赋值 G2(start) -> G4(calculate done) -> G13(disable)
    5. 立即赋值 G11(handleNow) -> SF -> G2
                              -> GU -> G7
    """
    
    def __init__(self):
        # 调用父类初始化方法
        super().__init__()
        # glue part
        # set func call events
        self.set_func_call_events = [] # will be changed when G12, G5 is triggered
        # gu的目前计算配方
        self.gu_value_state = {
            'GU1': {},
            'GU2': {},
            'GU3': {}
        }
        # gu的准备事件和中间事件
        self.gu_current_info = {}
        # sf的目前计算配方，键值对part-》配方，在完成事件出现后清理
        self.sf_value_state = {}
        # 记录sf的准备事件和中间事件出现的，值键值对part-》信息，在完成事件出现后清理
        self.sf_current_info = {
            'SF1': {},
            'SF2': {},
            'SF3': {}
        }
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
            # complete signal
            # setgluegu set gu value
            # based on self.gu_value_state, so G14 is triggered before G12
            parsed_values = row["ParsedValues"]
            # generate func event
            for k, v in self.gu_value_state.items():
                if v != {}:
                    event = {
                        'id': uuid.uuid1(),
                        'func': parsed_values['set_func_name'],
                        'part': k,
                        'type': 'glue',
                        'material': parsed_values['material'],
                        'flute_type': parsed_values['flute_type'],
                        'set_values': v, 
                        'time': str(row['Date']),
                        'event_issue': "normal",
                        'from': self.gu_current_info.get("from", 'normal')
                    }
                    self.set_func_call_events.append(event)
            # clear gu value state
            self.gu_value_state = {}
            self.gu_current_info = {}
        elif row['EventId'] == 'G5':
            # complete signal
            # setgluesf1/2 set gu value
            # based on self.sf_value_state, so G4 is triggered before G5
            parsed_values = row["ParsedValues"]
            # print(f"{row['EventId']} -> {parsed_values}")
            glue_part = parsed_values['glue_part']
            info = self.sf_current_info[glue_part]
            # generate func event
            event = {
                'id': uuid.uuid1(),
                'func': parsed_values['set_func_name'],
                'part': parsed_values['glue_part'],
                'type': 'glue',
                'material': parsed_values['material'],
                'flute_type': parsed_values['flute_type'],
                'set_values': self.sf_value_state.get(glue_part, {}), # align the sf glue set function and gu glue set function
                'time': str(row['Date']),
                'event_issue': "normal",
                'from': info.get("from", 'normal')
            }
            self.sf_value_state[glue_part] = {}
            self.sf_current_info[glue_part] = {}
            self.set_func_call_events.append(event)
        elif row['EventId'] == 'G4': # SF calculate value
            parsed_values = row["ParsedValues"]
            # print(f"{row['EventId']} -> {parsed_values}")
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
            # print(f"{row['EventId']} -> {parsed_values}")
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
            # print("set.............................")
            # print(self.gu_value_state)
        elif row['EventId'] == 'G7':
            # GU要开始赋值
            parsed_values = row["ParsedValues"]     
            # 进入 {set_func_name} 准备点位赋值，材质={material},楞型={flute_type}
            material = parsed_values['material']
            flute_type = parsed_values['flute_type']
            # GU部分用的是全材质，所以不需要分
            self.gu_current_info['material'] = material
            self.gu_current_info['flute_type'] = flute_type
        elif row['EventId'] == 'G2':
            # SF部位开始赋值的准备事件
            # SetGlueSF1       
            parsed_values = row["ParsedValues"]     
            set_func_name = parsed_values['set_func_name']
            # print(f"G2 -> {parsed_values}, {set_func_name.split('SetGlue')}")
            part = set_func_name.split('SetGlue')[1]
            # info = {
            #     'ms_material': parsed_values['medium_material'],
            #     'ls_material': parsed_values['liner_material'],
            #     'flute_type': parsed_values['flute_type']
            # }
            self.sf_current_info[part]['ms_material'] = parsed_values['medium_material']
            self.sf_current_info[part]['ls_material'] = parsed_values['liner_material']
            self.sf_current_info[part]['flute_type'] = parsed_values['flute_type']
            
        elif row['EventId'] == 'G13':
            # 由于没有开启开关，所以没有赋值的事件
            # 应该是SF和GU公用
            # complete signal
            parsed_values = row["ParsedValues"]
            # print(f"G13 -> {parsed_values}")
            part = parsed_values['glue_part']
            if part.startswith("SF"):
                # SF
                info = self.sf_current_info[part]
                # print(info)
                if info == {}:
                    return
                # generate func event
                event = {
                    'id': uuid.uuid1(),
                    'func': parsed_values['set_func_name'],
                    'part': parsed_values['glue_part'],
                    'type': 'glue',
                    'material': f"{info['ms_material']}/{info['ls_material']}",
                    'flute_type': info['flute_type'],
                    'set_values': self.sf_value_state[part], 
                    'time': str(row['Date']),
                    'event_issue': "disable",
                    'from': info.get("from", 'normal')
                }
                # clear gu value state
                self.sf_value_state[part] = {}
                self.sf_current_info[part] = {}
                self.set_func_call_events.append(event)
            else:
                # GU
                if self.gu_current_info == {}: # 一上来就有不赋值的情况，忽略
                    return
                # generate func event
                event = {
                    'id': uuid.uuid1(),
                    'func': parsed_values['set_func_name'],
                    'part': parsed_values['glue_part'],
                    'type': 'glue',
                    'material': self.gu_current_info['material'],
                    'flute_type': self.gu_current_info['flute_type'],
                    'set_values': self.gu_value_state[parsed_values['glue_part']], 
                    'time': str(row['Date']),
                    'event_issue': "disable",
                    'from': info.get("from", 'normal')
                }
                # clear gu value state
                self.gu_value_state[part] = {}
                self.gu_current_info = {}
                self.set_func_call_events.append(event)
        elif row['EventId'] == 'G11':
            # handle now
            parsed_values = row["ParsedValues"]
            part = parsed_values['glue_part']
            if part.startswith("SF"):
                self.sf_current_info[part]['from'] = 'handleNow'
            else:
                self.gu_current_info['from'] = 'handleNow'

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

    def get_all_events(self):
        # return all set funcs, and with lifecycle based on the machine material change event
        print(f"material len: {len(self.material_events)}, setfunc len: {len(self.set_func_call_events)}, order len: {len(self.order_events)}")
        all_events = self.material_events + self.set_func_call_events + self.order_events
        all_events.sort(key=lambda x: x['time'])
        return all_events
    
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
        return ""


class WarpEventExtractor(KeyEventExtractor):
    def __init__(self):
        super().__init__()
        self.cur_exec_status = ''
        self.prev_exec_status = ''
        self.detection_status = ''
        self.detection_degree = 0.0
        self.auto_adjust_events = []
        self.reset_events = []
        self.manual_adjust_events = []
        self.paper_change_events = []
        self.warp_raw_events = []

    def process(self, row):
        self.process_log_row(row)
        if pd.isna(row.get('EventId')):
            return
        if not str(row['EventId']).startswith('WARP'):
            return
        self.warp_raw_events.append(row.to_dict())
        eid = row['EventId']
        pv = row.get('ParsedValues') or {}
        if eid == 'WARP1':
            self.prev_exec_status = self.cur_exec_status
            self.cur_exec_status = ''
            self.reset_events.append({
                'type': 'auto' if pv.get('isAutoExeRest') else 'unknown',
                'time': str(row['Date'])
            })
        elif eid == 'WARP2':
            self.auto_adjust_events.append({
                'mode': 'auto',
                'action': 'exec',
                'time': str(row['Date'])
            })
        elif eid == 'WARP3':
            self.manual_adjust_events.append({
                'mode': 'manual',
                'action': 'exec',
                'time': str(row['Date'])
            })
        elif eid == 'WARP4':
            self.prev_exec_status = self.cur_exec_status
            self.cur_exec_status = ''
            self.reset_events.append({
                'type': 'manual',
                'time': str(row['Date'])
            })
        elif eid == 'WARP5':
            self.prev_exec_status = self.cur_exec_status
            self.cur_exec_status = pv.get('action', '')
            self.auto_adjust_events.append({
                'mode': 'auto',
                'action': pv.get('action', ''),
                'prev_status': pv.get('prev_status', ''),
                'cur_status': pv.get('cur_status', ''),
                'time': str(row['Date'])
            })
        elif eid == 'WARP6':
            self.prev_exec_status = self.cur_exec_status
            self.cur_exec_status = ''
            self.auto_adjust_events.append({
                'mode': 'auto',
                'action': 'reset',
                'prev_status': pv.get('prev_status', ''),
                'cur_status': pv.get('cur_status', ''),
                'time': str(row['Date'])
            })
        elif eid == 'WARP7':
            self.paper_change_events.append({
                'type': 'tracking',
                'df_remain': pv.get('df_remain'),
                'gu_range1': pv.get('gu_range1'),
                'gu_range2': pv.get('gu_range2'),
                'gu_range3': pv.get('gu_range3'),
                'time': str(row['Date'])
            })
        elif eid == 'WARP8':
            self.paper_change_events.append({
                'type': 'entered_change',
                'df_remain': pv.get('df_remain'),
                'time': str(row['Date'])
            })
        elif eid == 'WARP10':
            warp_data = pv.get('warp_data', '')
            if warp_data:
                try:
                    import json
                    cleaned = warp_data.replace('\\"', '"')
                    data = json.loads(cleaned)
                    self.detection_status = data.get('WarpState', '')
                    self.detection_degree = float(data.get('WarpDegree', 0.0))
                except Exception:
                    pass

    def get_summary(self):
        return {
            'total_warp_events': len(self.warp_raw_events),
            'auto_adjust_count': len(self.auto_adjust_events),
            'reset_count': len(self.reset_events),
            'manual_adjust_count': len(self.manual_adjust_events),
            'paper_change_count': len(self.paper_change_events),
            'cur_exec_status': self.cur_exec_status,
            'detection_status': self.detection_status,
            'detection_degree': self.detection_degree,
        }