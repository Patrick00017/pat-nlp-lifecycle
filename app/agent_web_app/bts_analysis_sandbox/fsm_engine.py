"""
FSM-based Glue Gap Diagnostic Engine
每个糊间隙位置（GU1/GU2/GU3/SF1/SF2/SF3）独立状态机，
事件驱动状态转移，校验失败进入错误节点。
"""

import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import defaultdict
from event_extractor import GlueEventExtractor
from parse import parse
import json
import decimal
import array
import uuid
class CustomJsonEncoder(json.JSONEncoder):
    """
    Json解析器，解决识别Decimal出错的问题
    """

    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        if isinstance(obj, int):
            return int(obj)
        elif isinstance(obj, float):
            return float(obj)
        elif isinstance(obj, decimal.Decimal):
            return float(obj)
        elif isinstance(obj, (uuid.UUID,)):
            return str(obj)
        else:
            return super(CustomJsonEncoder, self).default(obj)


# GU -> QDM col
qdm_col_map = {'GU1': 'F_Glue1', 'GU2': 'F_Glue2', 'GU3': 'F_Glue3'}

# ── 状态枚举 ──

class GlueState(Enum):
    IDLE = 'idle'
    MATERIAL_CHANGED = 'material_changed'
    IMMEDIATE_CHANGE = 'immediate_change'
    DELAY_WAITING = 'delay_waiting'
    G10_FALLBACK = 'g10_fallback'
    CANCELLED_DELAY = 'cancelled_delay'
    CALCULATED = 'calculated'
    CANCELLED_PRE_WRITE = 'cancelled_pre_write'
    COMPLETE = 'complete'

"""
                 ┌──────────┐
                 │   IDLE   │
                 └────┬─────┘
                  ┌───┴───┐
                  │       │
               G7/G1    G11
                  │       │
          ┌───────▼───────▼────────┐
          │  MATERIAL_CHANGED      │
          │  (材质变更/立即换材)    │
          └───────┬───────────────┘
              ┌───┴───┐
              │       │
             G10     G8
              │       │
          ┌───▼─┐  ┌──▼──────────────┐
          │FBK  │  │ CANCELLED_DELAY ❌│
          └───┬─┘  │ (被抢断)         │
              │    └─────────────────┘
              │
          ┌───▼──────────────────┐
          │ DELAY_WAITING        │
          └───┬──────────────────┘
              │
       ┌──────┴──────┐
       │             │
    G14 (GU)     G4 (SF)
       │             │
  ┌────▼────┐  ┌────▼────┐
  │ CALC    │  │ CALC    │
  │  +校验  │  │  +校验  │
  │  +跨来源│  │  +跨来源│
  └────┬────┘  └────┬────┘
       │             │
   ┌───┴───┐    ┌───┴───┐
   │       │    │       │
  G12     G15  G5     G15
   │       │    │       │
 ┌─▼──┐ ┌──▼──────────────┐
 │ OK │ │ CANCELLED_WRITE❌│
 │ ✓  │ │ (写值取消)       │
 └────┘ └─────────────────┘
   │
   └──→ IDLE (复位)
"""

# ── 问题枚举 ──

class IssueType(Enum):
    MATERIAL_DISMATCH = 'material_dismatch'
    QDM_DISMATCH = 'qdm_dismatch'
    QDM_NOT_EXIST = 'qdm_not_exist'
    WEIGHT_DISMATCH = 'weight_dismatch'
    WEIGHT_NOT_EXIST = 'weight_not_exist'
    BASEDOC_DISMATCH = 'basedoc_dismatch'
    BASEDOC_NOT_EXIST = 'basedoc_not_exist'
    SPEED_COEF_NOT_EXIST = 'speed_coef_not_exist'
    SPEED_COEF_DISMATCH = 'speed_coef_dismatch'
    NO_SET_VALUES = 'no_set_values'

class WarningType(Enum):
    CANCEL = 'cancel'

class PassType(Enum):
    MATERIAL_PASS = 'material_pass'
    QDM_PASS = 'qdm_pass'
    SPEED_PASS = 'speed_pass'
    WEIGHT_PASS = 'weight_pass'
    BASEDOC_PASS = 'basedoc_pass'
    

# ── 值得记录的关键事件，比如被取消 ──
@dataclass
class Warning:
    detail: str
    type: WarningType
    args: object

    def to_dict(self):
        args = self.args
        if isinstance(args, dict):
            args = {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                    for k, v in args.items()}
        return {
            'detail': self.detail,
            'type': self.type.value if isinstance(self.type, WarningType) else self.type,
            'args': args,
        }

# ── 问题记录 ──
@dataclass
class Issue:
    detail: str
    type: IssueType
    args: object

    def to_dict(self):
        args = self.args
        if isinstance(args, dict):
            args = {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                    for k, v in args.items()}
        return {
            'detail': self.detail,
            'type': self.type.value if isinstance(self.type, IssueType) else self.type,
            'args': args,
        }

# ── 问题记录 ──
@dataclass
class Pass:
    detail: str
    type: PassType
    args: object

    def to_dict(self):
        args = self.args
        if isinstance(args, dict):
            args = {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                    for k, v in args.items()}
        return {
            'detail': self.detail,
            'type': self.type.value if isinstance(self.type, PassType) else self.type,
            'args': args,
        }

# ── 单个位置 FSM ──

_POSITION_GU = ('GU1', 'GU2', 'GU3')
_POSITION_SF = ('SF1', 'SF2', 'SF3')
_ALL_POSITIONS = _POSITION_GU + _POSITION_SF


class PositionFSM:
    """
    单个糊间隙位置（GU1/GU2/...）的状态机。
    
    也可以不算一个状态机，只不过是一直读取环境变换信息，在成功赋值时检验一下
    状态机有如下状态：
    1. IDLE: 初始状态
    2. MATERIAL_CHANGE: 材质发生变化
    3. SET_OK: 赋值完成，处理后回归IDLE
    
    """

    def __init__(self, position: str, dev_ips=None):
        self.position = position
        self.state = GlueState.IDLE
        
        # material part
        self.material_event = {
            'ls0': {
                'id': 0,
                'material': '-',
                'width': 0,
                'flute': '-'
            },
            'ms1': {
                'id': 0,
                'material': '-',
                'width': 0,
                'flute': '-'
            },
            'ls1': {
                'id': 0,
                'material': '-',
                'width': 0,
                'flute': '-'
            },
            'ms2': {
                'id': 0,
                'material': '-',
                'width': 0,
                'flute': '-'
            },
            'ls2': {
                'id': 0,
                'material': '-',
                'width': 0,
                'flute': '-'
            },
            'df': {
                'id': 0,
                'material': '-.-.-.-.-',
                'width': 0,
                'flute': '-'
            },
        }
        
        self.qdm_event_id = 0
        self.weight_event_id = 0
        self.basedoc_event_id = 0
        self.speed_coef_id = 0
        
        self.material = ''
        self.flute = ''
        self.computed_segments = []
        self.dev_ips = dev_ips

        self.full_events = []

    # ── 事件分发 ──

    def process_event(self, event: dict) -> bool:
        type = event.get('type')
        handler = getattr(self, f'_on_{type}', None)
        if handler:
            handler(event)
            return True
        return False

    def _on_material(self, event):
        """
        event: {
            'id': uuid.uuid1(),
            'part': part,
            'msg': f"({prev_material_batch['material']},{prev_material_batch['width']},{prev_material_batch['flute_type']}) -> ({current_material_batch['material']},{current_material_batch['width']},{current_material_batch['flute_type']})",
            'time': str(row['Date']),
            'reason': 'reset'
        }
        
        msg: (-,0,-) -> (9.9.9.9.9,2000,5BA)
        """
        msg_template = "({prev_material},{prev_width},{prev_flute}) -> ({material},{width},{flute})"
        parsed = parse(msg_template, event['msg'])
        if parsed is None:
            return
        pv = parsed.named

        material = pv['material']
        width = int(pv['width'])
        flute = pv['flute']
        part = event['part']
        self.material_event[part] = {
            'id': event['id'],
            'material': material,
            'width': width,
            'flute': flute
        }
        
    def _on_glue(self, event):
        """
        event:{
                'id': uuid.uuid1(),
                'func': parsed_values['set_func_name'],
                'part': self.glue_part,
                'type': 'glue',
                'material': parsed_values['material'],
                'flute_type': parsed_values['flute_type'],
                'set_values': self.gu_value_state[self.glue_part], 
                'time': str(row['Date']),
                'event_issue': 'normal' or 'disable'
            }
        """
        
        warnings = []
        passes = []
        errors = []
        self.material = event['material']
        self.flute = event['flute_type']
        
        # apply some check
        # 1. material, if dismatch, return error and material change id
        if self.position.startswith("SF"):
            ms_material = self.material_event[f"ms{self.position[2]}"]['material']
            ls_material = self.material_event[f"ls{self.position[2]}"]['material']
            if len(event['material'].split('/')) == 1:
                return
            
            if event['material'].split('/')[0] != ms_material:
                errors.append(Issue("材质匹配失败", IssueType.MATERIAL_DISMATCH, {'id': self.material_event[f"ms{self.position[2]}"]['id'], 'msg':f"ms{self.position[2]},赋值材质：{event['material'].split('/')[0]},目前材质:{ms_material}"}))
            else:
                passes.append(Pass("材质匹配成功", PassType.MATERIAL_PASS, {'id': self.material_event[f"ms{self.position[2]}"]['id'], 'msg':f"ms{self.position[2]},赋值材质：{event['material'].split('/')[0]},目前材质:{ms_material}"}))
            
            if event['material'].split('/')[1] != ls_material:
                errors.append(Issue("材质匹配失败", IssueType.MATERIAL_DISMATCH, {'id': self.material_event[f"ls{self.position[2]}"]['id'], 'msg':f"ls{self.position[2]},赋值材质：{event['material'].split('/')[1]},目前材质:{ls_material}"}))   
            else:
                passes.append(Pass("材质匹配成功", PassType.MATERIAL_PASS, {'id': self.material_event[f"ls{self.position[2]}"]['id'], 'msg':f"ls{self.position[2]},赋值材质：{event['material'].split('/')[1]},目前材质:{ls_material}"})) 
        else:
            if event['material'] != self.material_event['df']['material']:
                errors.append(Issue("材质匹配失败", IssueType.MATERIAL_DISMATCH, {'id': self.material_event['df']['id'], 'msg': f"DF材质匹配失败，赋值材质：{event['material']}，目前材质：{self.material_event['df']['material']}"}))
            else:
                passes.append(Pass("材质匹配成功", PassType.MATERIAL_PASS, {'id': self.material_event['df']['id'], 'msg': f"DF材质匹配成功，赋值材质：{event['material']}，目前材质：{self.material_event['df']['material']}"}))
        
        # 2. no set_values
        if event['set_values'] == {}:
            errors.append(Issue("无计算结果", IssueType.NO_SET_VALUES, {}))

        # extract calculation segments
        segments = self._parse_segments(event['set_values'])
        # 2. qdm
        qdm_result = self._run_qdm_factor_check(segments)
        if qdm_result != None:
            if isinstance(qdm_result, Issue):
                errors.append(qdm_result)
            elif isinstance(qdm_result, Warning):
                warnings.append(qdm_result)
            elif isinstance(qdm_result, Pass):
                passes.append(qdm_result)
        # 3. weight
        weight_result = self._run_weight_factor_check(segments)
        if weight_result != None:
            if isinstance(weight_result, Issue):
                errors.append(weight_result)
            elif isinstance(weight_result, Warning):
                warnings.append(weight_result)
            elif isinstance(weight_result, Pass):
                passes.append(weight_result)
        # 4. base doc
        basedoc_result = self._run_base_setting_check(segments)
        if basedoc_result != None:
            if isinstance(basedoc_result, Issue):
                errors.append(basedoc_result)
            elif isinstance(basedoc_result, Warning):
                warnings.append(basedoc_result)
            elif isinstance(basedoc_result, Pass):
                passes.append(basedoc_result)
        # 5. speed coef
        speed_coef_result = self._run_speed_coef_check(segments)
        if speed_coef_result != None:
            if isinstance(speed_coef_result, Issue):
                errors.append(speed_coef_result)
            elif isinstance(speed_coef_result, Warning):
                warnings.append(speed_coef_result)
            elif isinstance(speed_coef_result, Pass):
                passes.append(speed_coef_result)
        
        # 1. is disable
        if event['event_issue'] == 'disable':
            # this event is been canceled
            warnings.append(Warning("部位未启用", WarningType.CANCEL, {}))
        full_event = event
        full_event['errors'] = errors
        full_event['warnings'] = warnings
        full_event['passes'] = passes
        self.full_events.append(full_event)

    def _parse_segments(self, pv):
        # sf:
        # ['speed', 'min_glue', 'max_glue', 'min_weight', 'max_weight', 'current_glue_weight', 'speed_factor', 'min_speed', 'qdm_factor', 'ui_factor', 'offset', 'value']
        # df:
        # ['speed', 'min_glue', 'max_glue', 'min_weight', 'max_weight', 'current_glue_weight', 'speed_factor', 'min_speed', 'qdm_factor', 'ui_factor', 'value']
        segments = []
        data = pv.get('data')
        if self.position.startswith("SF"):
            for i in range(0, 8):
                speed = data[i][0]
                min_glue = data[i][1]
                max_glue = data[i][2]
                min_weight = data[i][3]
                max_weight = data[i][4]
                current_glue_weight = data[i][5]
                speed_factor = data[i][6]
                min_speed = data[i][7]
                qdm_factor = data[i][8]
                ui_factor = data[i][9]
                warp_offset = data[i][10]
                value = data[i][11]
                segments.append({
                    'speed': speed, 'value': value,
                    'min_glue': min_glue, 'max_glue': max_glue,
                    'min_weight': min_weight, 'max_weight': max_weight,
                    'cur_weight': current_glue_weight, 'speed_factor': speed_factor,
                    'qdm_factor': qdm_factor, 'ui_factor': ui_factor,
                    'warp_offset': warp_offset, 'min_speed': min_speed
                })
        else:
            for i in range(0, 8):
                speed = data[i][0]
                min_glue = data[i][1]
                max_glue = data[i][2]
                min_weight = data[i][3]
                max_weight = data[i][4]
                current_glue_weight = data[i][5]
                speed_factor = data[i][6]
                min_speed = data[i][7]
                qdm_factor = data[i][8]
                ui_factor = data[i][9]
                value = data[i][10]
                segments.append({
                    'speed': speed, 'value': value,
                    'min_glue': min_glue, 'max_glue': max_glue,
                    'min_weight': min_weight, 'max_weight': max_weight,
                    'cur_weight': current_glue_weight, 'speed_factor': speed_factor,
                    'qdm_factor': qdm_factor, 'ui_factor': ui_factor,
                    'min_speed': min_speed
                })
        # print(f"segs: {segments}")
        return segments

    # ── 校验方法（由 _run_checks 统一调用） ──

    def _run_qdm_factor_check(self, segments):
        # print(f"{self.position} -> check qdm")
        first = segments[0]
        # print(f"{self.position} -> {first}")
        if not segments or not self.material:
            return
        if not self.dev_ips:
            expected_qdm = 1.0
            if expected_qdm != first['qdm_factor']:
                return Issue("QDM系数不匹配", IssueType.QDM_DISMATCH, {'qdm_id': self.qdm_event_id, 'msg': f"QDM设置:{expected_qdm}, QDM实际使用:{first['qdm_factor']}, 匹配不上"})
            else:
                return Pass("QDM模拟检验完成", PassType.QDM_PASS, {'qdm_id': self.qdm_event_id, 'msg': f"QDM设置:{expected_qdm}, QDM实际使用:{first['qdm_factor']}, 检验完成"})
        else:
            # QDM 校验
            if '/' in self.material and self.position.startswith('SF'):
                ms, ls = self.material.split('/')
                rows = self._query_ips('SELECT "F_Glue" FROM "TB_IPS_QdmCoefSF" WHERE "F_MS" = %s AND "F_LS" = %s AND "F_Flute" = %s', (ms, ls, self.flute))
                if rows and rows[0][0]:
                    expected_qdm = float(rows[0][0])
                    actual_qdm = first.get('qdm_factor')
                    # print(f"expected_qdm: {expected_qdm}\r\nactual_qdm: {actual_qdm}")
                    if actual_qdm is not None:
                        if abs(float(actual_qdm) - expected_qdm) > 0:
                            return Issue("QDM系数不匹配", IssueType.QDM_DISMATCH, {'qdm_id': self.qdm_event_id, 'msg': f"QDM设置:{expected_qdm}, QDM实际使用:{actual_qdm}, 匹配不上"})
                        else:
                            return Pass("QDM检验完成", PassType.QDM_PASS, {'qdm_id': self.qdm_event_id, 'msg': f"QDM设置:{expected_qdm}, QDM实际使用:{actual_qdm}, 检验完成"})
                else:
                    return Issue("QDM系数查询不到", IssueType.QDM_NOT_EXIST, {'qdm_id': self.qdm_event_id, 'msg': f"无对应QDM配置"}) # todo
            else:
                # 使用压缩码（去掉 -）
                compact = '.'.join([p for p in self.material.split('.') if p != '-'])
                col = qdm_col_map.get(self.position)
                if not col or not compact:
                    return None

                rows = self._query_ips(
                    f'SELECT "{col}" FROM "TB_IPS_QdmCoefDF" WHERE "F_Paper" = %s AND "F_Flute" = %s',
                    (compact, self.flute)
                )
                # LIKE 回退
                if not rows and '-' in self.material:
                    like_pat = compact.replace('-', '_')
                    rows = self._query_ips(
                        f'SELECT "{col}" FROM "TB_IPS_QdmCoefDF" WHERE "F_Paper" LIKE %s AND "F_Flute" = %s LIMIT 5',
                        (like_pat, self.flute)
                    )

                if rows and rows[0][0]:
                    expected_qdm = float(rows[0][0])
                    actual_qdm = first.get('qdm_factor')
                    if actual_qdm is not None and abs(float(actual_qdm) - expected_qdm) > 0:
                        return Issue("QDM系数不匹配", IssueType.QDM_DISMATCH, {'qdm_id': self.qdm_event_id, 'msg': f"QDM设置:{expected_qdm}, QDM实际使用:{actual_qdm}, 匹配不上"})
                    else:
                        return Pass("QDM检验完成", PassType.QDM_PASS, {'qdm_id': self.qdm_event_id, 'msg': f"QDM设置:{expected_qdm}, QDM实际使用:{actual_qdm}, 检验完成"})
                else:
                    return Issue("QDM系数查询不到", IssueType.QDM_NOT_EXIST, {'qdm_id': self.qdm_event_id, 'msg': "无对应QDM配置"})
        return None
        
    def _run_weight_factor_check(self, segments):
        # print(f"{self.position} -> check weight")
        first = segments[0]
        # print(f"{self.position} -> {first}")
        if not segments or not self.material:
            return None
        if not self.dev_ips:
            expected_weight = 500
            if expected_weight != first['cur_weight']:
                return Issue("克重不匹配", IssueType.WEIGHT_DISMATCH, {'weight_id': self.weight_event_id, 'msg': f"克重设置:{expected_weight}, QDM实际使用:{first['cur_weight']}, 匹配不上"})
            else:
                return Pass("克重匹配", PassType.WEIGHT_PASS, {'weight_id': self.weight_event_id, 'msg': f"克重设置:{expected_weight}, QDM实际使用:{first['cur_weight']}, 匹配完成"})
        else:
             # 判断材质格式
            if '/' in self.material:
                codes = self.material.split('/')          # "8/J" → ["8","J"]
            elif '.' in self.material:
                parts = self.material.split('.')
                pos_targets = {'GU1': (0, 1), 'GU2': (2, 3), 'GU3': (4, 5)}
                start, end = pos_targets.get(self.position, (-1, -1))
                if start < 0 or start >= len(parts) or parts[start] == '-':
                    codes = []
                elif end < len(parts) and parts[end] != '-':
                    codes = [parts[start], parts[end]]
                elif end >= len(parts):
                    codes = []
                else:
                    next_i = end + 1
                    while next_i < len(parts) and parts[next_i] == '-':
                        next_i += 1
                    codes = [parts[start], parts[next_i]] if next_i < len(parts) else []
            else:
                return None
            if not codes:
                return None
        
            total = 0.0
            for pc in codes:
                rows = self._query_ips(
                    'SELECT "SPC_GlueWeight" FROM "S_PaperCodes" WHERE "SPC_Code" = %s',
                    (pc,)
                )
                if not rows or not rows[0][0]:
                    return None
                total += float(rows[0][0])
                
            actual_w = first.get('cur_weight')
            if actual_w is not None:
                actual_w = float(actual_w)
                if abs(actual_w - total) > 1:
                    return Issue("克重不匹配", IssueType.WEIGHT_DISMATCH, {"weight_id": self.weight_event_id, "msg": f'克重={actual_w:.0f}g, 档案={total:.0f}g, 差异={actual_w-total:.0f}g'})
                else:
                    return Pass("克重匹配", PassType.WEIGHT_PASS, {'weight_id': self.weight_event_id, 'msg': f'克重={actual_w}g, 档案:{total}g, 匹配完成'})
            else:
                return Issue("克重查询不到", IssueType.WEIGHT_NOT_EXIST, {'weight_id': self.weight_event_id, 'msg': "克重设置不存在"})

    def _run_base_setting_check(self, segments):
        first = segments[0]
        if not segments or not self.flute:
            return
        if not self.dev_ips:
            return

        if self.position.startswith('GU'):
            pos = {'GU1': '1', 'GU2': '2', 'GU3': '3'}.get(self.position, '')
            rows = self._query_ips(
                'SELECT "F_MinGlue", "F_MaxGlue", "F_MinWeight", "F_MaxWeight" '
                'FROM "TB_IPS_GlueGu" WHERE "F_Flute" = %s AND "F_Position" = %s',
                (self.flute, pos)
            )
            if not rows and pos != '1':
                rows = self._query_ips(
                    'SELECT "F_MinGlue", "F_MaxGlue", "F_MinWeight", "F_MaxWeight" '
                    'FROM "TB_IPS_GlueGu" WHERE "F_Flute" = %s AND "F_Position" = %s',
                    (self.flute, '1')
                )
            if not rows:
                rows = self._query_ips(
                    'SELECT "F_MinGlue", "F_MaxGlue", "F_MinWeight", "F_MaxWeight" '
                    'FROM "TB_IPS_GlueGu" WHERE "F_Flute" = %s LIMIT 1',
                    (self.flute,)
                )
        elif self.position.startswith('SF'):
            rows = self._query_ips(
                'SELECT "F_MinGlue", "F_MaxGlue", "F_MinWeight", "F_MaxWeight" '
                'FROM "TB_IPS_GlueSF" WHERE "F_Flute" = %s',
                (self.flute,)
            )
        else:
            return

        if not rows:
            return Issue("基础资料缺失", IssueType.BASEDOC_NOT_EXIST, {'basedoc_id': self.basedoc_event_id, 'msg': f"基础资料缺失"})

        db_min_g, db_max_g, db_min_w, db_max_w = [float(v) if v else 0 for v in rows[0]]
        field_map = [
            ('min_glue', first['min_glue'], db_min_g),
            ('max_glue', first['max_glue'], db_max_g),
            ('min_weight', first['min_weight'], db_min_w),
            ('max_weight', first['max_weight'], db_max_w),
        ]
        for field_name, actual, expected in field_map:
            if abs(float(actual) - expected) > 0:
                return Issue(f"基础设置不匹配({field_name})", IssueType.BASEDOC_DISMATCH,
                            {'basedoc_id': self.basedoc_event_id, 'msg': f"实际: {float(actual)}, 预期: {expected}"})
            else:
                return Pass(f"基础设置匹配({field_name})", PassType.BASEDOC_PASS,
                            {'basedoc_id': self.basedoc_event_id, 'msg': f"实际: {float(actual)}, 预期: {expected}"})
        return None
    
    def _run_speed_coef_check(self, segments):
        if not segments or not self.flute:
            return
        if not self.dev_ips:
            return

        pos_map = {'GU1': 0, 'GU2': 1, 'GU3': 2, 'SF1': 3, 'SF2': 4, 'SF3': 5}
        pos = pos_map.get(self.position)
        if pos is None:
            return

        rows = self._query_ips(
            'SELECT "F_Speed", "F_Coef" FROM "TB_IPS_GlueSpeedCoef" '
            'WHERE "F_Position" = %s ORDER BY "F_Speed"',
            (pos,)
        )
        if not rows:
            return Issue("车速系数资料缺失", IssueType.SPEED_COEF_NOT_EXIST,
                        {'speed_coef_id': self.speed_coef_id, 'msg': f'{self.position}(position={pos})无配置'})

        db = {int(r[0]): float(r[1]) for r in rows if r[0] and r[1]}
        first = segments[0]
        speed = int(float(first['speed']))
        actual = float(first['speed_factor'])
        expected = db.get(speed)
        if expected is not None:
            if abs(actual - expected) > 0:
                return Issue("车速系数不匹配", IssueType.SPEED_COEF_DISMATCH,
                            {'speed_coef_id': self.speed_coef_id, 'msg': f"实际: {actual}, 预期: {expected}"})
            else:
                return Pass("车速系数匹配", PassType.SPEED_PASS,
                            {'speed_coef_id': self.speed_coef_id, 'msg': f"实际: {actual}, 预期: {expected}"})
        return Issue("车速系数不存在", IssueType.SPEED_COEF_NOT_EXIST,
                        {'speed_coef_id': self.speed_coef_id, 'msg': f"车速系数不存在"})
    # ── 查询辅助 ──

    def _query_ips(self, sql, params=None):
        if not self.dev_ips:
            return None
        try:
            cur = self.dev_ips.conn.cursor()
            cur.execute(sql, params or ())
            return cur.fetchall()
        except Exception:
            self.dev_ips.conn.rollback()
            return None


# ── Orchestrator ──

class GlueGapDiagnosticFSM:
    """FSM 编排器：接收事件流，分发给各 PositionFSM。"""

    def __init__(self, extractor: GlueEventExtractor, warp_extractor=None, dev_ips=None):
        self.extractor = extractor
        self.warp_extractor = warp_extractor
        self.dev_ips = dev_ips
        self.raw_events = extractor.raw_parsed_rows
        self.raw_events.sort(key=lambda x: str(x.get('Date', '')))
        self.set_func_events = extractor.get_glue_set_function_full_event()
        self.all_events = extractor.get_all_events()
        self.all_material_events = []
        
        # 加入一些可选的组件，比如用于qdm和克重数据拿取的数据库    
        self.build_components()
        
        self.fsms: Dict[str, PositionFSM] = {}
        for pos in _ALL_POSITIONS:
            self.fsms[pos] = PositionFSM(pos, self.dev_ips)
        
    def build_components(self):
        if self.dev_ips is None:
            from database_utils import PostgreSQLHelper
            try:
                self.dev_ips = PostgreSQLHelper.from_connection_string(
                    "PORT=5433;DATABASE=HNHY_IPS;HOST=127.0.0.1;PASSWORD=123456;USER ID=postgres"
                )
                self.dev_ips.connect()
            except Exception:
                self.dev_ips = None

    # ── 查询辅助 ──

    def _query_ips(self, sql, params=None):
        if not self.dev_ips:
            return None
        try:
            cur = self.dev_ips.conn.cursor()
            cur.execute(sql, params or ())
            return cur.fetchall()
        except Exception:
            self.dev_ips.conn.rollback()
            return None

    # ── 运行 ──

    def run(self):
        for evt in self.all_events:
            # print(evt)
            type = evt.get('type', '')
            part = evt.get('part', '')
            if type == 'material':
                self.all_material_events.append(evt) # 只用于后续输出
                for fsm in self.fsms.values():
                    fsm.process_event(evt)
            elif type == 'glue':
                self.fsms[part].process_event(evt)
                
    def get_results(self):
        results = {}
        fsm_events = {}
        for pos in _ALL_POSITIONS:
            fsm_events[pos] = self.fsms[pos].full_events

        # 转换 Issue 对象为字典，避免 JSON 序列化时变成字符串
        for events in fsm_events.values():
            for evt in events:
                if 'errors' in evt:
                    evt['errors'] = [e.to_dict() if isinstance(e, Issue) else e for e in evt['errors']]
                if 'warnings' in evt:
                    evt['warnings'] = [e.to_dict() if isinstance(e, Warning) else e for e in evt['warnings']]
                if 'passes' in evt:
                    evt['passes'] = [e.to_dict() if isinstance(e, Pass) else e for e in evt['passes']]

        results['glue_events'] = fsm_events
        results['material_events'] = self.all_material_events
        # ── 查询数据库参考表写入 results ──
        results['qdm_df'] = []
        results['qdm_sf'] = []
        results['basedoc_gu'] = []
        results['basedoc_sf'] = []
        results['speed_coef'] = []
        results['paper_codes'] = []

        if self.dev_ips:
            try:
                rows = self._query_ips('SELECT "F_Paper", "F_Flute", "F_Glue1", "F_Glue2", "F_Glue3" FROM "TB_IPS_QdmCoefDF"')
                if rows:
                    results['qdm_df'] = [{'paper': r[0], 'flute': r[1], 'glue1': r[2], 'glue2': r[3], 'glue3': r[4]} for r in rows]
            except Exception:
                pass
            try:
                rows = self._query_ips('SELECT "F_MS", "F_LS", "F_Flute", "F_Glue" FROM "TB_IPS_QdmCoefSF"')
                if rows:
                    results['qdm_sf'] = [{'ms': r[0], 'ls': r[1], 'flute': r[2], 'glue': r[3]} for r in rows]
            except Exception:
                pass
            try:
                rows = self._query_ips('SELECT "F_Flute", "F_Position", "F_MinGlue", "F_MaxGlue", "F_MinWeight", "F_MaxWeight" FROM "TB_IPS_GlueGu"')
                if rows:
                    results['basedoc_gu'] = [{'flute': r[0], 'position': r[1], 'min_glue': r[2], 'max_glue': r[3], 'min_weight': r[4], 'max_weight': r[5]} for r in rows]
            except Exception:
                pass
            try:
                rows = self._query_ips('SELECT "F_Flute", "F_MinGlue", "F_MaxGlue", "F_MinWeight", "F_MaxWeight" FROM "TB_IPS_GlueSF"')
                if rows:
                    results['basedoc_sf'] = [{'flute': r[0], 'min_glue': r[1], 'max_glue': r[2], 'min_weight': r[3], 'max_weight': r[4]} for r in rows]
            except Exception:
                pass
            try:
                rows = self._query_ips('SELECT "F_Position", "F_Speed", "F_Coef" FROM "TB_IPS_GlueSpeedCoef" ORDER BY "F_Position", "F_Speed"')
                if rows:
                    results['speed_coef'] = [{'position': r[0], 'speed': r[1], 'coef': r[2]} for r in rows]
            except Exception:
                pass
            try:
                rows = self._query_ips('SELECT "SPC_Code", "SPC_GlueWeight" FROM "S_PaperCodes"')
                if rows:
                    results['paper_codes'] = [{'code': r[0], 'weight': r[1]} for r in rows]
            except Exception:
                pass
        
        results['description'] = "glue_events中保存的是已经成功的胶水赋值事件，其中errors中是已经确认的错误，是最需要注意的部分。material_events是这个时间段内的所有换材事件，如果出现材质不匹配的错误，可以使用id在这里匹配对应的换材事件。qdm_df中保存的是GU1，GU2，GU3使用的用于取qdm_factor的数据表。qdm_sf中保存的是SF1，SF2，SF3在胶水赋值时取qdm_factor的数据表。basedoc_gu保存的是GU1，GU2，GU3在胶水赋值时取基础楞型的数据表。basedoc_sf保存的是SF1，SF2，SF3在胶水赋值时取基础楞型的数据表。speed_coef中保存的是取车速系数的数据表。paper_codes中保存的是取克重的数据表。"
        return results

    def save_results(self, filepath=None):
        import json, os, uuid
        results = self.get_results()
        # print(results['glue_events'])
        if filepath is None:
            filepath = os.path.join(os.path.dirname(__file__), "environments", "fsm_results.json")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, cls=CustomJsonEncoder)
        print(f"FSM 结果已保存到 {filepath}")
        return filepath