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
    BASEDOC_DISMATCH = 'basedoc_dismatch'
    NO_SET_VALUES = 'no_set_values'

# ── 问题记录 ──

@dataclass
class Issue:
    detail: str
    type: IssueType
    args: object

@dataclass
class AnalysisIssue:
    type: str
    detail: str
    cycle_index: int
    layer: str = ''
    severity: str = 'error'


# ── 状态转移记录 ──

@dataclass
class TransitionStep:
    time: str
    event_id: str
    from_state: str
    to_state: str
    detail: str = ''


# ── 周期记录 ──

@dataclass
class CycleRecord:
    index: int
    position: str
    start_time: str
    end_time: str
    trigger_id: str
    material: str
    flute: str
    end_status: str
    computed_segments: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    infos: list = field(default_factory=list)
    lifecycle: dict = field(default_factory=dict)
    transitions: list = field(default_factory=list)


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
        
        self.material = ''
        self.computed_segments = []
        self.dev_ips = dev_ips


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
                'time': str(row['Date'])
            }
        """
        
        errors = []
        self.material = event['material']
        
        # apply some check
        # 1. material, if dismatch, return error and material change id
        if self.position.startswith("SF"):
            ms_material = self.material_event[f"ms{self.position[2]}"]['material']
            ls_material = self.material_event[f"ls{self.position[2]}"]['material']
            if len(event['material'].split('/')) == 1:
                return
            if event['material'].split('/')[0] != ms_material:
                errors.append(Issue("材质匹配失败", IssueType.MATERIAL_DISMATCH, {'id': self.material_event[f"ms{self.position[2]}"]['id']}))
            if event['material'].split('/')[1] != ls_material:
                errors.append(Issue("材质匹配失败", IssueType.MATERIAL_DISMATCH, {'id': self.material_event[f"ls{self.position[2]}"]['id']}))    
        else:
            if event['material'] != self.material_event['df']['material']:
                errors.append(Issue("材质匹配失败", IssueType.MATERIAL_DISMATCH, {'id': self.material_event['df']['id']}))
        
        # 2. no set_values
        if event['set_values'] == {}:
            errors.append(Issue("无计算结果", IssueType.NO_SET_VALUES, {}))
            return errors
        # 2. qdm
        segments = self._parse_segments(event['set_values'])
        self._run_qdm_check(segments)
        # 3. base doc
        print(errors)
        return errors

    def _parse_segments(self, pv):
        # ['speed', 'min_glue', 'max_glue', 'min_weight', 'max_weight', 'current_glue_weight', 'speed_factor', 'min_speed', 'qdm_factor', 'ui_factor', 'warp_offset', 'value']
        segments = []
        data = pv.get('data')
        # print(data)
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
        print(f"segs: {segments}")
        return segments

    # ── 校验方法（由 _run_checks 统一调用） ──

    def _run_qdm_check(self, segments):
        print("check qdm")
        if not self.dev_ips:
            return
        if not segments or not self.material:
            return
        first = segments[0]
        print(first)
        # QDM 校验
        if '/' in self.material and self.position.startswith('SF'):
            ms, ls = self.material.split('/')
            rows = self._query_ips('SELECT "F_Glue" FROM "TB_IPS_QdmCoefSF" WHERE "F_MS" = %s AND "F_LS" = %s AND "F_Flute" = %s', (ms, ls, self.flute))
            if rows and rows[0][0]:
                expected_qdm = float(rows[0][0])
                actual_qdm = first.get('qdm_factor')
                if actual_qdm is not None:
                    try:
                        if abs(float(actual_qdm) - expected_qdm) > 0:
                            self.issues.append(AnalysisIssue('qdm_mismatch', f'QDM系数={actual_qdm}, 数据库={expected_qdm}', self.cycle_index, self.position, 'error'))
                    except (ValueError, TypeError):
                        pass
            else:
                self.issues.append(AnalysisIssue('qdm_no_data', f'材质"{ms}/{ls}"+楞型"{self.flute}"在SF QDM中无条目', self.cycle_index, self.position, 'error'))

    def _run_checks(self):
        self._check_cross_source()

    def _check_cross_source(self):
        if not self.dev_ips:
            return
        if not self.computed_segments or not self.material:
            return
        first = self.computed_segments[0]

        # 克重校验
        parts = [p for p in self.material.split('.') if p != '-'] if '.' in self.material else []
        if not parts and '/' in self.material:
            parts = []
        if parts:
            total = 0.0
            all_found = True
            for pc in parts:
                rows = self._query_ips('SELECT "SPC_GlueWeight" FROM "S_PaperCodes" WHERE "SPC_Code" = %s', (pc,))
                if rows and rows[0][0]:
                    total += float(rows[0][0])
                else:
                    all_found = False
                    break
            if all_found:
                actual_w = first.get('cur_weight')
                if actual_w is not None:
                    try:
                        actual_w = float(actual_w)
                        if abs(actual_w - total) > 1:
                            self.issues.append(AnalysisIssue('weight_mismatch', f'克重={actual_w:.0f}g, 档案={total:.0f}g, 差异={actual_w-total:.0f}g', self.cycle_index, self.position, 'error'))
                    except (ValueError, TypeError):
                        pass

        

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
        
        self.fsms: Dict[str, PositionFSM] = {}
        for pos in _ALL_POSITIONS:
            self.fsms[pos] = PositionFSM(pos, dev_ips)

    # ── 运行 ──

    def run(self):
        for evt in self.all_events:
            print(evt)
            type = evt.get('type', '')
            part = evt.get('part', '')
            if type == 'material':
                for fsm in self.fsms.values():
                    fsm.process_event(evt)
            elif type == 'glue':
                self.fsms[part].process_event(evt)
