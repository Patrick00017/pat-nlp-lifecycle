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


# ── 问题记录 ──

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
    """单个糊间隙位置（GU1/GU2/...）的状态机。"""

    def __init__(self, position: str, dev_ips=None):
        self.position = position
        self.state = GlueState.IDLE
        self.material = ''
        self.flute = ''
        self.start_time = ''
        self.end_time = ''
        self.trigger_id = ''
        self.computed_segments = []
        self.issues: List[AnalysisIssue] = []
        self.cycle_index = -1
        self._cycle_counter = -1
        self.lifecycle = {}
        self.dev_ips = dev_ips
        self.last_end_status = ''
        self._transitions: List[TransitionStep] = []

    # ── 事件分发 ──

    def process_event(self, event: dict) -> bool:
        eid = event.get('EventId')
        handler = getattr(self, f'_on_{eid}', None)
        if handler:
            handler(event)
            return True
        return False

    # ── 转移记录 ──

    def _record_transition(self, event=None, to_state=None, detail=''):
        eid = event.get('EventId', '') if event else 'check'
        ts = str(event.get('Date', ''))[:19] if event else ''
        from_s = self._transitions[-1].to_state if self._transitions else 'init'
        to_s = to_state.value if to_state else self.state.value
        self._transitions.append(TransitionStep(
            time=ts, event_id=eid,
            from_state=from_s, to_state=to_s,
            detail=detail,
        ))

    # ── G7 / G11: 开始新周期 ──

    def _on_G7(self, event):
        self._open_cycle(event, 'G7')

    def _on_G11(self, event):
        self._open_cycle(event, 'G11')

    def _open_cycle(self, event, trigger_id):
        if self.state != GlueState.IDLE:
            self._close_cycle('interrupted')
        self._cycle_counter += 1
        self.cycle_index = self._cycle_counter
        self.state = GlueState.MATERIAL_CHANGED
        self.trigger_id = trigger_id
        self.start_time = str(event.get('Date', ''))
        pv = event.get('ParsedValues', {})
        self.material = pv.get('material', pv.get('meterial', ''))
        self.flute = pv.get('flute_type', '')
        self.computed_segments = []
        self.issues = []
        self.lifecycle = {}
        self._transitions = []
        self._record_transition(event, GlueState.MATERIAL_CHANGED)

    # ── G14 / G4: 计算值到达 + 校验 ──

    def _on_G14(self, event):
        self._handle_calculation(event)

    def _on_G4(self, event):
        self._handle_calculation(event)

    def _handle_calculation(self, event):
        if self.state not in (GlueState.MATERIAL_CHANGED, GlueState.G10_FALLBACK, GlueState.CALCULATED):
            return
        pv = event.get('ParsedValues', {})
        self._parse_segments(pv)
        self.state = GlueState.CALCULATED
        self._record_transition(event, GlueState.CALCULATED)
        self._run_checks()

    def _parse_segments(self, pv):
        segments = []
        for i in range(1, 9):
            speed = pv.get(f'speed{i}')
            value = pv.get(f'value{i}')
            min_g = pv.get(f'min_glue{i}')
            max_g = pv.get(f'max_glue{i}')
            min_w = pv.get(f'min_weight{i}')
            max_w = pv.get(f'max_weight{i}')
            cur_w = pv.get(f'current_glue_weight{i}')
            sf = pv.get(f'speed_factor{i}')
            qdm = pv.get(f'qdm_factor{i}')
            ui = pv.get(f'ui_factor{i}')
            off = pv.get(f'offset{i}')
            if speed is not None:
                segments.append({
                    'speed': speed, 'value': value,
                    'min_glue': min_g, 'max_glue': max_g,
                    'min_weight': min_w, 'max_weight': max_w,
                    'cur_weight': cur_w, 'speed_factor': sf,
                    'qdm_factor': qdm, 'ui_factor': ui,
                    'offset': off,
                })
        self.computed_segments = segments

    # ── 其他事件 ──

    def _on_G10(self, event):
        if self.state in (GlueState.MATERIAL_CHANGED,):
            self.state = GlueState.G10_FALLBACK
            self._record_transition(event, GlueState.G10_FALLBACK)

    def _on_G8(self, event):
        if self.state in (GlueState.MATERIAL_CHANGED, GlueState.G10_FALLBACK):
            self.issues.append(AnalysisIssue('cancelled_delay', '延迟中被抢断', self.cycle_index, self.position, 'info'))
            self._record_transition(event, detail='cancelled_delay')

    def _on_G15(self, event):
        if self.state == GlueState.CALCULATED:
            self._record_transition(event, GlueState.CANCELLED_PRE_WRITE)
            self._close_cycle('cancelled_pre_write')

    def _on_G12(self, event):
        self._handle_complete(event)

    def _on_G5(self, event):
        self._handle_complete(event)

    def _handle_complete(self, event):
        if self.state == GlueState.CALCULATED:
            self.end_time = str(event.get('Date', ''))
            self._record_transition(event, GlueState.COMPLETE)
            self._close_cycle('complete')

    def _on_G3(self, event):
        if self.state in (GlueState.MATERIAL_CHANGED, GlueState.DELAY_WAITING, GlueState.G10_FALLBACK):
            self._record_transition(event, detail='task_cancelled')
            self._close_cycle('interrupted')

    # ── 周期管理 ──

    def _close_cycle(self, end_status):
        self.last_end_status = end_status
        self.end_time = self.end_time or self.start_time
        self.state = GlueState.IDLE

    def get_cycle(self) -> Optional[CycleRecord]:
        return CycleRecord(
            index=self.cycle_index,
            position=self.position,
            start_time=self.start_time,
            end_time=self.end_time,
            trigger_id=self.trigger_id,
            material=self.material,
            flute=self.flute,
            end_status=self.last_end_status or 'interrupted',
            computed_segments=self.computed_segments,
            errors=[i for i in self.issues if i.severity == 'error'],
            warnings=[i for i in self.issues if i.severity == 'warning'],
            infos=[i for i in self.issues if i.severity == 'info'],
            lifecycle=self.lifecycle,
            transitions=self._transitions,
        )

    # ── 校验方法（由 _run_checks 统一调用） ──

    def _run_checks(self):
        self._check_speed_monotonic()
        self._check_value_range()
        self._check_cross_source()

    def _check_speed_monotonic(self):
        speeds = [s['speed'] for s in self.computed_segments if s.get('speed') is not None]
        for i in range(1, len(speeds)):
            try:
                if float(speeds[i]) <= float(speeds[i - 1]):
                    self.issues.append(AnalysisIssue(
                        'speed_not_monotonic', f'车速段{i}不单调: {speeds[i-1]}->{speeds[i]}',
                        self.cycle_index, self.position, 'warning'
                    ))
            except (ValueError, TypeError):
                pass

    def _check_value_range(self):
        values = []
        for s in self.computed_segments:
            try:
                v = float(s['value']) if s.get('value') is not None else None
                if v is not None:
                    values.append(v)
            except (ValueError, TypeError):
                pass
        if values:
            if min(values) < 0:
                self.issues.append(AnalysisIssue('negative_value', f'负值={min(values)}', self.cycle_index, self.position, 'error'))
            if max(values) > 60:
                self.issues.append(AnalysisIssue('exceeds_hard_limit', f'超60={max(values)}', self.cycle_index, self.position, 'warning'))
            for i in range(1, len(values)):
                jump = abs(values[i] - values[i - 1])
                if jump > 2.0:
                    self.issues.append(AnalysisIssue('value_jump', f'段{i}跳变:{values[i-1]}->{values[i]}({jump:.2f})', self.cycle_index, self.position, 'warning'))

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

    def __init__(self, extractor, warp_extractor=None, dev_ips=None):
        self.extractor = extractor
        self.warp_extractor = warp_extractor
        self.dev_ips = dev_ips
        self.raw_events = extractor.raw_parsed_rows
        self.raw_events.sort(key=lambda x: str(x.get('Date', '')))
        self.set_func_events = extractor.get_glue_set_function_full_event()

        self.fsms: Dict[str, PositionFSM] = {}
        for pos in _ALL_POSITIONS:
            self.fsms[pos] = PositionFSM(pos, dev_ips)

        self.records: List[CycleRecord] = []

    # ── 运行 ──

    def run(self):
        # 建立 set_func_event 索引: time -> event
        sfe_by_time = {}
        for sfe in self.set_func_events:
            t = str(sfe.get('time', ''))
            if t:
                sfe_by_time[t] = sfe

        for evt in self.raw_events:
            eid = evt.get('EventId')
            pv = evt.get('ParsedValues', {})
            date_str = str(evt.get('Date', ''))

            if eid in ('G7', 'G11'):
                targets = self._resolve_targets(evt, pv, sfe_by_time)
                for pos in targets:
                    self.fsms[pos].process_event(evt)

            elif eid == 'G14':
                gp = pv.get('glue_part', '')
                if gp in self.fsms:
                    self.fsms[gp].process_event(evt)

            elif eid == 'G4':
                gp = pv.get('glue_part', '')
                if gp in self.fsms:
                    self.fsms[gp].process_event(evt)

            elif eid in ('G12', 'G5'):
                # 分发给当前在 CALCULATED 状态的位置
                for pos, fsm in self.fsms.items():
                    if fsm.state == GlueState.CALCULATED:
                        fsm.process_event(evt)
                        # 匹配 set_func_event 补充 lifecycle/material
                        if date_str in sfe_by_time:
                            sfe = sfe_by_time[date_str]
                            fsm.material = fsm.material or sfe.get('material', '')
                            fsm.flute = fsm.flute or sfe.get('flute_type', '')
                            fsm.lifecycle = sfe.get('lifecycle', {})

            elif eid == 'G15':
                for fsm in self.fsms.values():
                    if fsm.state == GlueState.CALCULATED:
                        fsm.process_event(evt)

            elif eid in ('G8', 'G10', 'G3'):
                for fsm in self.fsms.values():
                    if fsm.state in (GlueState.MATERIAL_CHANGED, GlueState.G10_FALLBACK, GlueState.DELAY_WAITING):
                        fsm.process_event(evt)

    # ── 目标位置解析 ──

    def _resolve_targets(self, evt, pv, sfe_by_time):
        eid = evt.get('EventId')
        date_str = str(evt.get('Date', ''))

        # G11: 可能含 glue_part
        if eid == 'G11':
            gp = pv.get('glue_part', '')
            if gp == 'GU':
                return list(_POSITION_GU)
            if gp in self.fsms:
                return [gp]
            # 回退：从 set_func_event 推断
            sfe = sfe_by_time.get(date_str)
            if sfe:
                part = sfe.get('part', '')
                return self._part_to_positions(part)
            return []

        # G7: 从 set_func_name 推断
        func = pv.get('set_func_name', '')
        if 'SetGlueGu' in func:
            return list(_POSITION_GU)
        if func == 'SetGlueSF1':
            return ['SF1']
        if func == 'SetGlueSF2':
            return ['SF2']
        # 回退 set_func_events
        sfe = sfe_by_time.get(date_str)
        if sfe:
            part = sfe.get('part', '')
            return self._part_to_positions(part)
        return []

    @staticmethod
    def _part_to_positions(part):
        if part == 'DF':
            return list(_POSITION_GU)
        if part in _ALL_POSITIONS:
            return [part]
        if part in ('SF1', 'SF2', 'SF3'):
            return [part]
        return []

    # ── 报告生成 ──

    def generate_json(self, target_time=None, expected_values=None):
        # 收集所有 FSM 的 issues
        all_issues = []
        for fsm in self.fsms.values():
            all_issues.extend(fsm.issues)

        cycles_data = []
        for fsm in self.fsms.values():
            if fsm.cycle_index < 0:
                continue
            segs = fsm.computed_segments
            computed = {}
            if segs:
                computed[fsm.position] = [
                    {'speed': s['speed'], 'value': s['value']}
                    for s in segs if s.get('speed') is not None
                ]

            errors = [{'type': i.type, 'label': i.type, 'detail': i.detail} for i in fsm.issues if i.severity == 'error']
            warnings = list(set(i.type for i in fsm.issues if i.severity == 'warning'))
            infos = [i.type for i in fsm.issues if i.severity == 'info']

            trig_labels = {'G7': '换材触发', 'G11': '立即换材'}
            status_labels = {'complete': '完成', 'cancelled_pre_write': '写值取消', 'interrupted': '中断'}

            cycles_data.append({
                'index': fsm.cycle_index,
                'position': fsm.position,
                'status': {'id': fsm.last_end_status or fsm.state.value, 'label': status_labels.get(fsm.last_end_status or fsm.state.value, fsm.last_end_status or fsm.state.value)},
                'trigger': {'id': fsm.trigger_id, 'label': f'{fsm.trigger_id}（{trig_labels.get(fsm.trigger_id, "其他触发")}）'},
                'material': fsm.material,
                'flute': fsm.flute,
                'start_time': fsm.start_time,
                'computed_values': computed,
                'errors': errors,
                'warnings': warnings,
                'infos': infos,
            })

        return {
            'cycles': cycles_data,
            'summary': {'total_cycles': len(cycles_data), 'total_issues': len(all_issues)},
        }

    def generate_report(self, target_time=None, expected_values=None):
        data = self.generate_json(target_time, expected_values)
        lines = ['# 糊间隙诊断报告（FSM 引擎）', '']
        cycles = data.get('cycles', [])
        if not cycles:
            lines.append('无数据')
            return '\n'.join(lines)

        lines.append(f'共 {len(cycles)} 个周期，{data["summary"]["total_issues"]} 个问题')
        lines.append('')

        for c in cycles:
            lines.append(f'### {c["position"]} 周期 #{c["index"]} ({c["status"]["label"]})')
            lines.append(f'- **触发**: {c["trigger"]["label"]}')
            lines.append(f'- **材质**: {c["material"]}')
            if c.get('computed_values'):
                for pos, segs in c['computed_values'].items():
                    vals = ' / '.join(f"@{s['speed']}={s['value']}" for s in segs)
                    lines.append(f'- **计算值**: {pos}: {vals}')
            for err in c.get('errors', []):
                lines.append(f'- ❌ **{err["label"]}**：{err["detail"]}')
            for w in set(c.get('warnings', [])):
                lines.append(f'- ⚠ **{w}**')
            for info in c.get('infos', []):
                lines.append(f'- ℹ {info}')
            lines.append('')

        return '\n'.join(lines)
