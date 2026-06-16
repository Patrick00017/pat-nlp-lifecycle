import pandas as pd
from datetime import datetime
from collections import defaultdict


class GlueGapDiagnostic:
    def __init__(self, extractor, warp_extractor=None):
        self.extractor = extractor
        self.warp_extractor = warp_extractor
        self.raw_events = extractor.raw_parsed_rows
        self.raw_events.sort(key=lambda x: str(x.get('Date', '')))
        self.set_func_events = extractor.get_glue_set_function_full_event()
        self.cycles = self._group_cycles()

    def _to_ts(self, val):
        if isinstance(val, pd.Timestamp):
            return val
        if isinstance(val, str):
            try:
                return pd.Timestamp(val)
            except Exception:
                return val
        return val

    def _group_cycles(self):
        cycles = []
        current = None
        for evt in self.raw_events:
            eid = evt.get('EventId')
            if eid in ('G7', 'G11'):
                if current:
                    if current.get('end') is None:
                        current['end'] = 'interrupted'
                    cycles.append(current)
                current = {'start': evt, 'events': [], 'end': None, 'end_event': None}
                current['events'].append(evt)
            elif current is None:
                continue
            else:
                current['events'].append(evt)
                if eid == 'G12':
                    current['end'] = 'complete'
                    current['end_event'] = evt
                    g12_time = str(evt.get('Date', ''))
                    for sfe in self.set_func_events:
                        if str(sfe.get('time', '')) == g12_time:
                            current['set_func_event'] = sfe
                            break
                    cycles.append(current)
                    current = None
                elif eid == 'G15':
                    current['end'] = 'cancelled_pre_write'
                    current['end_event'] = evt
                    cycles.append(current)
                    current = None
        if current:
            if current.get('end') is None:
                current['end'] = 'interrupted'
            cycles.append(current)
        for i, c in enumerate(cycles):
            c['index'] = i
        return cycles

    # ── Dimension 1: Cycle Completeness ──
    def check_cycle_completeness(self):
        anomalies = []
        for c in self.cycles:
            g14_count = sum(1 for e in c['events'] if e.get('EventId') == 'G14')
            has_g8 = any(e.get('EventId') == 'G8' for e in c['events'])
            has_g10 = any(e.get('EventId') == 'G10' for e in c['events'])
            start_type = c['start'].get('EventId', '')
            start_time = c['start'].get('Date', '')

            if c['end'] == 'interrupted':
                anomalies.append({
                    'type': 'no_termination',
                    'cycle_index': c['index'],
                    'start_time': str(start_time),
                    'start_type': start_type,
                    'detail': f'{start_type}触发的周期({start_time})缺少G12或G15结束事件'
                })
            elif c['end'] == 'cancelled_pre_write' and g14_count == 0:
                anomalies.append({
                    'type': 'pre_write_cancel_no_calc',
                    'cycle_index': c['index'],
                    'start_time': str(start_time),
                    'detail': '写值前被取消(G15)但没有G14计算事件 —— 计算可能失败'
                })
            elif c['end'] == 'complete' and g14_count == 0:
                anomalies.append({
                    'type': 'g12_no_g14',
                    'cycle_index': c['index'],
                    'start_time': str(start_time),
                    'detail': 'G12写值完成但未发现G14计算事件'
                })
            elif c['end'] == 'complete' and has_g10:
                anomalies.append({
                    'type': 'fallback_used',
                    'cycle_index': c['index'],
                    'start_time': str(start_time),
                    'detail': 'G10降级匹配被触发 —— 材质与设备部位映射不匹配'
                })
            elif g14_count > 3:
                anomalies.append({
                    'type': 'excessive_calculation',
                    'cycle_index': c['index'],
                    'start_time': str(start_time),
                    'detail': f'单个周期内有{g14_count}次G14计算 —— 重复计算过多'
                })
        return anomalies

    # ── Dimension 2: Cancellation Rate ──
    def calc_cancellation_rate(self):
        total_g7 = sum(1 for c in self.cycles if c['start'].get('EventId') == 'G7')
        total_g11 = sum(1 for c in self.cycles if c['start'].get('EventId') == 'G11')
        total_cycles = len(self.cycles)
        g8_count = sum(1 for c in self.cycles
                       for e in c['events'] if e.get('EventId') == 'G8')
        g15_count = sum(1 for c in self.cycles if c['end'] == 'cancelled_pre_write')
        completed = sum(1 for c in self.cycles if c['end'] == 'complete')
        interrupted = sum(1 for c in self.cycles if c['end'] == 'interrupted')

        return {
            'total_cycles': total_cycles,
            'g7_starts': total_g7,
            'g11_starts': total_g11,
            'completed': completed,
            'cancelled_pre_write': g15_count,
            'cancelled_delay': g8_count,
            'interrupted': interrupted,
            'cancellation_rate': round((g8_count + g15_count) / max(total_cycles, 1) * 100, 1),
            'alert': (g8_count + g15_count) / max(total_cycles, 1) > 0.3
        }

    # ── Dimension 3: Value Plausibility ──
    def check_value_plausibility(self, layer='GU1'):
        issues = []
        for c in self.cycles:
            if c['end'] != 'complete':
                continue
            sfe = c.get('set_func_event')
            if not sfe:
                continue
            layer_data = sfe.get('set_values', {}).get(layer)
            if not layer_data:
                continue
            rows = layer_data.get('data', [])
            if not rows:
                continue
            speeds = []
            values = []
            for r in rows:
                try:
                    speeds.append(float(r[0]) if r[0] else 0)
                    values.append(float(r[-1]) if r[-1] else 0)
                except (ValueError, TypeError):
                    continue
            if len(speeds) < 2:
                continue
            start_time = c['start'].get('Date', '')
            for i in range(1, len(speeds)):
                if speeds[i] <= speeds[i - 1]:
                    issues.append({
                        'type': 'speed_not_monotonic',
                        'cycle_index': c['index'],
                        'time': str(start_time),
                        'layer': layer,
                        'detail': f'车速段{i}不单调递增: {speeds[i-1]} -> {speeds[i]}'
                    })
                jump = abs(values[i] - values[i - 1])
                if jump > 2.0:
                    issues.append({
                        'type': 'value_jump',
                        'cycle_index': c['index'],
                        'time': str(start_time),
                        'layer': layer,
                        'detail': f'车速段{i}糊间隙值跳变: {values[i-1]} -> {values[i]} (差值={jump:.2f})'
                    })
            if len(values) > 0:
                vmin, vmax = min(values), max(values)
                if vmin < 0:
                    issues.append({
                        'type': 'negative_value',
                        'cycle_index': c['index'],
                        'time': str(start_time),
                        'layer': layer,
                        'detail': f'糊间隙值为负数: {vmin}'
                    })
                if vmax > 60:
                    issues.append({
                        'type': 'exceeds_hard_limit',
                        'cycle_index': c['index'],
                        'time': str(start_time),
                        'layer': layer,
                        'detail': f'糊间隙值{vmax}超过硬限制60'
                    })

            # ── Warp Offset Check ──
            # find offset column index (G4 uses 'offset', G14 uses 'warp_offset')
            cols = layer_data.get('columns', [])
            offset_idx = None
            for col_name in ('warp_offset', 'offset'):
                if col_name in cols:
                    offset_idx = cols.index(col_name)
                    break
            if offset_idx is not None:
                warp_active = False
                max_warp_offset = 0.0
                for r in rows:
                    try:
                        off_val = float(r[offset_idx]) if r[offset_idx] else 0.0
                        if off_val != 0.0:
                            warp_active = True
                            max_warp_offset = max(max_warp_offset, abs(off_val))
                    except (ValueError, TypeError):
                        continue
                if warp_active:
                    issues.append({
                        'type': 'warp_influence',
                        'cycle_index': c['index'],
                        'time': str(start_time),
                        'layer': layer,
                        'detail': f'弯翘偏移量非零（最大={max_warp_offset}）—— 该值受弯翘调平影响'
                    })
        return issues

    # ── Dimension 4: Material Consistency ──
    def check_material_consistency(self):
        issues = []
        for c in self.cycles:
            if c['end'] != 'complete':
                continue
            sfe = c.get('set_func_event')
            if not sfe:
                continue
            material = sfe.get('material', '')
            lifecycle = sfe.get('lifecycle', {})
            start_time = c['start'].get('Date', '')
            has_g11 = c['start'].get('EventId') == 'G11'
            if has_g11:
                g11_pv = c['start'].get('ParsedValues', {})
                offset = g11_pv.get('OffSetValue', 'N/A')
                issues.append({
                    'type': 'immediate_change',
                    'cycle_index': c['index'],
                    'time': str(start_time),
                    'detail': f'触发G11立即换材 —— 偏移量={offset}, 材质={material}'
                })
            df_lifecycle = lifecycle.get('df', {}).get('msg', '')
            if df_lifecycle and material:
                parts = material.split('.')
                material_codes = [p for p in parts if p != '-']
                df_msg_material = df_lifecycle.split('->')[-1].strip().strip('()').split(',')[0] if '->' in df_lifecycle else ''
                if df_msg_material and df_msg_material != material:
                    issues.append({
                        'type': 'material_mismatch',
                        'cycle_index': c['index'],
                        'time': str(start_time),
                        'detail': f'DF生命周期材质"{df_msg_material}"与G7材质"{material}"不一致'
                    })
        return issues

    # ── Dimension 5: Root Cause Traceback ──
    def traceback(self, target_time, expected_values=None):
        ts_target = self._to_ts(target_time)
        ts_target = pd.Timestamp(ts_target) if not isinstance(ts_target, pd.Timestamp) else ts_target

        completed_cycles = [c for c in self.cycles if c['end'] == 'complete']
        active_cycle = None
        prev_completed = None

        for c in completed_cycles:
            g12_time = c['end_event'].get('Date', '')
            if isinstance(g12_time, str):
                try:
                    g12_time = pd.Timestamp(g12_time)
                except Exception:
                    continue
            if g12_time <= ts_target:
                prev_completed = c

        if prev_completed:
            active_cycle = prev_completed

        cancelled_cycles = [c for c in self.cycles if c['end'] == 'cancelled_pre_write']
        cancel_interference = None
        for c in cancelled_cycles:
            g15_time = self._to_ts(c['end_event'].get('Date', ''))
            g15_ts = pd.Timestamp(g15_time) if not isinstance(g15_time, pd.Timestamp) else g15_time
            g7_time = self._to_ts(c['start'].get('Date', ''))
            g7_ts = pd.Timestamp(g7_time) if not isinstance(g7_time, pd.Timestamp) else g7_time
            if prev_completed is None:
                interference = g15_ts > ts_target
            else:
                prev_end = prev_completed['end_event'].get('Date', ts_target)
                prev_end_ts = self._to_ts(prev_end)
                prev_end_ts = pd.Timestamp(prev_end_ts) if not isinstance(prev_end_ts, pd.Timestamp) else prev_end_ts
                interference = g15_ts > prev_end_ts
            if g7_ts <= ts_target and interference:
                cancel_interference = c

        result = {
            'target_time': str(ts_target),
            'has_active_g12': active_cycle is not None,
            'cancel_interference_nearby': cancel_interference is not None,
        }

        if active_cycle:
            sfe = active_cycle.get('set_func_event')
            result['active_cycle'] = {
                'start_time': str(active_cycle['start'].get('Date', '')),
                'g12_time': str(active_cycle['end_event'].get('Date', '')),
                'start_type': active_cycle['start'].get('EventId', ''),
                'material': sfe.get('material', 'N/A') if sfe else 'N/A',
                'flute_type': sfe.get('flute_type', 'N/A') if sfe else 'N/A',
                'set_values': sfe.get('set_values', {}) if sfe else {},
                'lifecycle': sfe.get('lifecycle', {}) if sfe else {},
            }
            if expected_values:
                mismatches = []
                for layer, expected in expected_values.items():
                    layer_data = sfe.get('set_values', {}).get(layer, {}) if sfe else {}
                    data_rows = layer_data.get('data', [])
                    actual_values = []
                    for r in data_rows:
                        try:
                            actual_values.append(float(r[-1]))
                        except (ValueError, TypeError):
                            continue
                    if actual_values:
                        actual_str = f'{min(actual_values):.2f}~{max(actual_values):.2f}'
                        match = 'acceptable' if min(actual_values) <= expected <= max(actual_values) else 'mismatch'
                        mismatches.append({
                            'layer': layer,
                            'expected': expected,
                            'actual_range': actual_str,
                            'match': match
                        })
                    else:
                        mismatches.append({
                            'layer': layer,
                            'expected': expected,
                            'actual_range': 'N/A',
                            'match': 'no_data'
                        })
                result['expected_value_check'] = mismatches
        else:
            result['active_cycle'] = None
            result['message'] = 'No complete write cycle found before target_time'

        if cancel_interference:
            result['cancel_info'] = {
                'cancel_time': str(cancel_interference['end_event'].get('Date', '')),
                'cancel_type': cancel_interference['end'],
                'cancel_cycle_start': str(cancel_interference['start'].get('Date', '')),
                'detail': '写值前被取消 —— 该周期的计算值最终未写入设备'
            }

        # ── Warp Event Correlation ──
        result['warp_active'] = False
        result['warp_events_nearby'] = []
        if self.warp_extractor and active_cycle:
            warp_events = (self.warp_extractor.auto_adjust_events +
                           self.warp_extractor.manual_adjust_events +
                           self.warp_extractor.reset_events +
                           self.warp_extractor.paper_change_events)
            g12_ts = self._to_ts(active_cycle['end_event'].get('Date', ''))
            g12_ts = pd.Timestamp(g12_ts) if not isinstance(g12_ts, pd.Timestamp) else g12_ts
            window_start = g12_ts - pd.Timedelta(minutes=5)
            window_end = g12_ts + pd.Timedelta(minutes=1)
            nearby = []
            for we in warp_events:
                wt = self._to_ts(we.get('time', ''))
                wt_ts = pd.Timestamp(wt) if not isinstance(wt, pd.Timestamp) else wt
                if window_start <= wt_ts <= window_end:
                    nearby.append(we)
            nearby.sort(key=lambda x: str(x.get('time', '')))
            result['warp_active'] = len(nearby) > 0
            result['warp_events_nearby'] = nearby[:10]

        anomalies = self.check_cycle_completeness()
        result['cycle_anomalies'] = [a for a in anomalies
                                     if active_cycle and a['cycle_index'] == active_cycle['index']]

        return result

    # ── Generate Report ──
    def generate_report(self, target_time=None, expected_values=None):
        lines = []

        if target_time:
            tb = self.traceback(target_time, expected_values)
            lines.append('# 糊间隙赋值根因追溯报告')
            lines.append('')
            lines.append(f'**目标时间点**: `{tb["target_time"]}`')
            lines.append('')
            lines.append('## 根因追溯')
            lines.append('')
            lines.append('| 字段 | 值 |')
            lines.append('|------|-----|')
            lines.append(f'| 是否存在有效G12 | `{tb["has_active_g12"]}` |')
            lines.append(f'| 是否存在取消干扰 | `{tb["cancel_interference_nearby"]}` |')
            lines.append('')

            ac = tb.get('active_cycle')
            if ac:
                lines.append('### 生效周期')
                lines.append('')
                lines.append('| 字段 | 值 |')
                lines.append('|------|-----|')
                lines.append(f'| 周期开始时间 | `{ac["start_time"]}` |')
                lines.append(f'| G12写值完成时间 | `{ac["g12_time"]}` |')
                lines.append(f'| 触发类型 | `{ac["start_type"]}` |')
                lines.append(f'| 材质 | `{ac["material"]}` |')
                lines.append(f'| 楞型 | `{ac["flute_type"]}` |')
                lines.append('')

                lifecycle = ac.get('lifecycle', {})
                if lifecycle:
                    lines.append('### 材质变更生命周期')
                    lines.append('')
                    lines.append('| 部位 | 变更内容 | 时间 |')
                    lines.append('|------|----------|------|')
                    for part in ['ls0', 'ms1', 'ls1', 'ms2', 'ls2', 'df']:
                        info = lifecycle.get(part, {})
                        msg = info.get('msg', '') if isinstance(info, dict) else ''
                        tm = info.get('time', '') if isinstance(info, dict) else ''
                        if msg:
                            lines.append(f'| {part.upper()} | `{msg}` | {tm} |')
                    lines.append('')

                sv = ac.get('set_values', {})
                if sv:
                    lines.append('### 糊间隙计算值 (G14)')
                    lines.append('')
                    for layer, ld in sv.items():
                        lines.append(f'#### {layer}')
                        lines.append('')
                        cols = ld.get('columns', [])
                        data = ld.get('data', [])
                        if cols and data:
                            header = '| ' + ' | '.join(col.replace('_', ' ').title() for col in cols) + ' |'
                            sep = '|' + '|'.join(['---'] * len(cols)) + '|'
                            lines.append(header)
                            lines.append(sep)
                            for row in data:
                                lines.append('| ' + ' | '.join(str(v) for v in row) + ' |')
                            lines.append('')

                            # ── 计算说明（用第一段数据演示） ──
                            first = data[0] if data else None
                            if first and len(first) >= len(cols):
                                try:
                                    speed_i = cols.index('speed')
                                    min_g_i = cols.index('min_glue')
                                    max_g_i = cols.index('max_glue')
                                    min_w_i = cols.index('min_weight')
                                    max_w_i = cols.index('max_weight')
                                    cur_w_i = cols.index('current_glue_weight')
                                    qdm_i = cols.index('qdm_factor')
                                    ui_i = cols.index('ui_factor')
                                    spd_i = cols.index('speed_factor')
                                    val_i = cols.index('value')
                                    off_col = 'warp_offset' if 'warp_offset' in cols else ('offset' if 'offset' in cols else None)
                                    off_i = cols.index(off_col) if off_col else None

                                    min_g = float(first[min_g_i])
                                    max_g = float(first[max_g_i])
                                    min_w = float(first[min_w_i])
                                    max_w = float(first[max_w_i])
                                    cur_w = float(first[cur_w_i])
                                    qdm = float(first[qdm_i])
                                    ui = float(first[ui_i])
                                    spd = float(first[spd_i])
                                    off_v = float(first[off_i]) if off_i is not None else 0.0
                                    off_tag = '偏移量' if off_col == 'offset' else ('弯翘偏移' if off_col == 'warp_offset' else '偏移量')

                                    base_gap = min_g + (cur_w - min_w) / (max_w - min_w) * (max_g - min_g) if max_w != min_w else min_g

                                    lines.append('**计算说明**')
                                    lines.append('')
                                    lines.append('```')
                                    lines.append(f'公式：result = base_gap × qdm × ui × speed_coef + {off_tag}')
                                    lines.append(f'       base_gap = min_gap + (cur_weight - min_weight) / (max_weight - min_weight) × (max_gap - min_gap)')
                                    lines.append('')
                                    lines.append(f'base_gap（所有段共用）：')
                                    lines.append(f'  base_gap = {min_g} + ({cur_w:.0f} - {min_w:.0f}) / ({max_w:.0f} - {min_w:.0f}) × ({max_g} - {min_g})')
                                    base_step = (cur_w - min_w) / (max_w - min_w) if max_w != min_w else 0
                                    lines.append(f'           = {min_g} + {base_step:.2f} × {max_g - min_g}')
                                    lines.append(f'           = {base_gap:.2f}')
                                    lines.append('')
                                    lines.append('各段验证：')
                                    all_pass = True
                                    for ri, row in enumerate(data):
                                        try:
                                            rs = float(row[speed_i])
                                            rq = float(row[qdm_i])
                                            ru = float(row[ui_i])
                                            rsp = float(row[spd_i])
                                            rv = float(row[val_i])
                                            roff = float(row[off_i]) if off_i is not None else 0.0
                                            rc = base_gap * rq * ru * rsp + roff
                                            rd = abs(rc - rv)
                                            ok = '✓' if rd < 0.01 else '✗'
                                            if rd >= 0.01:
                                                all_pass = False
                                            lines.append(f'  段{ri+1} (车速={rs:.0f}): {base_gap:.2f} × {rq} × {ru} × {rsp} + {roff} = {rc:.2f}  {ok}')
                                        except (ValueError, IndexError, TypeError):
                                            lines.append(f'  段{ri+1}: 数据异常，跳过')
                                    lines.append('')
                                    lines.append(f'验证：{"全部通过 ✓" if all_pass else "存在偏差 ✗"}')
                                    lines.append('```')
                                    lines.append('')
                                except (ValueError, IndexError, ZeroDivisionError):
                                    pass

                evc = tb.get('expected_value_check')
                if evc:
                    lines.append('### 期望值对比')
                    lines.append('')
                    lines.append('| 部位 | 期望值 | 实际范围 | 匹配结果 |')
                    lines.append('|------|--------|---------|---------|')
                    for m in evc:
                        match_label = '一致' if m['match'] == 'acceptable' else ('不匹配' if m['match'] == 'mismatch' else '无数据')
                        lines.append(f'| {m["layer"]} | {m["expected"]} | {m["actual_range"]} | {match_label} |')
                    lines.append('')

            ci = tb.get('cancel_info')
            if ci:
                lines.append('### ⚠ 检测到赋值取消')
                lines.append('')
                lines.append('| 字段 | 值 |')
                lines.append('|------|-----|')
                cancel_type_label = '写值前取消' if ci['cancel_type'] == 'cancelled_pre_write' else ci['cancel_type']
                lines.append(f'| 取消类型 | `{cancel_type_label}` |')
                lines.append(f'| 取消时间 | `{ci["cancel_time"]}` |')
                lines.append(f'| 详情 | {ci["detail"]} |')
                lines.append('')

            anom = tb.get('cycle_anomalies', [])
            if anom:
                lines.append('### 周期异常')
                lines.append('')
                for a in anom:
                    lines.append(f'- **{a["type"]}**: {a["detail"]}')
                lines.append('')

            # ── Warp Influence Section ──
            if tb.get('warp_active'):
                lines.append('### 弯翘调平影响')
                lines.append('')
                lines.append(f'| 字段 | 值 |')
                lines.append(f'|------|-----|')
                lines.append(f'| 目标时间附近存在弯翘事件 | `是` |')
                lines.append(f'| 弯翘事件数 | `{len(tb["warp_events_nearby"])}` |')
                lines.append('')
                lines.append('#### 附近的弯翘事件')
                lines.append('')
                lines.append('| 时间 | 类型 | 详情 |')
                lines.append('|------|------|------|')
                for we in tb['warp_events_nearby']:
                    wt = we.get('time', '')
                    wtype = we.get('mode', we.get('type', 'unknown'))
                    action = we.get('action', '')
                    detail = f'{wtype}/{action}' if action else wtype
                    lines.append(f'| {wt} | `{detail}` | {we} |')
                lines.append('')

        lines.append('---')
        lines.append('')
        lines.append('# 总体周期统计')
        lines.append('')

        cr = self.calc_cancellation_rate()
        lines.append('| 指标 | 值 |')
        lines.append('|------|-----|')
        lines.append(f'| 总周期数 | {cr["total_cycles"]} |')
        lines.append(f'| 正常触发 (G7) | {cr["g7_starts"]} |')
        lines.append(f'| 立即换材 (G11) | {cr["g11_starts"]} |')
        lines.append(f'| 完成赋值 | {cr["completed"]} |')
        lines.append(f'| 写值前取消 (G15) | {cr["cancelled_pre_write"]} |')
        lines.append(f'| 延迟中取消 (G8) | {cr["cancelled_delay"]} |')
        lines.append(f'| 中断 | {cr["interrupted"]} |')
        lines.append(f'| 取消率 | {cr["cancellation_rate"]}% |')
        if cr.get('alert'):
            lines.append(f'| ⚠ 警告 | 取消率超过30% |')
        lines.append('')

        anom_all = self.check_cycle_completeness()
        if anom_all:
            lines.append('## 完整性异常列表')
            lines.append('')
            for a in anom_all:
                type_labels = {
                    'no_termination': '无终止事件',
                    'fallback_used': '降级匹配 (G10)',
                    'g12_no_g14': 'G12无G14',
                    'pre_write_cancel_no_calc': '写值取消无计算',
                    'excessive_calculation': '过多计算',
                    'value_jump': '值跳变',
                    'speed_not_monotonic': '车速不单调',
                    'negative_value': '负值',
                    'exceeds_hard_limit': '超硬限制',
                    'immediate_change': '立即换材',
                    'material_mismatch': '材质不匹配',
                    'warp_influence': '弯翘调平影响',
                }
                label = type_labels.get(a['type'], a['type'])
                lines.append(f'- 周期 #{a["cycle_index"]} | `{label}` | {a["detail"]}')
            lines.append('')

        return '\n'.join(lines)

    def print_cycle_summary(self):
        lines = []
        lines.append(f"{'序号':<5} {'触发':<8} {'结束状态':<22} {'计算部位':<10} {'G8':<4} {'G10':<4} {'G15':<4}")
        lines.append('-' * 60)
        for c in self.cycles:
            g14_layers = set()
            for e in c['events']:
                if e.get('EventId') == 'G14':
                    pv = e.get('ParsedValues', {})
                    gl = pv.get('glue_part', '?')
                    g14_layers.add(gl)
            has_g8 = any(e.get('EventId') == 'G8' for e in c['events'])
            has_g10 = any(e.get('EventId') == 'G10' for e in c['events'])
            has_g15 = c['end'] == 'cancelled_pre_write'
            trigger = c['start'].get('EventId', '?')
            end_labels = {
                'complete': '完成赋值',
                'cancelled_pre_write': '写值前取消',
                'interrupted': '中断',
                None: '?'
            }
            end_type = end_labels.get(c['end'], c['end'] or '?')
            layers = ','.join(sorted(g14_layers)) if g14_layers else '-'
            lines.append(f"{c['index']:<5} {trigger:<8} {end_type:<22} {layers:<10} {str(has_g8):<4} {str(has_g10):<4} {str(has_g15):<4}")
        return '\n'.join(lines)
