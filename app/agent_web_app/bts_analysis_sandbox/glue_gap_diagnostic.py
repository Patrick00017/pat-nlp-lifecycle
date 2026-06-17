import pandas as pd
from datetime import datetime
from collections import defaultdict


class GlueGapDiagnostic:
    TRIG_LABELS = {'G7': '换材触发', 'G11': '立即换材'}

    def __init__(self, extractor, warp_extractor=None, dev_ips=None):
        self.extractor = extractor
        self.warp_extractor = warp_extractor
        self.dev_ips = dev_ips
        self.raw_events = extractor.raw_parsed_rows
        self.raw_events.sort(key=lambda x: str(x.get('Date', '')))
        self.set_func_events = extractor.get_glue_set_function_full_event()
        self.cycles = self._group_cycles()

    @classmethod
    def from_params(cls, start_time, end_time, dev_ips=None):
        import os
        from log_parser import test_ips_and_glue_template
        sandbox = os.path.dirname(os.path.abspath(__file__))
        old_cwd = os.getcwd()
        try:
            os.chdir(sandbox)
            extractor = test_ips_and_glue_template(start_time=start_time, end_time=end_time)
        finally:
            os.chdir(old_cwd)
        return cls(extractor, dev_ips=dev_ips)

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
                current = {'start': evt, 'events': [], 'end': None, 'end_event': None, 'set_func_event': None}
                current['events'].append(evt)
            elif current is None:
                continue
            else:
                current['events'].append(evt)
                if eid in ('G12', 'G5'):
                    current['end'] = 'complete'
                    current['end_event'] = evt
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

        # Merge set_values from all G12/G5 events within each cycle
        for c in cycles:
            merged_sv = {}
            merged_sfe = None
            for evt in c['events']:
                if evt.get('EventId') in ('G12', 'G5'):
                    ts = str(evt.get('Date', ''))
                    for sfe in self.set_func_events:
                        if str(sfe.get('time', '')) == ts:
                            sv = sfe.get('set_values', {})
                            merged_sv.update(sv)
                            if merged_sfe is None:
                                merged_sfe = dict(sfe)
                            break
            if merged_sfe:
                merged_sfe['set_values'] = merged_sv
                c['set_func_event'] = merged_sfe

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
                    'detail': f'{start_type}（{self.TRIG_LABELS.get(start_type, "其他触发")}）触发的周期({start_time})缺少G12（写值完成）或G15（写值前取消）结束事件'
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
                    s = float(r[0]) if r[0] else 0
                    v = float(r[-1]) if r[-1] else 0
                except (ValueError, TypeError):
                    continue
                speeds.append(s)
                values.append(v)
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
        mismatch_indices = set()
        for c in self.cycles:
            if c['end'] != 'complete':
                continue
            sfe = c.get('set_func_event')
            if not sfe:
                continue
            material = sfe.get('material', '')
            lifecycle = sfe.get('lifecycle', {})
            start_time = c['start'].get('Date', '')
            df_lifecycle = lifecycle.get('df', {}).get('msg', '')
            if df_lifecycle and material:
                parts = material.split('.')
                material_codes = [p for p in parts if p != '-']
                df_msg_material = df_lifecycle.split('->')[-1].strip().strip('()').split(',')[0] if '->' in df_lifecycle else ''
                if df_msg_material and df_msg_material != material:
                    mismatch_indices.add(c['index'])
                    issues.append({
                        'type': 'material_mismatch',
                        'cycle_index': c['index'],
                        'time': str(start_time),
                        'detail': f'周期#{c["index"]}: DF生命周期材质"{df_msg_material}"与G7材质"{material}"不一致'
                    })

        # G11 立即换材：仅当该周期同时存在材质错位时才记录
        for c in self.cycles:
            if c['end'] != 'complete' or c['index'] not in mismatch_indices:
                continue
            if c['start'].get('EventId') == 'G11':
                sfe = c.get('set_func_event')
                material = sfe.get('material', '') if sfe else ''
                g11_pv = c['start'].get('ParsedValues', {})
                offset = g11_pv.get('OffSetValue', 'N/A')
                issues.append({
                    'type': 'immediate_change',
                    'cycle_index': c['index'],
                    'time': str(c['start'].get('Date', '')),
                    'detail': f'周期#{c["index"]}: 触发G11立即换材 —— 偏移量={offset}, 材质={material}'
                })
        return issues

    @staticmethod
    def _extract_layer_values(set_values):
        """Extract per-layer speed→value pairs from set_values."""
        layers = []
        for layer, ld in set_values.items():
            data = ld.get('data', [])
            cols = ld.get('columns', [])
            try:
                si = cols.index('speed')
                vi = cols.index('value')
            except (ValueError, IndexError):
                layers.append({'name': layer, 'segments': []})
                continue
            segs = []
            for r in data:
                try:
                    segs.append({'speed': r[si], 'value': r[vi]})
                except IndexError:
                    continue
            layers.append({'name': layer, 'segments': segs})
        return layers

    # ── Dimension 5: Cross-Source Consistency (requires dev_ips) ──
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

    @staticmethod
    def _compact_material(material):
        if not material:
            return ''
        parts = material.split('.')
        compact = [p for p in parts if p != '-']
        return '.'.join(compact)

    def check_cross_source_consistency(self):
        issues = []
        if self.dev_ips is None:
            return issues

        for c in self.cycles:
            if c['end'] != 'complete':
                continue
            sfe = c.get('set_func_event')
            if not sfe:
                continue

            material = sfe.get('material', '')
            flute = sfe.get('flute_type', '')
            sv = sfe.get('set_values', {})
            paper_codes = [p for p in material.split('.') if p != '-'] if material else []

            # Get weight from devIPS S_PaperCodes (sum of all non-dash parts' SPC_GlueWeight)
            expected_weight = None
            if paper_codes:
                total = 0.0
                all_found = True
                for pc in paper_codes:
                    rows = self._query_ips(
                        'SELECT "SPC_GlueWeight" FROM "S_PaperCodes" WHERE "SPC_Code" = %s',
                        (pc,)
                    )
                    if rows and rows[0][0]:
                        total += float(rows[0][0])
                    else:
                        all_found = False
                        break
                if all_found:
                    expected_weight = total

            # Get QDM coefficients from devIPS (uses compacted code, no dashes)
            qdm_map = {}
            compact_code = self._compact_material(material)
            col_map = {'GU1': 'F_Glue1', 'GU2': 'F_Glue2', 'GU3': 'F_Glue3'}
            gu_cols = ', '.join(f'"{c}"' for c in col_map.values())

            qdm_rows = self._query_ips(
                f'SELECT "F_Paper", {gu_cols} FROM "TB_IPS_QdmCoefDF" WHERE "F_Paper" = %s AND "F_Flute" = %s',
                (compact_code, flute)
            )
            if not qdm_rows and compact_code != material:
                like_pat = compact_code.replace('-', '_')
                qdm_rows = self._query_ips(
                    f'SELECT "F_Paper", {gu_cols} FROM "TB_IPS_QdmCoefDF" WHERE "F_Paper" LIKE %s AND "F_Flute" = %s LIMIT 5',
                    (like_pat, flute)
                )
            if not qdm_rows and '-' in compact_code:
                like_pat2 = compact_code.replace('-', '%')
                qdm_rows = self._query_ips(
                    f'SELECT "F_Paper", {gu_cols} FROM "TB_IPS_QdmCoefDF" WHERE "F_Paper" LIKE %s AND "F_Flute" = %s LIMIT 5',
                    (like_pat2, flute)
                )
            if qdm_rows:
                for layer_key, col_name in col_map.items():
                    ci = list(col_map.values()).index(col_name)
                    qdm_map[layer_key] = float(qdm_rows[0][ci + 1]) if len(qdm_rows[0]) > ci + 1 and qdm_rows[0][ci + 1] else None

            if not qdm_rows:
                issues.append({
                    'type': 'qdm_no_data',
                    'cycle_index': c['index'],
                    'layer': ','.join(sv.keys()),
                    'start_time': str(c['start'].get('Date', '')),
                    'detail': f'材质"{compact_code}"+楞型"{flute}"在QDM配置表中无对应条目，无法验证QDM系数'
                })

            for layer, ld in sv.items():
                data = ld.get('data', [])
                if not data:
                    continue
                cols = ld.get('columns', [])
                try:
                    wi = cols.index('current_glue_weight')
                    qi = cols.index('qdm_factor')
                except ValueError:
                    continue

                first = data[0]
                actual_weight = float(first[wi]) if first[wi] else 0
                actual_qdm = float(first[qi]) if first[qi] else 0

                # Weight check
                if expected_weight is not None:
                    diff = abs(actual_weight - expected_weight)
                    if diff > 1:
                        issues.append({
                            'type': 'weight_mismatch',
                            'cycle_index': c['index'],
                            'layer': layer,
                            'start_time': str(c['start'].get('Date', '')),
                            'detail': f'G14使用克重={actual_weight:.0f}g, 数据库(纸板档案)={expected_weight:.0f}g, 差异={actual_weight - expected_weight:.0f}g'
                        })

                # QDM check
                expected_qdm = qdm_map.get(layer)
                if expected_qdm is not None:
                    if abs(actual_qdm - expected_qdm) > 0.01:
                        issues.append({
                            'type': 'qdm_mismatch',
                            'cycle_index': c['index'],
                            'layer': layer,
                            'start_time': str(c['start'].get('Date', '')),
                            'detail': f'G14使用QDM系数={actual_qdm}, 数据库(QDM配方)={expected_qdm}, 差异={actual_qdm - expected_qdm:.2f}'
                        })

                # Base setting check (min_glue, max_glue, min_weight, max_weight)
                try:
                    min_g_i = cols.index('min_glue')
                    max_g_i = cols.index('max_glue')
                    min_w_i = cols.index('min_weight')
                    max_w_i = cols.index('max_weight')
                except ValueError:
                    continue

                actual_min_g = float(first[min_g_i]) if first[min_g_i] else 0
                actual_max_g = float(first[max_g_i]) if first[max_g_i] else 0
                actual_min_w = float(first[min_w_i]) if first[min_w_i] else 0
                actual_max_w = float(first[max_w_i]) if first[max_w_i] else 0

                if layer.startswith('GU'):
                    pos = {'GU1': '1', 'GU2': '2', 'GU3': '3'}.get(layer, '')
                    rows = self._query_ips(
                        'SELECT "F_MinGlue", "F_MaxGlue", "F_MinWeight", "F_MaxWeight" FROM "TB_IPS_GlueGu" WHERE "F_Flute" = %s AND "F_Position" = %s',
                        (flute, pos)
                    )
                elif layer.startswith('SF'):
                    rows = self._query_ips(
                        'SELECT "F_MinGlue", "F_MaxGlue", "F_MinWeight", "F_MaxWeight" FROM "TB_IPS_GlueSF" WHERE "F_Flute" = %s',
                        (flute,)
                    )
                else:
                    rows = None

                if rows:
                    db_min_g, db_max_g, db_min_w, db_max_w = [float(v) if v else 0 for v in rows[0]]
                    field_map = [
                        ('min_glue', actual_min_g, db_min_g),
                        ('max_glue', actual_max_g, db_max_g),
                        ('min_weight', actual_min_w, db_min_w),
                        ('max_weight', actual_max_w, db_max_w),
                    ]
                    for field_name, actual, expected in field_map:
                        if abs(actual - expected) > 0:
                            issues.append({
                                'type': 'base_setting_mismatch',
                                'cycle_index': c['index'],
                                'layer': layer,
                                'start_time': str(c['start'].get('Date', '')),
                                'detail': f'G14使用{field_name}={actual}, 数据库基础设置={expected}, 差异={actual - expected:.0f}'
                            })

                # Speed coefficient check (TB_IPS_GlueSpeedCoef)
                pos_map = {'GU1': 1, 'GU2': 2, 'GU3': 3, 'SF1': 4, 'SF2': 5, 'SF3': 6}
                pos = pos_map.get(layer)
                if pos is not None:
                    spd_rows = self._query_ips(
                        'SELECT "F_Speed", "F_Coef", "F_MinValue" FROM "TB_IPS_GlueSpeedCoef" WHERE "F_Position" = %s ORDER BY "F_Speed"',
                        (pos,)
                    )
                    if spd_rows:
                        try:
                            spd_i = cols.index('speed')
                            sf_i = cols.index('speed_factor')
                        except ValueError:
                            spd_i = sf_i = None
                        if spd_i is not None:
                            db_speed_map = {int(r[0]): float(r[1]) for r in spd_rows}
                            for ri, row in enumerate(data):
                                actual_spd = float(row[spd_i]) if row[spd_i] else 0
                                actual_sf = float(row[sf_i]) if sf_i is not None and row[sf_i] else 0
                                expected_sf = db_speed_map.get(int(actual_spd))
                                if expected_sf is not None and abs(actual_sf - expected_sf) > 0:
                                    issues.append({
                                        'type': 'speed_coef_mismatch',
                                        'cycle_index': c['index'],
                                        'layer': layer,
                                        'start_time': str(c['start'].get('Date', '')),
                                        'detail': f'段{ri+1}(speed={actual_spd:.0f}): G14使用speed_factor={actual_sf}, 数据库(GlueSpeedCoef)={expected_sf}, 差异={actual_sf - expected_sf:+.2f}'
                                    })
                                    break  # 每层只报第一个不匹配段

        return issues

    # ── Dimension 6: Root Cause Traceback ──
    def traceback(self, target_time, expected_values=None, recent_count=5):
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

        all_anomalies = self.check_cycle_completeness()
        result['cycle_anomalies'] = [a for a in all_anomalies
                                     if active_cycle and a['cycle_index'] == active_cycle['index']]

        # ── Recent Assignment Events ──
        ts_min = ts_target - pd.Timedelta(hours=1)
        recent = []
        mc_all = self.check_material_consistency()
        cs_all = self.check_cross_source_consistency()
        for c in reversed(self.cycles):
            end_ts = None
            if c['end'] == 'complete' and c.get('end_event'):
                raw = c['end_event'].get('Date', '')
                try:
                    end_ts = pd.Timestamp(raw) if not isinstance(raw, pd.Timestamp) else raw
                except Exception:
                    continue
            elif c['end'] == 'cancelled_pre_write' and c.get('end_event'):
                raw = c['end_event'].get('Date', '')
                try:
                    end_ts = pd.Timestamp(raw) if not isinstance(raw, pd.Timestamp) else raw
                except Exception:
                    continue
            elif c['end'] == 'interrupted':
                raw = c['start'].get('Date', '')
                try:
                    end_ts = pd.Timestamp(raw) if not isinstance(raw, pd.Timestamp) else raw
                except Exception:
                    continue
            if end_ts is None:
                continue
            if end_ts > ts_target:
                continue
            if end_ts < ts_min:
                break

            sfe = c.get('set_func_event')
            entry = {
                'index': c['index'],
                'start_time': str(c['start'].get('Date', '')),
                'end_time': str(end_ts),
                'end': c['end'],
                'start_type': c['start'].get('EventId', ''),
                'material': sfe.get('material', 'N/A') if sfe else 'N/A',
                'layers': self._extract_layer_values(sfe.get('set_values', {})) if sfe else [],
                'is_active': active_cycle is not None and c['index'] == active_cycle['index'],
            }

            tags = []
            for a in all_anomalies:
                if a['cycle_index'] == c['index']:
                    tag_map = {'no_termination': '被抢断', 'fallback_used': '降级匹配',
                                'g12_no_g14': '写值完成但缺少计算过程', 'pre_write_cancel_no_calc': '写值取消且没有计算记录',
                               'excessive_calculation': '重复计算'}
                    tags.append(tag_map.get(a['type'], a['type']))

            # direct warp_offset check on set_values
            if sfe:
                has_warp = False
                for ld in sfe.get('set_values', {}).values():
                    cols = ld.get('columns', [])
                    off_name = next((n for n in ('warp_offset', 'offset') if n in cols), None)
                    if off_name is None:
                        continue
                    off_idx = cols.index(off_name)
                    for r in ld.get('data', []):
                        try:
                            if float(r[off_idx]) != 0.0:
                                has_warp = True
                                break
                        except (ValueError, TypeError, IndexError):
                            continue
                    if has_warp:
                        break
                if has_warp:
                    tags.append('弯翘影响')

            # ── 错误汇总（材质不匹配 + 跨来源，按 type 去重） ──
            error_details = []
            error_seen = set()
            for mm in mc_all:
                if mm.get('type') == 'material_mismatch' and mm['cycle_index'] == c['index']:
                    error_details.append(mm['detail'])
                    error_seen.add('material_mismatch')
                    tags.append('材质不匹配')
                    break
            cs_labels = {'weight_mismatch': '克重不匹配', 'qdm_mismatch': 'QDM系数不匹配',
                         'qdm_no_data': 'QDM无配置', 'base_setting_mismatch': '基础设置不匹配', 'speed_coef_mismatch': '车速系数不匹配'}
            for cs in cs_all:
                if cs['cycle_index'] == c['index'] and cs['type'] not in error_seen:
                    error_seen.add(cs['type'])
                    error_details.append(cs['detail'])
                    tags.append(cs_labels.get(cs['type'], cs['type']))
            entry['error_detail'] = '；'.join(error_details) if error_details else '-'

            entry['anomalies'] = tags
            recent.append(entry)
            if len(recent) >= recent_count:
                break
        result['recent_assignments'] = recent

        # ── Cross-Source Consistency ──
        result['cross_source_issues'] = cs_all

        return result

    # ── Generate Report ──
    def generate_report(self, target_time=None, expected_values=None):
        lines = []
        anom_all = self.check_cycle_completeness()
        cs_all = self.check_cross_source_consistency()
        mc_all = self.check_material_consistency()

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
            lines.append(f'| 是否存在有效G12/G5 | `{tb["has_active_g12"]}` |')
            lines.append(f'| 是否存在取消干扰 | `{tb["cancel_interference_nearby"]}` |')
            ac = tb.get('active_cycle')
            if ac:
                st = ac["start_type"]
                trig_desc = self.TRIG_LABELS.get(st, '其他触发')
                lines.append(f'| 生效周期 | #{ac.get("g12_time", "N/A")} — `{st}（{trig_desc}）` — 材质 `{ac["material"]}` |')
            ci = tb.get('cancel_info')
            if ci:
                cancel_type_label = '写值前取消' if ci['cancel_type'] == 'cancelled_pre_write' else ci['cancel_type']
                lines.append(f'| 取消干扰 | `{cancel_type_label}` 于 `{ci["cancel_time"]}` |')
            evc = tb.get('expected_value_check')
            if evc:
                lines.append('')
                lines.append('| 部位 | 期望值 | 实际范围 | 匹配结果 |')
                lines.append('|------|--------|---------|---------|')
                for m in evc:
                    match_label = '一致' if m['match'] == 'acceptable' else ('不匹配' if m['match'] == 'mismatch' else '无数据')
                    lines.append(f'| {m["layer"]} | {m["expected"]} | {m["actual_range"]} | {match_label} |')
            lines.append('')

        # ── 周期详情（所有完成周期，每个周期独立小节） ──
        lines.append('## 周期详情')
        lines.append('')
        for c in self.cycles:
            sfe = c.get('set_func_event')
            eid = c['start'].get('EventId', '?')
            trig_desc = self.TRIG_LABELS.get(eid, '其他触发')
            material = sfe.get('material', '-') if sfe else '-'
            flute = sfe.get('flute_type', '-') if sfe else '-'
            start_t = str(c['start'].get('Date', ''))[:19]
            end_labels = {'complete': '完成', 'cancelled_pre_write': '写值取消', 'interrupted': '中断'}
            cl = end_labels.get(c['end'], str(c['end']))

            lines.append(f'### 周期 #{c["index"]} ({cl})')
            lines.append('')
            lines.append(f'- **触发时间**: {start_t}')
            lines.append(f'- **触发原因**: {eid}（{trig_desc}）')
            lines.append(f'- **材质**: {material}')
            lines.append(f'- **楞型**: {flute}')
            if c['end'] != 'complete':
                if c['end'] == 'interrupted':
                    lines.append(f'- **状态说明**: 被新任务抢断，未完成赋值')
                elif c['end'] == 'cancelled_pre_write':
                    lines.append(f'- **状态说明**: 写值前被取消，未写入 PLC')
                lines.append('')
                continue

            lifecycle = sfe.get('lifecycle', {}) if sfe else {}
            if lifecycle:
                df_msg = lifecycle.get('df', {}).get('msg', '')
                if df_msg:
                    lines.append(f'- **材质变更**: {df_msg}')
            lines.append('')

            sv = sfe.get('set_values', {}) if sfe else {}

            # ── 克重计算说明 ──
            paper_parts = [p for p in material.split('.') if p != '-'] if material else []
            if paper_parts and self.dev_ips:
                weights = []
                all_found = True
                for pc in paper_parts:
                    rows = self._query_ips(
                        'SELECT "SPC_GlueWeight" FROM "S_PaperCodes" WHERE "SPC_Code" = %s',
                        (pc,)
                    )
                    if rows and rows[0][0]:
                        weights.append(float(rows[0][0]))
                    else:
                        weights.append(0)
                        all_found = False
                if all_found:
                    total_db = sum(weights)
                    part_names = ['纸种', 'SF1层', 'SF2层', 'GU层', '品牌']
                    lines.append('**克重计算**')
                    lines.append('')
                    lines.append(f'材质 `{material}` 各段解析：')
                    for i, (pc, w) in enumerate(zip(paper_parts, weights)):
                        name = part_names[i] if i < len(part_names) else f'第{i+1}段'
                        lines.append(f'  - {name}: {pc} → SPC_GlueWeight = **{w:.0f}**')
                    lines.append('')
                    sum_str = ' + '.join(f'{int(w)}' for w in weights)
                    lines.append(f'总克重: {sum_str} = **{total_db:.0f} g**')
                    g14_weight = None
                    for layer, ld in sv.items():
                        data = ld.get('data', [])
                        if not data:
                            continue
                        try:
                            wi = ld.get('columns', []).index('current_glue_weight')
                            g14_weight = float(data[0][wi]) if data[0][wi] else None
                        except (ValueError, IndexError):
                            pass
                        break
                    if g14_weight is not None:
                        diff = g14_weight - total_db
                        lines.append(f'G14 实际使用克重: {g14_weight:.0f} g  → 差异: {diff:+.0f} g')
                    lines.append('')

            # ── 层计算值（每层独立表格 + 公式验证） ──
            if sv:
                for layer, ld in sv.items():
                    cols = ld.get('columns', [])
                    data = ld.get('data', [])
                    if not cols or not data:
                        continue
                    lines.append(f'#### {layer} 糊间隙计算值')
                    lines.append('')
                    header = '| ' + ' | '.join(col.replace('_', ' ').title() for col in cols) + ' |'
                    sep = '|' + '|'.join(['---'] * len(cols)) + '|'
                    lines.append(header)
                    lines.append(sep)
                    for row in data:
                        lines.append('| ' + ' | '.join(str(v) for v in row) + ' |')
                    lines.append('')

                    # 公式验证
                    first = data[0]
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

            # ── 错误 / 警告 / 信息 ──
            errs, warns, infos = [], [], []
            err_tag_map = {'material_mismatch': '材质不匹配', 'weight_mismatch': '克重不匹配',
                           'qdm_mismatch': 'QDM系数不匹配', 'qdm_no_data': 'QDM无配置',
                           'base_setting_mismatch': '基础设置不匹配'}
            warn_tag_map = {'fallback_used': '降级匹配', 'warp_influence': '弯翘影响',
                            'excessive_calculation': '重复计算', 'value_jump': '值跳变',
                            'negative_value': '负值', 'exceeds_hard_limit': '超硬限制',
                            'speed_not_monotonic': '车速不单调'}
            info_tag_map = {'no_termination': '被抢断', 'g12_no_g14': '写值完成但缺少计算过程',
                            'pre_write_cancel_no_calc': '写值取消且没有计算记录'}
            for a in anom_all:
                if a['cycle_index'] == c['index']:
                    t = a['type']
                    if t in err_tag_map:
                        errs.append(f"{err_tag_map[t]}（{a['detail']}）")
                    elif t in warn_tag_map:
                        warns.append(warn_tag_map[t])
                    elif t in info_tag_map:
                        infos.append(info_tag_map[t])
            for cs in cs_all:
                if cs['cycle_index'] == c['index']:
                    cs_labels = {'weight_mismatch': '克重不匹配', 'qdm_mismatch': 'QDM系数不匹配',
                                 'qdm_no_data': 'QDM无配置', 'base_setting_mismatch': '基础设置不匹配', 'speed_coef_mismatch': '车速系数不匹配'}
                    errs.append(f"{cs_labels.get(cs['type'], cs['type'])}（{cs['detail']}）")
            for wp in [x for x in self.check_value_plausibility('GU1') + self.check_value_plausibility('GU2')
                       + self.check_value_plausibility('GU3') + self.check_value_plausibility('SF1')
                       + self.check_value_plausibility('SF2') + self.check_value_plausibility('SF3')
                       if x.get('type') == 'warp_influence']:
                if wp['cycle_index'] == c['index']:
                    warns.append('弯翘影响')

            if errs:
                lines.append('**错误**')
                lines.append('')
                for e in errs:
                    lines.append(f'- ❌ {e}')
                lines.append('')
            if warns:
                lines.append('**警告**')
                lines.append('')
                for w in set(warns):
                    lines.append(f'- ⚠ {w}')
                lines.append('')
            if infos:
                lines.append('**信息**')
                lines.append('')
                for i in infos:
                    lines.append(f'- ℹ {i}')
                lines.append('')

        # ── 最近赋值事件序列（保持原样） ──
        if target_time and tb:
            recent = tb.get('recent_assignments', [])
            if recent:
                lines.append('## 最近赋值事件序列')
                lines.append('')
                lines.append(f'目标时间点 `{tb["target_time"]}` 前最近的 {len(recent)} 次赋值事件：')
                lines.append('')
                lines.append('| 序号 | 触发时间 | 完成时间 | 材质 | 层 | 最终值 | 异常标签 | 错误 |')
                lines.append('|------|---------|---------|------|----|--------|---------|------|')
                for pos, ra in enumerate(recent):
                    rev_idx = len(recent) - pos
                    idx_label = f"T-{rev_idx} (#{ra['index']})"
                    if ra.get('is_active'):
                        idx_label = f"T-{rev_idx} (#{ra['index']}) ←生效"
                    t_start = str(ra['start_time']).split('.')[0] if '.' in str(ra['start_time']) else str(ra['start_time'])[:19]
                    t_end = str(ra['end_time']).split('.')[0] if '.' in str(ra['end_time']) else str(ra['end_time'])[:19]
                    material = ra['material']
                    layer_names = ', '.join(l['name'] for l in ra['layers']) if ra['layers'] else '-'
                    if ra['end'] == 'complete':
                        layer_parts = []
                        for lyr in ra['layers']:
                            segs = lyr.get('segments', [])
                            if segs:
                                val_str = ' / '.join(f"@{s['speed']}={s['value']}" for s in segs)
                                layer_parts.append(f"{lyr['name']}: {val_str}")
                            else:
                                layer_parts.append(f"{lyr['name']}: (无数据)")
                        values_str = '; '.join(layer_parts)
                    elif ra['end'] == 'cancelled_pre_write':
                        values_str = '(写值取消)'
                        layer_names = '-'
                    else:
                        values_str = '(中断)'
                        layer_names = '-'
                    anom_str = ', '.join(ra['anomalies']) if ra['anomalies'] else '-'
                    err_str = ra.get('error_detail', '-')
                    lines.append(f'| {idx_label} | {t_start} | {t_end} | {material} | {layer_names} | {values_str} | {anom_str} | {err_str} |')
                lines.append('')

                # 结论
                error_cycles = []
                for ra_item in recent:
                    idx = ra_item['index']
                    labels = []
                    seen = set()
                    if '材质不匹配' in ra_item.get('anomalies', []):
                        labels.append('材质和系统记录对不上')
                        seen.add('material_mismatch')
                    for cs in cs_all:
                        if cs['cycle_index'] == idx and cs['type'] not in seen:
                            seen.add(cs['type'])
                            cs_plain = {'weight_mismatch': '实际克重和档案不一致', 'qdm_mismatch': 'QDM系数和配方不一致',
                                        'qdm_no_data': 'QDM配方没找到对应配置', 'base_setting_mismatch': '糊间隙基础参数设定对不上', 'speed_coef_mismatch': '车速系数和数据库对不上'}
                            labels.append(cs_plain.get(cs['type'], cs['type']))
                    if labels:
                        error_cycles.append((idx, labels))
                if error_cycles:
                    lines.append('**结论：发现了问题**')
                    lines.append('')
                    sep = '；'
                    for idx, labels in error_cycles:
                        lines.append(f'- 周期 #{idx} 存在以下错误：{sep.join(labels)}')
                else:
                    lines.append('**结论：这几次赋值都没有发现任何问题，数据正常**')
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

        if anom_all:
            lines.append('## 完整性异常列表')
            lines.append('')
            for a in anom_all:
                type_labels = {
                    'no_termination': '无终止事件（无G12/G15）',
                    'fallback_used': '降级匹配（G10）',
                    'g12_no_g14': '写值完成无计算（G12无G14）',
                    'pre_write_cancel_no_calc': '写值取消无计算（G15无G14）',
                    'excessive_calculation': '过多计算（G14>3）',
                    'value_jump': '值跳变',
                    'speed_not_monotonic': '车速不单调',
                    'negative_value': '负值',
                    'exceeds_hard_limit': '超硬限制',
                    'immediate_change': '立即换材（G11）',
                    'material_mismatch': '材质不匹配',
                    'warp_influence': '弯翘调平影响',
                }
                label = type_labels.get(a['type'], a['type'])
                lines.append(f'- 周期 #{a["cycle_index"]} | `{label}` | {a["detail"]}')
            lines.append('')

        # ── 跨来源一致性检查 ──
        if cs_all:
            lines.append('## 跨来源一致性检查')
            lines.append('')
            lines.append('| 周期 | 部位 | 类型 | 说明 |')
            lines.append('|------|------|------|------|')
            type_labels_cs = {'weight_mismatch': '克重不匹配', 'qdm_mismatch': 'QDM系数不匹配', 'qdm_no_data': 'QDM无配置', 'base_setting_mismatch': '基础设置不匹配', 'speed_coef_mismatch': '车速系数不匹配'}
            for cs in cs_all:
                label = type_labels_cs.get(cs['type'], cs['type'])
                lines.append(f'| #{cs["cycle_index"]} | {cs["layer"]} | {label} | {cs["detail"]} |')
            lines.append('')

        return '\n'.join(lines)

    def generate_json(self, target_time=None, expected_values=None):
        result = {}
        anomalies = self.check_cycle_completeness()
        mc_issues = self.check_material_consistency()
        cs_issues = self.check_cross_source_consistency()
        cr = self.calc_cancellation_rate()
        wp_issues = []
        for layer in ('GU1', 'GU2', 'GU3', 'SF1', 'SF2', 'SF3'):
            for iss in self.check_value_plausibility(layer):
                if iss['type'] == 'warp_influence':
                    wp_issues.append(iss)

        trig_labels = {'G7': '换材触发', 'G11': '立即换材'}
        end_labels = {'complete': '完成', 'cancelled_pre_write': '写值取消', 'interrupted': '中断'}

        result['analysis_period'] = {}
        result['anomaly_summary'] = {}
        cycles_data = []
        seen_cs_types_per_cycle = {}

        for c in self.cycles:
            sfe = c.get('set_func_event')
            eid = c['start'].get('EventId', '?')
            material = sfe.get('material', '-') if sfe else '-'
            flute = sfe.get('flute_type', '-') if sfe else '-'
            start_t = str(c['start'].get('Date', ''))[:19]

            cycle = {
                'index': c['index'],
                'status': {'id': c['end'], 'label': end_labels.get(c['end'], str(c['end']))},
                'trigger_time': start_t,
                'trigger': {'id': eid, 'label': f'{eid}（{trig_labels.get(eid, "其他触发")}）'},
                'material': material,
                'flute': flute,
                'computed_values': {},
                'errors': [],
                'warnings': [],
                'infos': [],
            }

            # Computed values
            if c['end'] == 'complete' and sfe:
                sv = sfe.get('set_values', {})
                for layer, ld in sv.items():
                    data = ld.get('data', [])
                    cols = ld.get('columns', [])
                    if not data:
                        continue
                    try:
                        si = cols.index('speed')
                        vi = cols.index('value')
                        segments = []
                        for row in data:
                            segments.append({
                                'speed': row[si] if si < len(row) else None,
                                'value': row[vi] if vi < len(row) else None,
                            })
                        if segments:
                            cycle['computed_values'][layer] = segments
                    except ValueError:
                        continue

            # Errors / Warnings / Infos
            for a in anomalies:
                if a['cycle_index'] == c['index']:
                    err_tag_map = {'material_mismatch': '材质不匹配', 'weight_mismatch': '克重不匹配',
                                   'qdm_mismatch': 'QDM系数不匹配', 'qdm_no_data': 'QDM无配置',
                                   'base_setting_mismatch': '基础设置不匹配'}
                    warn_tag_map = {'fallback_used': '降级匹配', 'warp_influence': '弯翘影响',
                                    'excessive_calculation': '重复计算', 'value_jump': '值跳变',
                                    'negative_value': '负值', 'exceeds_hard_limit': '超硬限制',
                                    'speed_not_monotonic': '车速不单调'}
                    info_tag_map = {'no_termination': '被抢断', 'g12_no_g14': '写值完成但缺少计算过程',
                                    'pre_write_cancel_no_calc': '写值取消且没有记录计算'}
                    t = a['type']
                    if t in err_tag_map:
                        cycle['errors'].append({'type': t, 'label': err_tag_map[t], 'detail': a['detail']})
                    elif t in warn_tag_map:
                        cycle['warnings'].append(warn_tag_map[t])
                    elif t in info_tag_map:
                        cycle['infos'].append(info_tag_map[t])
            for cs in cs_issues:
                if cs['cycle_index'] == c['index']:
                    cs_labels = {'weight_mismatch': '克重不匹配', 'qdm_mismatch': 'QDM系数不匹配',
                                 'qdm_no_data': 'QDM无配置', 'base_setting_mismatch': '基础设置不匹配', 'speed_coef_mismatch': '车速系数不匹配'}
                    cycle['errors'].append({'type': cs['type'], 'label': cs_labels.get(cs['type'], cs['type']), 'detail': cs['detail']})
            for wp in wp_issues:
                if wp['cycle_index'] == c['index']:
                    cycle['warnings'].append('弯翘影响')

            cycle['warnings'] = list(set(cycle['warnings']))
            cycles_data.append(cycle)

        result['cycles'] = cycles_data

        # Recent assignments
        if target_time:
            tb = self.traceback(target_time, expected_values)
            ra_list = tb.get('recent_assignments', [])
            ra_json = []
            for pos, e in enumerate(ra_list):
                rev_idx = len(ra_list) - pos
                layers_list = [l['name'] for l in e.get('layers', [])]
                vals = []
                for lyr in e.get('layers', []):
                    segs = lyr.get('segments', [])
                    if segs:
                        vals.extend(f"@{s['speed']}={s['value']}" for s in segs)
                ra_json.append({
                    't_label': f'T-{rev_idx}',
                    'cycle_index': e['index'],
                    'is_active': e.get('is_active', False),
                    'end_time': str(e['end_time'])[:19],
                    'end_status': {'id': e['end'], 'label': end_labels.get(e['end'], e['end'])},
                    'material': e['material'],
                    'layers': layers_list,
                    'values': vals,
                    'anomalies': e.get('anomalies', []),
                    'error_detail': e.get('error_detail', '-'),
                })

            # Conclusion
            seen = set()
            error_cycles = []
            for ra_item in ra_list:
                seen = set()
                idx = ra_item['index']
                labels = []
                if '材质不匹配' in ra_item.get('anomalies', []):
                    labels.append('材质和系统记录对不上')
                    seen.add('material_mismatch')
                for cs in cs_issues:
                    if cs['cycle_index'] == idx and cs['type'] not in seen:
                        seen.add(cs['type'])
                        cs_plain = {'weight_mismatch': '实际克重和档案不一致', 'qdm_mismatch': 'QDM系数和配方不一致',
                                    'qdm_no_data': 'QDM配方没找到对应配置', 'base_setting_mismatch': '糊间隙基础参数设定对不上', 'speed_coef_mismatch': '车速系数和数据库对不上'}
                        labels.append(cs_plain.get(cs['type'], cs['type']))
                if labels:
                    error_cycles.append({'index': idx, 'labels': labels})

            sep = '；'
            if error_cycles:
                plain_lines = [f'周期 #{ec["index"]} 存在以下错误：{sep.join(ec["labels"])}' for ec in error_cycles]
                plain_text = '；'.join(plain_lines)
            else:
                plain_text = '这几次赋值都没有发现任何问题，数据正常'

            result['recent_assignments'] = {
                'target_time': str(tb['target_time'])[:19],
                'events': ra_json,
                'conclusion': {
                    'has_errors': len(error_cycles) > 0,
                    'cycles_with_errors': error_cycles,
                    'plain_text': plain_text,
                },
            }

        return result

    def print_cycle_summary(self):
        lines = []
        lines.append(f"{'序号':<5} {'触发原因':<16} {'结束状态':<22} {'计算部位':<10} {'G8(延迟取消)':<12} {'G10(降匹配)':<12} {'G15(写取消)':<12}")
        lines.append('-' * 92)
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
            eid = c['start'].get('EventId', '?')
            trigger = f"{eid}（{self.TRIG_LABELS.get(eid, '?')}）"
            end_labels = {
                'complete': '完成赋值',
                'cancelled_pre_write': '写值前取消',
                'interrupted': '中断',
                None: '?'
            }
            end_type = end_labels.get(c['end'], c['end'] or '?')
            layers = ','.join(sorted(g14_layers)) if g14_layers else '-'
            lines.append(f"{c['index']:<5} {trigger:<16} {end_type:<22} {layers:<10} {str(has_g8):<12} {str(has_g10):<12} {str(has_g15):<12}")
        return '\n'.join(lines)
