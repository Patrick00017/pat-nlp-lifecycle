import uuid
import re
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from cachetools import TTLCache
from glue_diagnostic import GlueRecordDiagnostic
from utils import parse_time_flexible


# ──────────────────────────────────────────────
# 1. State 定义
# ──────────────────────────────────────────────
class AnalysisState(BaseModel):
    state_id: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    material: Optional[str] = None
    tool_name: str
    tool_args: dict
    tool_result: Any
    analysis_results: Dict[str, Any] = {}
    execution_path: List[str] = []


# ──────────────────────────────────────────────
# 2. 节点定义
# ──────────────────────────────────────────────
class AnalysisNode:
    def __init__(self, id: str, name: str, description: str, tool_name: str, fn: Callable, next_nodes: List[str]):
        self.id = id
        self.name = name
        self.description = description
        self.tool_name = tool_name
        self.fn = fn
        self.next_nodes = next_nodes


def _parse_new_material(msg: str) -> str:
    """从 lifecycle msg 中解析换入的新材料
    输入: '(-,0,-) -> (8,2400,B)'
    输出: '8'
    """
    match = re.search(r'->\s*\(([^,]+)', msg)
    return match.group(1).strip() if match else ''


# ──────────────────────────────────────────────
# 3. 占位处理函数
# ──────────────────────────────────────────────
def _mc_timeline(state: AnalysisState) -> str:
    return f"[时间线视图] 材料 {state.material or '未指定'} 在 {state.start_time} ~ {state.end_time} 间的换材事件：\n  - event 1: N → P\n  - event 2: P → 8\n  - event 3: 8 → J"

def _mc_usage(state: AnalysisState) -> str:
    return f"[用料统计] 各材料使用时长：\n  - P: 45min\n  - N: 30min\n  - 8: 25min\n  - J: 10min"

def _mc_anomaly(state: AnalysisState) -> str:
    return f"[异常检测] 发现 2 次异常换材：\n  - 1. 材料 P 仅在位 30 秒后被切换\n  - 2. 材料 N 在 5 分钟内被切换 2 次"

def _mc_gap(state: AnalysisState) -> str:
    return f"[间隔分析] 相邻换材间隔：\n  - N → P: 45min\n  - P → 8: 30min\n  - 8 → J: 停滞中…"

def _mc_report(state: AnalysisState) -> str:
    prev = state.analysis_results.get(state.execution_path[-2], "") if len(state.execution_path) >= 2 else ""
    return f"[换材分析报告] 基于以下分析生成：\n{prev}"

def _glue_list(state: AnalysisState) -> str:
    """胶量关联分析: 按时间线列出每次胶量设定及关联的换材事件"""
    results = state.tool_result
    if not results:
        return "**无胶量设定事件数据**"
    lines = []
    for record in results:
        func = record['func']
        material = record['material']
        set_time = record['time']
        lifecycle = record.get('lifecycle', {})
        time_short = set_time[:19]
        lines.append(f"### {func} ({material}) @ {time_short}")
        lines.append("")
        lines.append("| 时间 | 事件 |")
        lines.append("|------|------|")
        events = []
        for part in ['ls0', 'ms1', 'ls1', 'ms2', 'ls2', 'df']:
            info = lifecycle.get(part)
            if info and info.get('msg'):
                events.append((info['time'], info['msg'], part.upper()))
        events.sort(key=lambda x: x[0])
        for t, msg, p in events:
            lines.append(f"| {t[:19]} | **{p}** `{msg}` |")
        lines.append(f"| {time_short} | **→ 胶量设定:** `{func}` |")
        lines.append("")
    return "\n".join(lines)

def _glue_trend(state: AnalysisState) -> str:
    """胶量趋势分析: 提取 set_values 中各车速点的胶量值，分析变化趋势"""
    results = state.tool_result
    if not results:
        return "**无胶量设定事件数据**"
    lines = []
    for record in results:
        func = record['func']
        material = record['material']
        for part_key, part_data in record['set_values'].items():
            columns = part_data['columns']
            data = part_data['data']
            speed_idx = columns.index('speed')
            value_idx = columns.index('value')
            points = [(int(row[speed_idx]), float(str(row[value_idx]).split('\\r')[0])) for row in data]
            points.sort(key=lambda x: x[0])
            vals = [v for _, v in points]
            init_v, final_v = vals[0], vals[-1]
            min_v, max_v = min(vals), max(vals)
            trend = "上升" if final_v > init_v else "下降" if final_v < init_v else "稳定"
            lines.append(f"### {func} ({material}) — {part_key}")
            lines.append("")
            lines.append("| 车速 | 胶量 |")
            lines.append("|------|------|")
            for s, v in points:
                marker = " ⬅ 最低" if v == min_v else " ⬅ 最高" if v == max_v else ""
                lines.append(f"| {s} | {v:.2f}{marker} |")
            lines.append("")
            lines.append(f"> 趋势: **{trend}** ({init_v:.2f} → {final_v:.2f}) | 最高={max_v:.2f} | 最低={min_v:.2f}")
            lines.append("")
    return "\n".join(lines)

def _glue_cross(state: AnalysisState) -> str:
    """材质匹配校验: 比对设定材质与 lifecycle 中各部件实际换入的材质"""
    results = state.tool_result
    if not results:
        return "**无数据**"
    lines = []
    for record in results:
        func = record['func']
        material = record['material']
        part = record.get('part', '')
        lifecycle = record.get('lifecycle', {})
        lines.append(f"### {func} ({material})")
        lines.append("")
        lines.append("| 部件 | 换材信息 | 期望材质 | 结果 |")
        lines.append("|------|----------|----------|------|")
        if part in ('SF1', 'SF2'):
            parts_arr = material.split('/')
            if len(parts_arr) != 2:
                continue
            ms_m, ls_m = parts_arr
            ms_p = 'ms1' if part == 'SF1' else 'ms2'
            ls_p = 'ls1' if part == 'SF1' else 'ls2'
            for p, exp in [(ms_p, ms_m), (ls_p, ls_m)]:
                info = lifecycle.get(p)
                if info and info.get('msg'):
                    actual = _parse_new_material(info['msg'])
                    ok = "✅" if actual == exp else "❌ 错位"
                    lines.append(f"| **{p.upper()}** | `{info['msg']}` | `{exp}` | {ok} |")
        elif part == 'DF':
            mat_parts = material.split('.')
            part_map = {'ls0': 0, 'ms1': 1, 'ls1': 2, 'ms2': 3, 'ls2': 4}
            for p, idx in part_map.items():
                info = lifecycle.get(p)
                if info and info.get('msg'):
                    actual = _parse_new_material(info['msg'])
                    expected = mat_parts[idx] if idx < len(mat_parts) else '-'
                    ok = "✅" if actual == expected else "❌ 错位"
                    lines.append(f"| **{p.upper()}** | `{info['msg']}` | `{expected}` | {ok} |")
            df_info = lifecycle.get('df')
            if df_info and df_info.get('msg'):
                actual = _parse_new_material(df_info['msg'])
                ok = "✅" if actual == material else "❌ 错位"
                lines.append(f"| **DF** | `{df_info['msg']}` | `{material}` | {ok} |")
        lines.append("")
    return "\n".join(lines)

NORMAL_RANGES = {
    'qdm_factor': (0.5, 2.0),
    'ui_factor': (0.5, 2.0),
    'offset': (-10, 10),
    'speed_factor': (0.5, 2.0),
}

def _glue_diagnose(state: AnalysisState) -> str:
    """赋值参数诊断: 检查参数合理性、赋值饱和度、材质一致性"""
    results = state.tool_result
    if not results:
        return "**无数据**"
    lines = []
    for record in results:
        func = record['func']
        material = record['material']
        lines.append(f"### {func} ({material})")
        for part_key, part_data in record['set_values'].items():
            columns = part_data['columns']
            data = part_data['data']
            lines.append(f"#### {part_key}")
            lines.append("")
            # 参数合理性
            param_fields = ['qdm_factor', 'ui_factor', 'offset', 'speed_factor']
            param_rows = []
            for field in param_fields:
                if field in columns:
                    idx = columns.index(field)
                    vals = [float(row[idx]) for row in data]
                    lo, hi = NORMAL_RANGES.get(field, (0, 999))
                    ok = all(lo <= v <= hi for v in vals)
                    val_str = f"{vals[0]:.2f}" if len(set(vals)) == 1 else f"{min(vals):.2f} ~ {max(vals):.2f}"
                    param_rows.append((field, val_str, f"{lo} ~ {hi}", "✅" if ok else "⚠️ 异常"))
            if param_rows:
                lines.append("**参数合理性**")
                lines.append("| 参数 | 值 | 正常范围 | 判定 |")
                lines.append("|------|-----|---------|------|")
                for name, val, rng, ok in param_rows:
                    lines.append(f"| `{name}` | {val} | {rng} | {ok} |")
                lines.append("")
            # 赋值饱和度
            min_idx = columns.index('min_glue')
            max_idx = columns.index('max_glue')
            val_idx = columns.index('value')
            if min_idx is not None and max_idx is not None and val_idx is not None:
                min_glues = [float(r[min_idx]) for r in data]
                max_glues = [float(r[max_idx]) for r in data]
                values = [float(str(r[val_idx]).split('\\r')[0]) for r in data]
                touched_min = any(v <= mg for v, mg in zip(values, min_glues))
                touched_max = any(v >= mg for v, mg in zip(values, max_glues))
                lines.append("**赋值饱和度**")
                lines.append(f"- 最低赋值: {min(values):.2f} (下限: {min_glues[0]:.2f}) {'⚠️ 触底' if touched_min else '✅ 正常'}")
                lines.append(f"- 最高赋值: {max(values):.2f} (上限: {max_glues[0]:.2f}) {'⚠️ 触顶' if touched_max else '✅ 正常'}")
                lines.append("")
        # 深度诊断 (9 项)
        diag = GlueRecordDiagnostic(record, None)
        checks = diag.diagnose_all()
        lines.append("**深度诊断 (9 项)**")
        lines.append("| 检查项 | 结果 | 详情 |")
        lines.append("|--------|------|------|")
        for c in checks:
            lines.append(f"| {c.name} | {c.icon} | {c.detail} |")
        lines.append("")
    return "\n".join(lines)

def _glue_advise(state: AnalysisState) -> str:
    """综合分析结果，输出分级诊断建议"""
    results = state.tool_result
    if not results:
        return "**无数据**"

    criticals = []
    warnings = []
    infos = []

    for record in results:
        func = record['func']
        material = record['material']
        tag = f"{func} ({material})"

        for part_key, part_data in record['set_values'].items():
            columns = part_data['columns']
            data = part_data['data']

            # parameter anomalies
            param_fields = ['qdm_factor', 'ui_factor', 'offset', 'speed_factor']
            for field in param_fields:
                if field in columns:
                    idx = columns.index(field)
                    vals = [float(str(row[idx]).split('\\r')[0]) for row in data]
                    lo, hi = NORMAL_RANGES.get(field, (0, 999))
                    abnormal = [(i, v) for i, v in enumerate(vals) if v < lo or v > hi]
                    if abnormal:
                        val_str = ', '.join(f"点{i+1}={v:.2f}" for i, v in abnormal)
                        warnings.append((tag + f" {part_key}", f"`{field}` 越限 [{val_str}]", f"正常范围 {lo}~{hi}，建议检查参数设定"))

            # assignment saturation
            min_idx = columns.index('min_glue')
            max_idx = columns.index('max_glue')
            val_idx = columns.index('value')
            min_glues = [float(str(r[min_idx]).split('\\r')[0]) for r in data]
            max_glues = [float(str(r[max_idx]).split('\\r')[0]) for r in data]
            values = [float(str(r[val_idx]).split('\\r')[0]) for r in data]
            if any(v <= mg for v, mg in zip(values, min_glues)):
                warnings.append((tag + f" {part_key}", "赋值触底", "value 触及 min_glue 下限，建议评估是否需降低下限"))
            if any(v >= mg for v, mg in zip(values, max_glues)):
                warnings.append((tag + f" {part_key}", "赋值触顶", "value 触及 max_glue 上限，建议评估是否需抬升上限"))

            # weight compliance
            if 'current_glue_weight' in columns and 'min_weight' in columns and 'max_weight' in columns:
                cw_idx = columns.index('current_glue_weight')
                minw_idx = columns.index('min_weight')
                maxw_idx = columns.index('max_weight')
                for row in data:
                    cw = float(str(row[cw_idx]).split('\\r')[0])
                    minw = float(str(row[minw_idx]).split('\\r')[0])
                    maxw = float(str(row[maxw_idx]).split('\\r')[0])
                    if cw < minw * 0.95:
                        criticals.append((tag + f" {part_key}", "胶重偏低", f"current_glue_weight={cw:.1f} < min_weight={minw:.1f}，存在上胶不足风险"))
                        break
                    elif cw > maxw * 1.05:
                        criticals.append((tag + f" {part_key}", "胶重偏高", f"current_glue_weight={cw:.1f} > max_weight={maxw:.1f}，存在胶水浪费风险"))
                        break

        # trend cross-reference
        if 'glue_trend' in state.analysis_results:
            for part_key, part_data in record['set_values'].items():
                columns = part_data['columns']
                data = part_data['data']
                speed_idx = columns.index('speed')
                value_idx = columns.index('value')
                points = [(int(str(row[speed_idx]).split('\\r')[0]), float(str(row[value_idx]).split('\\r')[0])) for row in data]
                vals = [v for _, v in points]
                if len(vals) >= 2:
                    change_pct = (vals[-1] - vals[0]) / vals[0] * 100 if vals[0] != 0 else 0
                    if change_pct > 20:
                        infos.append((tag + f" {part_key}", "趋势显著上升", f"胶量上升 {change_pct:.0f}%，建议关注胶辊/刮刀磨损"))
                    elif change_pct < -20:
                        infos.append((tag + f" {part_key}", "趋势显著下降", f"胶量下降 {change_pct:.0f}%，建议检查胶泵或管路"))

        # lifecycle coverage
        lifecycle = record.get('lifecycle', {})
        if record['part'] == 'DF':
            expected_parts = ['ls0', 'ms1', 'ls1', 'ms2', 'ls2', 'df']
        elif record['part'] in ('SF1', 'SF2'):
            expected_parts = ['ms1', 'ls1'] if record['part'] == 'SF1' else ['ms2', 'ls2']
        else:
            expected_parts = []
        if expected_parts:
            covered = [p for p in expected_parts if lifecycle.get(p, {}).get('msg')]
            missing = [p for p in expected_parts if not lifecycle.get(p, {}).get('msg')]
            if missing:
                warnings.append((tag, "换材覆盖不完整", f"缺 {'/'.join(missing).upper()} 的换材事件，共 {len(covered)}/{len(expected_parts)}"))
            else:
                infos.append((tag, "换材覆盖完整", f"{len(covered)}/{len(expected_parts)} 部件均有换材记录"))

    # cross-reference cross results
    cross_match_count = 0
    cross_total_count = 0
    if 'glue_cross' in state.analysis_results:
        cross_text = state.analysis_results['glue_cross']
        for line in cross_text.split('\n'):
            if '|' in line and ('✅' in line or '❌' in line):
                cols = [c.strip() for c in line.split('|') if c.strip()]
                if len(cols) >= 3:
                    cross_total_count += 1
                    if '❌' in line:
                        part_name = cols[0].replace('*', '')
                        actual_info = cols[1]
                        expected = cols[2]
                        criticals.append((part_name, "材质错位", f"实际 {actual_info}，期望 {expected}，请核实换材操作"))
                    elif '✅' in line:
                        cross_match_count += 1
        if cross_total_count > 0:
            match_pct = cross_match_count / cross_total_count * 100
            status = "全部匹配" if match_pct == 100 else f"{match_pct:.0f}% 匹配"
            infos.append(("材质校验", f"换材匹配率 {match_pct:.0f}%", f"{cross_match_count}/{cross_total_count} 部件{status}"))

    if not criticals and not warnings and not infos:
        return "**未发现异常**"

    lines = ["## 诊断建议"]
    lines.append("")

    if criticals:
        lines.append("### 🔴 Critical")
        lines.append("| 来源 | 问题 | 建议 |")
        lines.append("|------|------|------|")
        for src, issue, suggestion in criticals:
            lines.append(f"| {src} | {issue} | {suggestion} |")
        lines.append("")

    if warnings:
        lines.append("### 🟡 Warning")
        lines.append("| 来源 | 问题 | 建议 |")
        lines.append("|------|------|------|")
        for src, issue, suggestion in warnings:
            lines.append(f"| {src} | {issue} | {suggestion} |")
        lines.append("")

    if infos:
        lines.append("### 🔵 Info")
        lines.append("| 来源 | 问题 | 建议 |")
        lines.append("|------|------|------|")
        for src, issue, suggestion in infos:
            lines.append(f"| {src} | {issue} | {suggestion} |")
        lines.append("")

    return "\n".join(lines)

def _glue_report(state: AnalysisState) -> str:
    """汇总子节点的分析结果"""
    parts = []
    for node_id in ['glue_trend', 'glue_cross', 'glue_diagnose', 'glue_advise']:
        if node_id in state.analysis_results:
            parts.append(state.analysis_results[node_id])
    if not parts:
        return "**请先运行趋势分析、材质校验、参数诊断或诊断建议**"
    return "## 胶量分析报告\n\n" + "\n---\n".join(parts)

def _track_lifecycle(state: AnalysisState) -> str:
    return f"[生命周期总览] 材料 {state.material} 的流转路径：\n  - LS0 → MS1 → LS1 → MS2 → LS2\n  - 当前状态: LS2 在位中"

def _track_part(state: AnalysisState) -> str:
    return f"[部件分解] 各部件详情：\n  - LS0: 材料 N (2600mm, B楞)\n  - MS1: 无\n  - LS1: 无\n  - MS2: 材料 8\n  - LS2: 材料 J"

def _track_report(state: AnalysisState) -> str:
    prev = state.analysis_results.get(state.execution_path[-2], "") if len(state.execution_path) >= 2 else ""
    return f"[生命周期报告] 基于以下分析生成：\n{prev}"

def _press_list(state: AnalysisState) -> str:
    return f"[事件列表] MP 压力辊设定事件 ({state.start_time} ~ {state.end_time})：\n  - 10:05:00 → 压力设至 3.5bar\n  - 10:20:00 → 压力设至 4.0bar\n  - 10:35:00 → 压力设至 3.8bar"

def _press_trend(state: AnalysisState) -> str:
    return f"[趋势分析] 压力变化趋势：\n  - 初始: 3.5bar\n  - 最高: 4.0bar\n  - 当前: 3.8bar"

def _press_cross(state: AnalysisState) -> str:
    return f"[关联分析] 压力设定与换材关联：\n  - 10:05 压力变更 → 10:06 换材至 P\n  - 10:20 压力变更 → 10:22 换材至 N"

def _press_report(state: AnalysisState) -> str:
    prev = state.analysis_results.get(state.execution_path[-2], "") if len(state.execution_path) >= 2 else ""
    return f"[压力辊分析报告] 基于以下分析生成：\n{prev}"


# ──────────────────────────────────────────────
# 4. 节点注册
# ──────────────────────────────────────────────
ENTRY_NODE_MAP = {
    "get_material_change_in_log": "mc_timeline",
    "get_glue_set_func_call_in_log": "glue_list",
    "track_material_in_log": "track_lifecycle",
    "get_pressroll_mp_set_func_call_in_log": "press_list",
}

NODES: Dict[str, AnalysisNode] = {
    # ── get_material_change_in_log ──
    "mc_timeline": AnalysisNode("mc_timeline", "时间线视图", "按时间顺序展示所有换材事件", "get_material_change_in_log", _mc_timeline, ["mc_usage", "mc_anomaly", "mc_gap"]),
    "mc_usage":    AnalysisNode("mc_usage", "用料统计", "统计各材料的使用时长", "get_material_change_in_log", _mc_usage, ["mc_report"]),
    "mc_anomaly":  AnalysisNode("mc_anomaly", "异常检测", "检测频繁切换等异常行为", "get_material_change_in_log", _mc_anomaly, ["mc_report"]),
    "mc_gap":      AnalysisNode("mc_gap", "间隔分析", "分析相邻换材的时间间隔", "get_material_change_in_log", _mc_gap, ["mc_report"]),
    "mc_report":   AnalysisNode("mc_report", "生成报告", "汇总分析结果生成换材报告", "get_material_change_in_log", _mc_report, []),

    # ── get_glue_set_func_call_in_log ──
    "glue_list":   AnalysisNode("glue_list", "事件列表", "展示胶量设定事件", "get_glue_set_func_call_in_log", _glue_list, ["glue_trend", "glue_cross", "glue_diagnose", "glue_advise"]),
    "glue_trend":  AnalysisNode("glue_trend", "趋势分析", "分析胶量设定值的变化趋势", "get_glue_set_func_call_in_log", _glue_trend, ["glue_report"]),
    "glue_cross":  AnalysisNode("glue_cross", "关联分析", "关联胶量设定与换材事件", "get_glue_set_func_call_in_log", _glue_cross, ["glue_report"]),
    "glue_diagnose": AnalysisNode("glue_diagnose", "赋值参数诊断", "诊断参数合理性、赋值饱和度", "get_glue_set_func_call_in_log", _glue_diagnose, ["glue_report"]),
    "glue_advise": AnalysisNode("glue_advise", "诊断建议", "综合分析结果输出分级诊断建议", "get_glue_set_func_call_in_log", _glue_advise, ["glue_report"]),
    "glue_report": AnalysisNode("glue_report", "生成报告", "汇总分析结果生成胶量报告", "get_glue_set_func_call_in_log", _glue_report, []),

    # ── track_material_in_log ──
    "track_lifecycle": AnalysisNode("track_lifecycle", "生命周期总览", "展示材料在系统中的完整流转路径", "track_material_in_log", _track_lifecycle, ["track_part"]),
    "track_part":      AnalysisNode("track_part", "部件分解", "查看各部件（LS0/MS1/…）的详情", "track_material_in_log", _track_part, ["track_report"]),
    "track_report":    AnalysisNode("track_report", "生成报告", "汇总分析结果生成生命周期报告", "track_material_in_log", _track_report, []),

    # ── get_pressroll_mp_set_func_call_in_log ──
    "press_list":   AnalysisNode("press_list", "事件列表", "展示压力辊设定事件", "get_pressroll_mp_set_func_call_in_log", _press_list, ["press_trend", "press_cross"]),
    "press_trend":  AnalysisNode("press_trend", "趋势分析", "分析压力设定值的变化趋势", "get_pressroll_mp_set_func_call_in_log", _press_trend, ["press_report"]),
    "press_cross":  AnalysisNode("press_cross", "关联分析", "关联压力设定与换材事件", "get_pressroll_mp_set_func_call_in_log", _press_cross, ["press_report"]),
    "press_report": AnalysisNode("press_report", "生成报告", "汇总分析结果生成压力辊报告", "get_pressroll_mp_set_func_call_in_log", _press_report, []),
}


# ──────────────────────────────────────────────
# 5. State 提取器（从 tool_args 中提取 start/end/material）
# ──────────────────────────────────────────────
TOOL_STATE_EXTRACTORS = {
    "get_material_change_in_log": lambda args: {
        "start_time": args.get("start_time"),
        "end_time": args.get("end_time"),
        "material": args.get("material"),
    },
    "get_glue_set_func_call_in_log": lambda args: {
        "start_time": (parse_time_flexible(args.get("time")) - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "end_time": (parse_time_flexible(args.get("time")) + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "material": args.get("desire_material"),
    },
    "track_material_in_log": lambda args: {
        "start_time": args.get("start_time"),
        "end_time": args.get("end_time"),
        "material": args.get("material"),
    },
    "get_pressroll_mp_set_func_call_in_log": lambda args: {
        "start_time": args.get("start_time"),
        "end_time": args.get("end_time"),
        "material": args.get("desire_material"),
    },
}


# ──────────────────────────────────────────────
# 6. Router & Endpoints
# ──────────────────────────────────────────────
router = APIRouter(prefix="/analysis")

states: TTLCache[str, AnalysisState] = TTLCache(maxsize=128, ttl=1800)


@router.post("/init")
async def init_analysis(body: dict):
    tool_name = body.get("tool_name")
    tool_args = body.get("tool_args", {})
    tool_result = body.get("tool_result")

    if tool_name not in ENTRY_NODE_MAP:
        available = list(ENTRY_NODE_MAP.keys())
        raise HTTPException(status_code=400, detail=f"Unknown tool '{tool_name}', available: {available}")

    if tool_result is None:
        from tools_definition import tools
        tm = {t.name: t for t in tools}
        t = tm.get(tool_name)
        if t is None:
            raise HTTPException(status_code=400, detail=f"Tool '{tool_name}' not found in tools_definition")
        try:
            tool_result = t.invoke(tool_args)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")

    extractor = TOOL_STATE_EXTRACTORS[tool_name]
    extracted = extractor(tool_args)

    state_id = str(uuid.uuid4())
    state = AnalysisState(
        state_id=state_id,
        tool_name=tool_name,
        tool_args=tool_args,
        tool_result=tool_result,
        **extracted,
    )
    states[state_id] = state

    entry_id = ENTRY_NODE_MAP[tool_name]
    entry_node = NODES[entry_id]

    return {
        "state_id": state_id,
        "context": {
            "start_time": state.start_time,
            "end_time": state.end_time,
            "material": state.material,
        },
        "available_nodes": [
            {
                "id": entry_node.id,
                "name": entry_node.name,
                "description": entry_node.description,
            }
        ],
    }


@router.post("/step")
async def step_analysis(body: dict):
    state_id = body.get("state_id")
    node_id = body.get("node_id")

    if state_id not in states:
        raise HTTPException(status_code=404, detail=f"State '{state_id}' not found")

    if node_id not in NODES:
        raise HTTPException(status_code=400, detail=f"Unknown node '{node_id}'")

    state = states[state_id]
    node = NODES[node_id]

    if node.tool_name != state.tool_name:
        raise HTTPException(status_code=400, detail=f"Node '{node_id}' does not belong to tool '{state.tool_name}'")

    result = node.fn(state)
    state.analysis_results[node_id] = result
    state.execution_path.append(node_id)

    next_nodes = []
    for nid in node.next_nodes:
        nn = NODES[nid]
        next_nodes.append({
            "id": nn.id,
            "name": nn.name,
            "description": nn.description,
        })

    return {
        "result": result,
        "available_nodes": next_nodes,
        "execution_path": state.execution_path,
        "is_terminal": len(node.next_nodes) == 0,
    }


@router.post("/stepback")
async def stepback_analysis(body: dict):
    state_id = body.get("state_id")

    if state_id not in states:
        raise HTTPException(status_code=404, detail=f"State '{state_id}' not found")

    state = states[state_id]

    if not state.execution_path:
        raise HTTPException(status_code=400, detail="No steps to step back from")

    undone_node_id = state.execution_path.pop()
    state.analysis_results.pop(undone_node_id, None)

    if state.execution_path:
        last_node_id = state.execution_path[-1]
        last_node = NODES[last_node_id]
        next_nodes = [
            {"id": nn.id, "name": nn.name, "description": nn.description}
            for nn in (NODES[nid] for nid in last_node.next_nodes)
        ]
    else:
        entry_id = ENTRY_NODE_MAP[state.tool_name]
        entry_node = NODES[entry_id]
        next_nodes = [
            {"id": entry_node.id, "name": entry_node.name, "description": entry_node.description}
        ]

    previous_result = ""
    if state.execution_path:
        last_id = state.execution_path[-1]
        previous_result = state.analysis_results.get(last_id, "")

    return {
        "undone_node": undone_node_id,
        "previous_result": previous_result,
        "execution_path": state.execution_path,
        "available_nodes": next_nodes,
        "is_terminal": len(next_nodes) == 0,
    }


@router.get("/state/{state_id}")
async def get_analysis_state(state_id: str):
    if state_id not in states:
        raise HTTPException(status_code=404, detail=f"State '{state_id}' not found")
    return states[state_id]
