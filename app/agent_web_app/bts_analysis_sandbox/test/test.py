"""
Glue Gap Diagnostic Test
Usage: conda run -n pat-nlp-lifecycle python test\test.py
"""

import sys, os, traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
from log_parser import LogParser, test_ips_and_glue_template, test_wrap_template
from glue_gap_diagnostic import GlueGapDiagnostic


def run_diagnostic_from_db():
    """Run diagnostic with real data from database."""
    print("=" * 60)
    print("正在连接数据库并解析日志...")
    print("=" * 60)

    start_time = "2026-01-08 17:03:50.690"
    end_time = "2026-01-08 18:03:50.690"

    extractor = test_ips_and_glue_template(start_time=start_time, end_time=end_time)

    print(f"\n匹配到 {len(extractor.raw_parsed_rows)} 个G事件")
    print(f"材质变更事件: {len(extractor.material_events)}")
    print(f"赋值函数调用: {len(extractor.set_func_call_events)}")

    print("\n正在查询弯翘数据...")
    warp_extractor = test_wrap_template(start_time=start_time, end_time=end_time)
    ws = warp_extractor.get_summary()
    print(f"弯翘事件总数: {ws['total_warp_events']}")
    print(f"自动调平: {ws['auto_adjust_count']}, 手动调平: {ws['manual_adjust_count']}")
    print(f"复位: {ws['reset_count']}, 换材跟踪: {ws['paper_change_count']}")

    diagnostic = GlueGapDiagnostic(extractor, warp_extractor=warp_extractor)

    # ── 收集所有异常 ──
    anomalies = diagnostic.check_cycle_completeness()
    cr = diagnostic.calc_cancellation_rate()
    mc_issues = diagnostic.check_material_consistency()
    wp_issues = []
    for layer in ("GU1", "GU2", "GU3", "SF1", "SF2", "SF3"):
        for iss in diagnostic.check_value_plausibility(layer):
            if iss["type"] == "warp_influence":
                wp_issues.append(iss)

    # ── 输出报告 ──
    print()
    print("=" * 60)
    print("糊间隙赋值异常分析报告")
    print("=" * 60)
    print(f"分析时段: {start_time.split('.')[0]} ~ {end_time.split('.')[0]}")
    print(f"共匹配 {len(extractor.raw_parsed_rows)} 条事件")

    print()
    print("--- 发现的异常 ---")
    print()

    idx = 1

    # 1) 取消率过高
    if cr["alert"]:
        print(f"{idx}. [警告] 取消率过高（{cr['cancellation_rate']}%）")
        print(
            f"   {cr['total_cycles']}次赋值请求中{cr['cancelled_delay'] + cr['cancelled_pre_write']}次被新任务打断"
        )
        print(
            f"   ({cr['cancelled_delay']}次延迟中取消，{cr['cancelled_pre_write']}次写值前取消)"
        )
        print(f"   说明换材请求过于密集，系统来不及处理")
        print(f"   → 建议：检查生产排程，避免短时间内频繁换材")
        print()
        idx += 1

    # 2) G10 降级匹配
    g10_cycles = [a for a in anomalies if a["type"] == "fallback_used"]
    if g10_cycles:
        print(f"{idx}. [警告] 材质与设备部位不匹配（{len(g10_cycles)}次降级匹配）")
        print(f"   系统无法将材质层数与用户勾选的部位对应，使用了默认匹配规则")
        print(f"   此时胶水曲线可能不准确，影响糊间隙质量")
        print(f"   → 建议：检查 HMI 上糊机糊间隙使用部位的勾选是否与实际纸层一致")
        print()
        idx += 1

    # 3) 无终止事件
    no_term = [a for a in anomalies if a["type"] == "no_termination"]
    if no_term:
        print(f"{idx}. [信息] 有 {len(no_term)} 个赋值请求被新任务抢断（未完成）")
        for nt in no_term:
            t = (
                str(nt.get("start_time", "")).split(".")[0]
                if "." in str(nt.get("start_time", ""))
                else str(nt.get("start_time", ""))
            )
            print(f"   · {t} 触发的请求被后续请求取代")
        print()
        idx += 1

    # 4) 弯翘影响
    if wp_issues:
        max_off = max(
            abs(float(i["detail"].split("最大=")[-1].split(")")[0]))
            for i in wp_issues
            if "最大=" in i["detail"]
        )
        print(f"{idx}. [警告] 弯翘调平正在影响胶水赋值")
        print(f"   弯翘偏移量最大为 {max_off}，涉及 {len(wp_issues)} 个部位")
        print(f"   → 建议：检查弯翘模块的调平记录（WARP事件），确认偏移量是否合理")
        print()
        idx += 1
    else:
        print(f"{idx}. [信息] 本次分析未发现弯翘调平的影响")
        print(f"   所有胶水计算中的弯翘偏移量均为0")
        print()
        idx += 1

    # 5) 材质一致性
    if mc_issues:
        print(f"{idx}. [信息] 材质变更记录不一致")
        for mi in mc_issues:
            print(f"   · {mi['detail']}")
        print()
        idx += 1

    # ── 周期概览 ──
    print("--- 周期概览 ---")
    print()
    print(
        f"{'周期':<6} {'触发时间':<23} {'触发原因':<18} {'最终状态':<12} {'问题标签'}"
    )
    print("-" * 78)
    end_labels = {
        "complete": "已完成",
        "cancelled_pre_write": "写值取消",
        "interrupted": "未完成",
    }
    trig_labels = {"G7": "换材触发", "G11": "立即换材"}
    for c in diagnostic.cycles:
        cl = end_labels.get(c["end"], str(c["end"]))
        eid = c["start"].get("EventId", "?")
        trig = f"{eid}（{trig_labels.get(eid, '其他触发')}）"
        tags = []
        for a in anomalies:
            if a["cycle_index"] == c["index"]:
                tag_map = {
                    "no_termination": "被抢断",
                    "fallback_used": "降级匹配",
                    "g12_no_g14": "缺计算",
                    "pre_write_cancel_no_calc": "取消无计算",
                    "excessive_calculation": "重复计算",
                }
                tags.append(tag_map.get(a["type"], a["type"]))
        for wp in wp_issues:
            if wp["cycle_index"] == c["index"]:
                tags.append("弯翘影响")
        tag_str = ", ".join(tags) if tags else "-"
        t = (
            str(c["start"].get("Date", "")).split(".")[0]
            if "." in str(c["start"].get("Date", ""))
            else str(c["start"].get("Date", ""))
        )
        print(f"#{c['index']:<4} {t:<23} {trig:<18} {cl:<12} {tag_str}")

    # ── 显示最终写入值 ──
    print("--- 糊间隙计算值 ---")
    print()
    for c in diagnostic.cycles:
        if c["end"] != "complete":
            continue
        sfe = c.get("set_func_event")
        if not sfe:
            continue
        sv = sfe.get("set_values", {})
        if not sv:
            continue
        for layer, ld in sv.items():
            data = ld.get("data", [])
            if not data:
                continue
            cols = ld.get("columns", [])
            try:
                speed_idx = cols.index("speed")
                val_idx = cols.index("value")
                segments = []
                for row in data:
                    s = row[speed_idx] if speed_idx < len(row) else "?"
                    v = row[val_idx] if val_idx < len(row) else "?"
                    segments.append(f"@{s}={v}")
                print(f"周期 #{c['index']} ({layer}) → {' / '.join(segments)}")
            except ValueError:
                continue
    print()

    # -- Root Cause Traceback: Recent Assignment Events --
    completed_indices = [
        c["index"] for c in diagnostic.cycles if c["end"] == "complete"
    ]
    if completed_indices:
        # last_completed = diagnostic.cycles[completed_indices[-1]]
        # g12_raw = last_completed.get("end_event", {}).get("Date", "")
        # g12_ts = str(g12_raw).split(".")[0] if "." in str(g12_raw) else str(g12_raw)
        # target = str(pd.Timestamp(g12_ts) + pd.Timedelta(seconds=30))
        target = "2026-01-08 17:50:10"
        tb = diagnostic.traceback(target, recent_count=5)
        ra = tb.get("recent_assignments", [])
        if ra:
            print("--- 最近赋值事件序列 ---")
            print(f"   目标时间: {target.split('.')[0]}")
            print()
            end_labels = {
                "complete": "完成",
                "cancelled_pre_write": "取消",
                "interrupted": "中断",
            }
            for pos, e in enumerate(ra):
                rev_idx = len(ra) - pos
                idx = f"T-{rev_idx} (#{e['index']})" + (
                    " *生效" if e.get("is_active") else ""
                )
                t = (
                    str(e["end_time"]).split(".")[0]
                    if "." in str(e["end_time"])
                    else str(e["end_time"])[:19]
                )
                end_s = end_labels.get(e["end"], e["end"])
                layers = (
                    ", ".join(l["name"] for l in e["layers"]) if e["layers"] else "-"
                )
                if e["end"] == "complete":
                    vals = []
                    for lyr in e["layers"]:
                        segs = lyr.get("segments", [])
                        if segs:
                            vals.append(
                                " / ".join(
                                    f"@{s['speed']}={s['value']}" for s in segs[:4]
                                )
                            )
                    val_s = "; ".join(vals)
                else:
                    val_s = end_s
                anom = ", ".join(e["anomalies"]) if e["anomalies"] else "-"
                print(f"  {idx:<12} {t:<20} {layers:<6} {e['material']:<16} {val_s}")
                if anom != "-":
                    print(f"              异常: {anom}")
                if e.get("error_detail"):
                    print(f"              错误: {e['error_detail']}")
            print()

    # ── 确认错误汇总 ──
    if mc_issues:
        print("--- 确认错误 ---")
        print()
        mm_real = [m for m in mc_issues if m.get("type") == "material_mismatch"]
        for mi in mm_real:
            print(f"  [#{mi['cycle_index']:<4}] 材质不匹配: {mi['detail']}")
        print()

    print("--- 排除建议 ---")
    print()
    print("如要忽略上述问题重新排查其他方向：")
    print("  1. 修改 start_time / end_time，选择不同时段再次分析")
    print(f"  2. 如需关注面纸糊机(GU)，需确认该时段有 GU 赋值活动")
    print("  3. 分析结果已保存到 diagnostic_report.md（含完整技术细节）")

    return diagnostic


def run_diagnostic_synthetic():
    """Run diagnostic with synthetic test data (no DB needed)."""
    print("=" * 60)
    print("使用模拟数据进行诊断（含弯翘影响）")
    print("=" * 60)

    from event_extractor import GlueEventExtractor, WarpEventExtractor

    ext = GlueEventExtractor()
    ext.raw_parsed_rows = [
        {
            "EventId": "G1",
            "ParsedValues": {
                "handle_func_name": "HandleGuGlueMsg",
                "meterial": "P.-.-.8.J",
                "flute_type": "3B",
            },
            "Date": "2026-01-08 14:00:00",
        },
        {
            "EventId": "G7",
            "ParsedValues": {
                "set_func_name": "SetGlueGu",
                "material": "P.-.-.8.J",
                "flute_type": "3B",
            },
            "Date": "2026-01-08 14:00:01",
        },
        {
            "EventId": "G14",
            "ParsedValues": {"glue_part": "GU1"},
            "Date": "2026-01-08 14:00:05",
        },
        {
            "EventId": "G14",
            "ParsedValues": {"glue_part": "GU2"},
            "Date": "2026-01-08 14:00:06",
        },
        {
            "EventId": "G12",
            "ParsedValues": {
                "set_func_name": "SetGlueGu",
                "material": "P.-.-.8.J",
                "flute_type": "3B",
            },
            "Date": "2026-01-08 14:00:10",
        },
        {
            "EventId": "G7",
            "ParsedValues": {
                "set_func_name": "SetGlueGu",
                "material": "P.-.-.8.J",
                "flute_type": "3B",
            },
            "Date": "2026-01-08 14:01:00",
        },
        {
            "EventId": "G14",
            "ParsedValues": {"glue_part": "GU1"},
            "Date": "2026-01-08 14:01:05",
        },
        {"EventId": "G15", "ParsedValues": {}, "Date": "2026-01-08 14:01:08"},
        {
            "EventId": "G7",
            "ParsedValues": {
                "set_func_name": "SetGlueGu",
                "material": "T.-.-.7.J",
                "flute_type": "B",
            },
            "Date": "2026-01-08 14:02:00",
        },
        {"EventId": "G8", "ParsedValues": {}, "Date": "2026-01-08 14:02:02"},
        {"EventId": "G10", "ParsedValues": {}, "Date": "2026-01-08 14:02:03"},
        {
            "EventId": "G14",
            "ParsedValues": {"glue_part": "GU1"},
            "Date": "2026-01-08 14:02:05",
        },
        {
            "EventId": "G14",
            "ParsedValues": {"glue_part": "GU3"},
            "Date": "2026-01-08 14:02:06",
        },
        {"EventId": "G8", "ParsedValues": {}, "Date": "2026-01-08 14:02:07"},
        {
            "EventId": "G14",
            "ParsedValues": {"glue_part": "GU1"},
            "Date": "2026-01-08 14:02:08",
        },
        {"EventId": "G15", "ParsedValues": {}, "Date": "2026-01-08 14:02:09"},
    ]

    # 模拟弯翘数据
    warp_ext = WarpEventExtractor()
    warp_ext.auto_adjust_events = [
        {"mode": "auto", "action": "UP1", "time": "2026-01-08 14:00:08"},
        {"mode": "auto", "action": "DOWN2", "time": "2026-01-08 14:01:03"},
    ]
    warp_ext.reset_events = [
        {"type": "auto", "time": "2026-01-08 14:00:12"},
    ]
    warp_ext.paper_change_events = [
        {"type": "tracking", "df_remain": "50", "time": "2026-01-08 14:00:30"},
    ]

    d = GlueGapDiagnostic(ext, warp_extractor=warp_ext)

    print("\n--- 周期汇总 ---")
    print(d.print_cycle_summary())

    print("\n--- 取消率 ---")
    cr = d.calc_cancellation_rate()
    labels_cr = {
        "total_cycles": "总周期数",
        "g7_starts": "G7触发",
        "g11_starts": "G11触发",
        "completed": "完成",
        "cancelled_pre_write": "写值前取消",
        "cancelled_delay": "延迟取消",
        "interrupted": "中断",
        "cancellation_rate": "取消率",
        "alert": "警告",
    }
    for k, v in cr.items():
        label = labels_cr.get(k, k)
        print(f"  {label}: {v}")
    if cr.get("alert"):
        print("  [警告] 取消率超过30%")

    print("\n--- 完整性异常 ---")
    type_labels_anom = {
        "no_termination": "无终止事件",
        "fallback_used": "降级匹配",
        "g12_no_g14": "G12无G14",
        "pre_write_cancel_no_calc": "写值取消无计算",
        "excessive_calculation": "过多计算",
    }
    anomalies = d.check_cycle_completeness()
    if anomalies:
        for a in anomalies:
            anom_label = type_labels_anom.get(a["type"], a["type"])
            print(f"  [{anom_label}] 周期#{a['cycle_index']}: {a['detail']}")
    else:
        print("  (无)")

    print("\n--- 糊间隙值合理性检查 + 弯翘偏移检查 (GU1) ---")
    d.cycles[0]["set_func_event"] = {
        "set_values": {
            "GU1": {
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
                    "warp_offset",
                    "value",
                ],
                "data": [
                    [
                        "50",
                        "1",
                        "5",
                        "100",
                        "200",
                        "150",
                        "0.9",
                        "1.0",
                        "1.0",
                        "1.0",
                        "1.5",
                        "2.0",
                    ],
                    [
                        "100",
                        "1",
                        "5",
                        "100",
                        "200",
                        "150",
                        "0.9",
                        "1.0",
                        "1.0",
                        "1.0",
                        "1.5",
                        "3.5",
                    ],
                    [
                        "150",
                        "1",
                        "5",
                        "100",
                        "200",
                        "150",
                        "1.2",
                        "1.0",
                        "1.0",
                        "1.0",
                        "1.5",
                        "8.0",
                    ],
                ],
            }
        }
    }
    issues = d.check_value_plausibility("GU1")
    if issues:
        for iss in issues:
            print(f"  [{iss['type']}] 周期#{iss['cycle_index']}: {iss['detail']}")
    else:
        print("  (无)")

    print("\n--- 时间点追溯 14:01:30 (值来自周期#0，因为周期#1被取消) ---")
    tb = d.traceback("2026-01-08 14:01:30", expected_values={"GU1": 2.5})
    print(f"  有效G12时间: {tb.get('active_cycle', {}).get('g12_time', 'N/A')}")
    print(f"  存在取消干扰: {tb['cancel_interference_nearby']}")
    if tb.get("cancel_info"):
        print(f"  取消详情: {tb['cancel_info']['detail']}")
    print(f"  附近弯翘事件: {'是' if tb.get('warp_active') else '否'}")
    if tb.get("warp_events_nearby"):
        for we in tb["warp_events_nearby"][:3]:
            print(
                f"     {we.get('time', '')} [{we.get('mode', we.get('type', '?'))}] {we.get('action', '')}"
            )

    print("\n--- 材质一致性检查 ---")
    d.cycles[0]["set_func_event"]["lifecycle"] = {
        "df": {"msg": "(T.-.-.7.J,2400,3B) -> (P.-.-.8.J,2350,3B)", "time": "13:59:50"},
    }
    d.cycles[0]["set_func_event"]["material"] = "P.-.-.8.J"
    mc = d.check_material_consistency()
    if mc:
        for iss in mc:
            print(f"  [{iss['type']}] 周期#{iss['cycle_index']}: {iss['detail']}")
    else:
        print("  (无)")

    report = d.generate_report(
        target_time="2026-01-08 14:01:30", expected_values={"GU1": 2.5}
    )
    with open("diagnostic_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + "=" * 60)
    print("完整报告 (已保存到 diagnostic_report.md)")
    print("=" * 60)

    return d


if __name__ == "__main__":
    from datetime import datetime

    try:
        diagnostic = run_diagnostic_from_db()
        completed = [c for c in diagnostic.cycles if c["end"] == "complete"]
        if completed:
            g12_raw = completed[-1].get("end_event", {}).get("Date", "")
            g12_ts = str(g12_raw).split(".")[0] if "." in str(g12_raw) else str(g12_raw)
            target_time = str(pd.Timestamp(g12_ts) + pd.Timedelta(seconds=30))
        else:
            target_time = None
        report = diagnostic.generate_report(target_time=target_time)
        with open("diagnostic_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n报告已保存到 diagnostic_report.md")
    except Exception as e:
        print(f"数据库测试失败: {e}")
        print("回退到模拟数据进行诊断...\n")
        traceback.print_exc()
        print("\n" + "-" * 60 + "\n")
        diagnostic = run_diagnostic_synthetic()

    print("\n诊断完成。")
