"""
Glue Gap Diagnostic Test
Usage: conda run -n pat-nlp-lifecycle python test\test.py
"""

import sys, os, traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from log_parser import LogParser, test_ips_and_glue_template, test_wrap_template
from glue_gap_diagnostic import GlueGapDiagnostic
from database_utils import PostgreSQLHelper


def run_diagnostic_from_db(source="mssql"):
    """Run diagnostic with real data from database."""
    print("=" * 60)
    print(f"正在连接数据库并解析日志 (数据源: {source})...")
    print("=" * 60)

    start_time = "2026-06-05 12:03:50.690"
    end_time = "2026-06-08 16:03:50.690"

    if source == "postgresql":
        diagnostic = GlueGapDiagnostic.from_params(
            start_time, end_time, source="postgresql"
        )
    else:
        extractor = test_ips_and_glue_template(start_time=start_time, end_time=end_time)
        diagnostic = GlueGapDiagnostic(extractor)

    print(f"\n匹配到 {len(diagnostic.extractor.raw_parsed_rows)} 个G事件")
    print(f"材质变更事件: {len(diagnostic.extractor.material_events)}")
    print(f"赋值函数调用: {len(diagnostic.extractor.set_func_call_events)}")

    # print("\n正在查询弯翘数据...")
    # diagnostic.warp_extractor = test_wrap_template(
    #     start_time=start_time, end_time=end_time
    # )
    # ws = diagnostic.warp_extractor.get_summary()
    # print(f"弯翘事件总数: {ws['total_warp_events']}")
    # print(f"自动调平: {ws['auto_adjust_count']}, 手动调平: {ws['manual_adjust_count']}")
    # print(f"复位: {ws['reset_count']}, 换材跟踪: {ws['paper_change_count']}")

    print("\n正在连接 devIPS 数据库...")
    diagnostic.dev_ips = PostgreSQLHelper.from_connection_string(
        "PORT=5432;DATABASE=devIPS;HOST=192.168.110.82;PASSWORD=123456;USER ID=postgres"
    )
    try:
        diagnostic.dev_ips.connect()
    except Exception as e:
        print(f"devIPS 连接失败: {e}，跨来源一致性检查将跳过")
        diagnostic.dev_ips = None

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
    print(f"共匹配 {len(diagnostic.extractor.raw_parsed_rows)} 条事件")

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
    # if wp_issues:
    #     max_off = max(
    #         abs(float(i["detail"].split("最大=")[-1].split(")")[0]))
    #         for i in wp_issues
    #         if "最大=" in i["detail"]
    #     )
    #     print(f"{idx}. [警告] 弯翘调平正在影响胶水赋值")
    #     print(f"   弯翘偏移量最大为 {max_off}，涉及 {len(wp_issues)} 个部位")
    #     print(f"   → 建议：检查弯翘模块的调平记录（WARP事件），确认偏移量是否合理")
    #     print()
    #     idx += 1
    # else:
    #     print(f"{idx}. [信息] 本次分析未发现弯翘调平的影响")
    #     print(f"   所有胶水计算中的弯翘偏移量均为0")
    #     print()
    #     idx += 1

    # 5) 材质一致性
    if mc_issues:
        print(f"{idx}. [信息] 材质变更记录不一致")
        for mi in mc_issues:
            print(f"   · {mi['detail']}")
        print()
        idx += 1

    # ── 周期详细报告（合并周期概览 + 计算值 + 错误 + 警告） ──
    print("--- 周期详细报告 ---")
    print()
    cs_all = diagnostic.check_cross_source_consistency()
    error_types = (
        "material_mismatch",
        "weight_mismatch",
        "qdm_mismatch",
        "qdm_no_data",
        "base_setting_mismatch",
    )
    warn_types = (
        "fallback_used",
        "warp_influence",
        "excessive_calculation",
        "value_jump",
        "negative_value",
        "exceeds_hard_limit",
        "speed_not_monotonic",
    )
    info_types = ("no_termination", "g12_no_g14", "pre_write_cancel_no_calc")
    trig_labels = {"G7": "换材触发", "G11": "立即换材"}

    for c in diagnostic.cycles:
        sfe = c.get("set_func_event")
        material = sfe.get("material", "-") if sfe else "-"
        eid = c["start"].get("EventId", "?")
        trig = f"{eid}（{trig_labels.get(eid, '其他触发')}）"
        end_labels = {
            "complete": "完成",
            "cancelled_pre_write": "写值取消",
            "interrupted": "中断",
        }
        cl = end_labels.get(c["end"], str(c["end"]))
        t = str(c["start"].get("Date", ""))[:19]
        print(f"周期 #{c['index']} ({cl})  {trig}  {t}  材质={material}")

        # Computed values
        if c["end"] == "complete" and sfe:
            sv = sfe.get("set_values", {})
            for layer, ld in sv.items():
                data = ld.get("data", [])
                cols = ld.get("columns", [])
                if not data:
                    continue
                try:
                    si = cols.index("speed")
                    vi = cols.index("value")
                    segs = [
                        f"@{r[si]}={r[vi]}" for r in data if si < len(r) and vi < len(r)
                    ]
                    if segs:
                        print(f"  计算值: {layer}: {' / '.join(segs)}")
                except ValueError:
                    continue

        # Errors (确认错误)
        errs = []
        for a in anomalies:
            if a["cycle_index"] == c["index"] and a["type"] in error_types:
                tag_map = {
                    "material_mismatch": "材质不匹配",
                    "fallback_used": "降级匹配",
                    "weight_mismatch": "克重不匹配",
                    "qdm_mismatch": "QDM系数不匹配",
                    "qdm_no_data": "QDM无配置",
                    "base_setting_mismatch": "基础设置不匹配",
                }
                errs.append(f"{tag_map.get(a['type'], a['type'])}（{a['detail']}）")
        for cs in cs_all:
            if cs["cycle_index"] == c["index"] and cs["type"] in error_types:
                tag_map = {
                    "weight_mismatch": "克重不匹配",
                    "qdm_mismatch": "QDM系数不匹配",
                    "qdm_no_data": "QDM无配置",
                    "base_setting_mismatch": "基础设置不匹配",
                }
                errs.append(f"{tag_map.get(cs['type'], cs['type'])}（{cs['detail']}）")
        if errs:
            print(f"  错误: {'; '.join(errs)}")

        # Warnings
        warns = []
        for a in anomalies:
            if a["cycle_index"] == c["index"] and a["type"] in warn_types:
                tag_map = {
                    "fallback_used": "降级匹配",
                    "warp_influence": "弯翘影响",
                    "excessive_calculation": "重复计算",
                    "value_jump": "值跳变",
                    "negative_value": "负值",
                    "exceeds_hard_limit": "超硬限制",
                    "speed_not_monotonic": "车速不单调",
                }
                warns.append(tag_map.get(a["type"], a["type"]))
        for wp in wp_issues:
            if wp["cycle_index"] == c["index"]:
                warns.append("弯翘影响")
        if warns:
            print(f"  警告: {'; '.join(set(warns))}")

        # Info
        infos = []
        for a in anomalies:
            if a["cycle_index"] == c["index"] and a["type"] in info_types:
                tag_map = {
                    "no_termination": "被抢断",
                    "g12_no_g14": "写值完成但缺少计算过程",
                    "pre_write_cancel_no_calc": "写值取消且没有计算记录",
                }
                infos.append(tag_map.get(a["type"], a["type"]))
        if infos:
            print(f"  信息: {'; '.join(infos)}")

        print()

    # -- Root Cause Traceback: Recent Assignment Events --
    completed_indices = [
        c["index"] for c in diagnostic.cycles if c["end"] == "complete"
    ]
    if completed_indices:
        last_completed = diagnostic.cycles[completed_indices[-1]]
        g12_raw = last_completed.get("end_event", {}).get("Date", "")
        g12_ts = str(g12_raw).split(".")[0] if "." in str(g12_raw) else str(g12_raw)
        target = str(pd.Timestamp(g12_ts) + pd.Timedelta(seconds=30))
        # target = "2026-01-08 17:50:10"
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
                                " / ".join(f"@{s['speed']}={s['value']}" for s in segs)
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

            # ── 结论 ──
            if ra:
                cs_all_console = diagnostic.check_cross_source_consistency()
                error_cycles = []
                for ra_item in ra:
                    idx = ra_item["index"]
                    labels = []
                    seen = set()
                    if "材质不匹配" in ra_item.get("anomalies", []):
                        labels.append("材质和系统记录对不上")
                        seen.add("material_mismatch")
                    for cs in cs_all_console:
                        if cs["cycle_index"] == idx and cs["type"] not in seen:
                            seen.add(cs["type"])
                            cs_plain = {
                                "weight_mismatch": "实际克重和档案不一致",
                                "qdm_mismatch": "QDM系数和配方不一致",
                                "qdm_no_data": "QDM配方没找到对应配置",
                                "base_setting_mismatch": "糊间隙基础参数设定对不上",
                                "speed_coef_mismatch": "车速系数和数据库对不上",
                            }
                            labels.append(cs_plain.get(cs["type"], cs["type"]))
                    if labels:
                        error_cycles.append((idx, labels))
                if error_cycles:
                    print("结论: 发现了问题")
                    sep = "；"
                    for idx, labels in error_cycles:
                        print(f"  周期 #{idx} 存在以下错误：{sep.join(labels)}")
                else:
                    print("结论: 这几次赋值都没有发现任何问题，数据正常")
            print()

    print("--- 后续操作建议 ---")
    print()
    print(
        "  · 如果发现确认错误（材质不匹配、克重不一致等），建议核实对应周期的原材料信息和设备基础参数"
    )
    print("  · 如果取消率过高（>30%），建议检查生产排程是否过于密集")
    print("  · 如果出现 QDM 无配置或基础设置不匹配，建议检查 IPS 档案配置")
    print("  · 完整技术细节请查阅 diagnostic_report.md")
    print("  · 结构化数据请使用 diagnostic_data.json（供前端展示）")

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

    target_time = None
    try:
        diagnostic = run_diagnostic_from_db(source="postgresql")
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

    import json

    data = diagnostic.generate_json(target_time=target_time)
    with open("diagnostic_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("JSON 数据已保存到 diagnostic_data.json")

    # ── FSM 引擎测试 ──
    try:
        from fsm_engine import GlueGapDiagnosticFSM

        fsm = GlueGapDiagnosticFSM(diagnostic.extractor)
        fsm.run()
        fsm_data = fsm.generate_json()
        print(f"\nFSM 引擎运行完成：{len(fsm_data.get('cycles', []))} 个周期")
        for c in fsm_data.get("cycles", []):
            print(f"  {c['position']}#{c['index']} {c['status']['id']} {c['trigger']['label']} mat={c['material']}")
            if c.get("errors"):
                for e in c["errors"]:
                    print(f"    错误: {e['label']}: {e['detail']}")
            if c.get("warnings"):
                print(f"    警告: {', '.join(set(c['warnings']))}")

        fsm_report = fsm.generate_report()
        with open("fsm_report.md", "w", encoding="utf-8") as f:
            f.write(fsm_report)
        print("FSM 报告已保存到 fsm_report.md")
    except Exception as e:
        print(f"FSM 引擎测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n诊断完成。")
