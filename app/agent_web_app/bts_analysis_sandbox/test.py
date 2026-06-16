"""
Glue Gap Diagnostic Test
Usage: conda run -n pat-nlp-lifecycle python test.py
"""
import os, sys, traceback
sys.path.insert(0, os.getcwd())
import pandas as pd
from log_parser import LogParser, test_ips_and_glue_template
from glue_gap_diagnostic import GlueGapDiagnostic


def run_diagnostic_from_db():
    """Run diagnostic with real data from database."""
    print("=" * 60)
    print("正在连接数据库并解析日志...")
    print("=" * 60)

    extractor = test_ips_and_glue_template(
        start_time="2026-01-08 14:03:50.690",
        end_time="2026-01-08 15:03:50.690"
    )

    print(f"\n匹配到 {len(extractor.raw_parsed_rows)} 个G事件")
    print(f"材质变更事件: {len(extractor.material_events)}")
    print(f"赋值函数调用: {len(extractor.set_func_call_events)}")

    diagnostic = GlueGapDiagnostic(extractor)

    print("\n" + "=" * 60)
    print("周期汇总")
    print("=" * 60)
    print(diagnostic.print_cycle_summary())

    print("\n" + "=" * 60)
    print("取消率统计")
    print("=" * 60)
    cr = diagnostic.calc_cancellation_rate()
    labels_cr = {
        'total_cycles': '总周期数', 'g7_starts': 'G7触发', 'g11_starts': 'G11触发',
        'completed': '完成', 'cancelled_pre_write': '写值前取消', 'cancelled_delay': '延迟取消',
        'interrupted': '中断', 'cancellation_rate': '取消率', 'alert': '警告'
    }
    for k, v in cr.items():
        label = labels_cr.get(k, k)
        print(f"  {label}: {v}")

    print("\n" + "=" * 60)
    print("完整性异常")
    print("=" * 60)
    type_labels_anom = {
        'no_termination': '无终止事件', 'fallback_used': '降级匹配',
        'g12_no_g14': 'G12无G14', 'pre_write_cancel_no_calc': '写值取消无计算',
        'excessive_calculation': '过多计算',
    }
    anomalies = diagnostic.check_cycle_completeness()
    if anomalies:
        for a in anomalies:
            anom_label = type_labels_anom.get(a['type'], a['type'])
            print(f"  [{anom_label}] 周期#{a['cycle_index']}: {a['detail']}")
    else:
        print("  (无)")

    print("\n" + "=" * 60)
    print("值合理性检查 (GU1)")
    print("=" * 60)
    issues = diagnostic.check_value_plausibility('GU1')
    if issues:
        for iss in issues:
            print(f"  [{iss['type']}] 周期#{iss['cycle_index']}: {iss['detail']}")
    else:
        print("  (无)")

    print("\n" + "=" * 60)
    print("材质一致性检查")
    print("=" * 60)
    mc_issues = diagnostic.check_material_consistency()
    if mc_issues:
        for iss in mc_issues:
            print(f"  [{iss['type']}] 周期#{iss['cycle_index']}: {iss['detail']}")
    else:
        print("  (无)")

    if extractor.raw_parsed_rows:
        mid_time = extractor.raw_parsed_rows[len(extractor.raw_parsed_rows) // 2].get('Date', '')
        if isinstance(mid_time, pd.Timestamp):
            mid_time = str(mid_time)

        print(f"\n" + "=" * 60)
        print(f"时间点追溯: {mid_time}")
        print("=" * 60)
        tb = diagnostic.traceback(str(mid_time))
        print(f"  存在有效G12: {tb['has_active_g12']}")
        print(f"  存在取消干扰: {tb['cancel_interference_nearby']}")
        if tb.get('cancel_info'):
            print(f"  取消详情: {tb['cancel_info']['detail']}")

    return diagnostic


def run_diagnostic_synthetic():
    """Run diagnostic with synthetic test data (no DB needed)."""
    print("=" * 60)
    print("使用模拟数据进行诊断")
    print("=" * 60)

    from event_extractor import GlueEventExtractor
    ext = GlueEventExtractor()
    ext.raw_parsed_rows = [
        {'EventId': 'G1', 'ParsedValues': {'handle_func_name': 'HandleGuGlueMsg', 'meterial': 'P.-.-.8.J', 'flute_type': '3B'}, 'Date': '2026-01-08 14:00:00'},
        {'EventId': 'G7', 'ParsedValues': {'set_func_name': 'SetGlueGu', 'material': 'P.-.-.8.J', 'flute_type': '3B'}, 'Date': '2026-01-08 14:00:01'},
        {'EventId': 'G14', 'ParsedValues': {'glue_part': 'GU1'}, 'Date': '2026-01-08 14:00:05'},
        {'EventId': 'G14', 'ParsedValues': {'glue_part': 'GU2'}, 'Date': '2026-01-08 14:00:06'},
        {'EventId': 'G12', 'ParsedValues': {'set_func_name': 'SetGlueGu', 'material': 'P.-.-.8.J', 'flute_type': '3B'}, 'Date': '2026-01-08 14:00:10'},
        {'EventId': 'G7', 'ParsedValues': {'set_func_name': 'SetGlueGu', 'material': 'P.-.-.8.J', 'flute_type': '3B'}, 'Date': '2026-01-08 14:01:00'},
        {'EventId': 'G14', 'ParsedValues': {'glue_part': 'GU1'}, 'Date': '2026-01-08 14:01:05'},
        {'EventId': 'G15', 'ParsedValues': {}, 'Date': '2026-01-08 14:01:08'},
        {'EventId': 'G7', 'ParsedValues': {'set_func_name': 'SetGlueGu', 'material': 'T.-.-.7.J', 'flute_type': 'B'}, 'Date': '2026-01-08 14:02:00'},
        {'EventId': 'G8', 'ParsedValues': {}, 'Date': '2026-01-08 14:02:02'},
        {'EventId': 'G10', 'ParsedValues': {}, 'Date': '2026-01-08 14:02:03'},
        {'EventId': 'G14', 'ParsedValues': {'glue_part': 'GU1'}, 'Date': '2026-01-08 14:02:05'},
        {'EventId': 'G14', 'ParsedValues': {'glue_part': 'GU3'}, 'Date': '2026-01-08 14:02:06'},
        {'EventId': 'G8', 'ParsedValues': {}, 'Date': '2026-01-08 14:02:07'},
        {'EventId': 'G14', 'ParsedValues': {'glue_part': 'GU1'}, 'Date': '2026-01-08 14:02:08'},
        {'EventId': 'G15', 'ParsedValues': {}, 'Date': '2026-01-08 14:02:09'},
    ]
    d = GlueGapDiagnostic(ext)

    print("\n--- 周期汇总 ---")
    print(d.print_cycle_summary())

    print("\n--- 取消率 ---")
    cr = d.calc_cancellation_rate()
    labels_cr = {
        'total_cycles': '总周期数', 'g7_starts': 'G7触发', 'g11_starts': 'G11触发',
        'completed': '完成', 'cancelled_pre_write': '写值前取消', 'cancelled_delay': '延迟取消',
        'interrupted': '中断', 'cancellation_rate': '取消率', 'alert': '警告'
    }
    for k, v in cr.items():
        label = labels_cr.get(k, k)
        print(f"  {label}: {v}")
    if cr.get('alert'):
        print("  ⚠ 取消率超过30%")

    print("\n--- 完整性异常 ---")
    type_labels_anom = {
        'no_termination': '无终止事件', 'fallback_used': '降级匹配',
        'g12_no_g14': 'G12无G14', 'pre_write_cancel_no_calc': '写值取消无计算',
        'excessive_calculation': '过多计算',
    }
    anomalies = d.check_cycle_completeness()
    if anomalies:
        for a in anomalies:
            anom_label = type_labels_anom.get(a['type'], a['type'])
            print(f"  [{anom_label}] 周期#{a['cycle_index']}: {a['detail']}")
    else:
        print("  (无)")

    print("\n--- 糊间隙值合理性检查 (GU1) ---")
    d.cycles[0]['set_func_event'] = {
        'set_values': {
            'GU1': {
                'columns': ['speed', 'min_glue', 'max_glue', 'min_weight', 'max_weight', 'current_glue_weight', 'speed_factor', 'min_speed', 'qdm_factor', 'ui_factor', 'value'],
                'data': [
                    ['50', '1', '5', '100', '200', '150', '0.9', '1.0', '1.0', '1.0', '2.0'],
                    ['100', '1', '5', '100', '200', '150', '0.9', '1.0', '1.0', '1.0', '3.5'],
                    ['150', '1', '5', '100', '200', '150', '1.2', '1.0', '1.0', '1.0', '8.0'],
                ]
            }
        }
    }
    issues = d.check_value_plausibility('GU1')
    if issues:
        for iss in issues:
            print(f"  [{iss['type']}] 周期#{iss['cycle_index']}: {iss['detail']}")
    else:
        print("  (无)")

    print("\n--- 时间点追溯 14:01:30 (值来自周期#0，因为周期#1被取消) ---")
    tb = d.traceback('2026-01-08 14:01:30', expected_values={'GU1': 2.5})
    print(f"  有效G12时间: {tb.get('active_cycle', {}).get('g12_time', 'N/A')}")
    print(f"  存在取消干扰: {tb['cancel_interference_nearby']}")
    if tb.get('cancel_info'):
        print(f"  取消详情: {tb['cancel_info']['detail']}")

    print("\n--- 材质一致性检查 ---")
    d.cycles[0]['set_func_event']['lifecycle'] = {
        'df': {'msg': '(T.-.-.7.J,2400,3B) -> (P.-.-.8.J,2350,3B)', 'time': '13:59:50'},
    }
    d.cycles[0]['set_func_event']['material'] = 'P.-.-.8.J'
    mc = d.check_material_consistency()
    if mc:
        for iss in mc:
            print(f"  [{iss['type']}] 周期#{iss['cycle_index']}: {iss['detail']}")
    else:
        print("  (无)")

    report = d.generate_report(target_time='2026-01-08 14:01:30', expected_values={'GU1': 2.5})
    with open('diagnostic_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print("\n" + "=" * 60)
    print("完整报告 (已保存到 diagnostic_report.md)")
    print("=" * 60)

    return d


if __name__ == '__main__':
    from datetime import datetime
    try:
        diagnostic = run_diagnostic_from_db()
        report = diagnostic.generate_report()
        with open('diagnostic_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n报告已保存到 diagnostic_report.md")
    except Exception as e:
        print(f"数据库测试失败: {e}")
        print("回退到模拟数据进行诊断...\n")
        traceback.print_exc()
        print("\n" + "-" * 60 + "\n")
        diagnostic = run_diagnostic_synthetic()

    print("\n诊断完成。")
