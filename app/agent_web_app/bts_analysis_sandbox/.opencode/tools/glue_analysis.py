import sys
import os
import json
import traceback

# Locate sandbox root: script is at .opencode/tools/glue_analysis.py
script_dir = os.path.dirname(os.path.abspath(__file__))
sandbox_root = os.path.dirname(os.path.dirname(script_dir))
os.chdir(sandbox_root)
sys.path.insert(0, sandbox_root)

try:
    from log_parser import test_from_csv
    from event_extractor import GlueEventExtractor
    from fsm_engine import GlueGapDiagnosticFSM

    extractor = test_from_csv()
    all_events = extractor.get_all_events()
    set_func_events = extractor.get_glue_set_function_full_event()

    fsm = GlueGapDiagnosticFSM(extractor)
    fsm.run()
    fsm.get_results()

    output = {
        "status": "ok",
        "all_events_count": len(all_events),
        "set_func_events_count": len(set_func_events),
        "details": "FSM 执行完成，详细结果请查看 fsm_results.json / diagnostic_data.json",
    }
    print(json.dumps(output, ensure_ascii=False))

except Exception as e:
    error_output = {
        "status": "error",
        "error": str(e),
        "traceback": traceback.format_exc(),
    }
    print(json.dumps(error_output, ensure_ascii=False))
    sys.exit(1)