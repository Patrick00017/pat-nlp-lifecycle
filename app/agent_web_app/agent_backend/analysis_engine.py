import uuid
from typing import Any, Callable, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from cachetools import TTLCache


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
    return f"[事件列表] 胶量设定事件 ({state.start_time} ~ {state.end_time})：\n  - 10:01:00 → 胶量设至 120g/m²\n  - 10:15:00 → 胶量设至 115g/m²\n  - 10:30:00 → 胶量设至 125g/m²"

def _glue_trend(state: AnalysisState) -> str:
    return f"[趋势分析] 胶量变化趋势：\n  - 初始: 120g/m²\n  - 最低: 115g/m²\n  - 最高: 125g/m²\n  - 结论: 胶量呈现小幅波动，趋势稳定"

def _glue_cross(state: AnalysisState) -> str:
    return f"[关联分析] 胶量设定与换材关联：\n  - 10:01 胶量变更 → 10:02 材料切换为 P\n  - 10:15 胶量变更 → 10:16 材料切换为 N"

def _glue_report(state: AnalysisState) -> str:
    prev = state.analysis_results.get(state.execution_path[-2], "") if len(state.execution_path) >= 2 else ""
    return f"[胶量分析报告] 基于以下分析生成：\n{prev}"

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
    "glue_list":   AnalysisNode("glue_list", "事件列表", "展示胶量设定事件", "get_glue_set_func_call_in_log", _glue_list, ["glue_trend", "glue_cross"]),
    "glue_trend":  AnalysisNode("glue_trend", "趋势分析", "分析胶量设定值的变化趋势", "get_glue_set_func_call_in_log", _glue_trend, ["glue_report"]),
    "glue_cross":  AnalysisNode("glue_cross", "关联分析", "关联胶量设定与换材事件", "get_glue_set_func_call_in_log", _glue_cross, ["glue_report"]),
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
        "start_time": args.get("time"),
        "end_time": args.get("time"),
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

    return {
        "undone_node": undone_node_id,
        "execution_path": state.execution_path,
        "available_nodes": next_nodes,
        "is_terminal": len(next_nodes) == 0,
    }


@router.get("/state/{state_id}")
async def get_analysis_state(state_id: str):
    if state_id not in states:
        raise HTTPException(status_code=404, detail=f"State '{state_id}' not found")
    return states[state_id]
