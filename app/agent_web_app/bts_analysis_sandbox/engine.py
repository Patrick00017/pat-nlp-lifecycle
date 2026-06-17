"""
动态节点分析引擎
核心：AnalysisNode + AnalysisEngine + FastAPI Router
节点可动态注册，每个节点是一个分析步骤，支持 DAG 链式调用。
"""

import sys, os

_sandbox = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _sandbox not in sys.path:
    sys.path.insert(0, _sandbox)

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from cachetools import TTLCache

# ── 状态模型 ──


class AnalysisState(BaseModel):
    state_id: str
    start_time: str
    end_time: str
    target_time: Optional[str] = None
    material: Optional[str] = None
    tool_name: str
    diagnostic_result: Optional[Dict[str, Any]] = None
    analysis_results: Dict[str, Any] = {}
    execution_path: List[str] = []


# ── 节点定义 ──


@dataclass
class AnalysisNode:
    id: str
    name: str
    description: str
    fn: Callable  # (state: AnalysisState) -> str
    next_nodes: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "next_nodes": self.next_nodes,
        }


# ── 引擎容器 ──


class AnalysisEngine:
    def __init__(self, ttl: int = 1800):
        self._nodes: Dict[str, AnalysisNode] = {}
        self._entry_nodes: Dict[str, str] = {}
        self._states: TTLCache[str, AnalysisState] = TTLCache(maxsize=128, ttl=ttl)

    # ── 注册 / 卸载 ──

    def register_node(
        self, node: AnalysisNode, is_entry: bool = False, tool_name: str = ""
    ):
        if node.id in self._nodes:
            raise ValueError(f"Node '{node.id}' already registered")
        self._nodes[node.id] = node
        if is_entry and tool_name:
            self._entry_nodes[tool_name] = node.id

    def register_nodes(
        self, *nodes: AnalysisNode, is_entry: bool = False, tool_name: str = ""
    ):
        for node in nodes:
            self.register_node(
                node, is_entry=(is_entry and nodes[0] is node), tool_name=tool_name
            )

    def unregister_node(self, node_id: str):
        self._nodes.pop(node_id, None)
        for tool, nid in list(self._entry_nodes.items()):
            if nid == node_id:
                del self._entry_nodes[tool]

    # ── 查询 ──

    def get_node(self, node_id: str) -> AnalysisNode:
        node = self._nodes.get(node_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
        return node

    def get_entry_node(self, tool_name: str) -> AnalysisNode:
        nid = self._entry_nodes.get(tool_name)
        if not nid:
            available = list(self._entry_nodes.keys())
            raise HTTPException(
                status_code=400,
                detail=f"No entry for '{tool_name}', available: {available}",
            )
        return self.get_node(nid)

    def list_nodes(self) -> List[dict]:
        return [n.to_dict() for n in self._nodes.values()]

    # ── 状态管理 ──

    def create_state(
        self,
        tool_name: str,
        start_time: str,
        end_time: str,
        target_time: Optional[str] = None,
        material: Optional[str] = None,
    ) -> str:
        state_id = str(uuid.uuid4())
        self._states[state_id] = AnalysisState(
            state_id=state_id,
            start_time=start_time,
            end_time=end_time,
            target_time=target_time,
            material=material,
            tool_name=tool_name,
        )
        return state_id

    def get_state(self, state_id: str) -> AnalysisState:
        state = self._states.get(state_id)
        if not state:
            raise HTTPException(
                status_code=404, detail=f"State '{state_id}' not found or expired"
            )
        return state

    # ── 步骤执行 ──

    def step(self, state_id: str, node_id: str) -> dict:
        state = self.get_state(state_id)
        node = self.get_node(node_id)

        result = node.fn(state)
        state.analysis_results[node_id] = result
        state.execution_path.append(node_id)

        next_nodes = [self.get_node(nid).to_dict() for nid in node.next_nodes]

        return {
            "result": result,
            "available_nodes": next_nodes,
            "execution_path": state.execution_path,
            "is_terminal": len(node.next_nodes) == 0,
        }

    def stepback(self, state_id: str) -> dict:
        state = self.get_state(state_id)
        if not state.execution_path:
            raise HTTPException(status_code=400, detail="No steps to step back from")

        undone_id = state.execution_path.pop()
        state.analysis_results.pop(undone_id, None)

        if state.execution_path:
            last_id = state.execution_path[-1]
            last_node = self.get_node(last_id)
            next_nodes = [self.get_node(nid).to_dict() for nid in last_node.next_nodes]
        else:
            entry = self.get_entry_node(state.tool_name)
            next_nodes = [entry.to_dict()]

        previous = (
            state.analysis_results.get(state.execution_path[-1], "")
            if state.execution_path
            else ""
        )

        return {
            "undone_node": undone_id,
            "previous_result": previous,
            "execution_path": state.execution_path,
            "available_nodes": next_nodes,
            "is_terminal": len(next_nodes) == 0,
        }


# ── 引擎实例（全局单例） ──

engine = AnalysisEngine()

# ── 内置节点注册 ──


def _build_diagnostic(start_time, end_time, dev_ips=None):
    from glue_gap_diagnostic import GlueGapDiagnostic

    return GlueGapDiagnostic.from_params(start_time, end_time, dev_ips=dev_ips)


def _glue_diagnose(state: AnalysisState) -> str:
    from database_utils import PostgreSQLHelper

    try:
        dev_ips = PostgreSQLHelper.from_connection_string(
            "PORT=5432;DATABASE=devIPS;HOST=192.168.110.82;PASSWORD=123456;USER ID=postgres"
        )
        dev_ips.connect()
    except Exception:
        dev_ips = None

    diagnostic = _build_diagnostic(state.start_time, state.end_time, dev_ips=dev_ips)

    anom = diagnostic.check_cycle_completeness()
    cr = diagnostic.calc_cancellation_rate()
    mc = diagnostic.check_material_consistency()
    diagnostic_result = diagnostic.generate_json(target_time=state.target_time)

    state.diagnostic_result = diagnostic_result
    state.material = (
        diagnostic_result.get("cycles", [{}])[0].get("material", "")
        if diagnostic_result.get("cycles")
        else ""
    )

    errors = sum(1 for c in diagnostic_result.get("cycles", []) if c.get("errors"))
    warns = sum(1 for c in diagnostic_result.get("cycles", []) if c.get("warnings"))
    return (
        f"诊断完成：共 {cr['total_cycles']} 个周期，{cr['completed']} 个完成，"
        f"{len(anom)} 项异常，{errors} 个周期有确认错误，{warns} 个周期有警告"
    )


def _glue_detail(state: AnalysisState) -> str:
    data = state.diagnostic_result
    if not data:
        return "请先运行 glue_diagnose"
    parts = []
    for c in data.get("cycles", []):
        idx = c["index"]
        status = c["status"]["label"]
        trigger = c["trigger"]["label"]
        material = c["material"]
        parts.append(f"### 周期 #{idx} ({status})")
        parts.append(f"- **触发**: {trigger}")
        parts.append(f"- **材质**: {material}")
        if c.get("computed_values"):
            for layer, segs in c["computed_values"].items():
                vals = " / ".join(f"@{s['speed']}={s['value']}" for s in segs)
                parts.append(f"- **计算值**: {layer}: {vals}")
        for err in c.get("errors", []):
            parts.append(f"- ❌ **{err['label']}**：{err['detail']}")
        for w in set(c.get("warnings", [])):
            parts.append(f"- ⚠ **{w}**")
        for info in c.get("infos", []):
            parts.append(f"- ℹ {info}")
        parts.append("")
    return "\n".join(parts) if parts else "无周期数据"


def _glue_report(state: AnalysisState) -> str:
    from glue_gap_diagnostic import GlueGapDiagnostic

    if state.diagnostic_result:
        start = state.start_time
        end = state.end_time
        diagnostic = _build_diagnostic(start, end)
        return diagnostic.generate_report(target_time=state.target_time)
    return "请先运行 glue_diagnose"


def _glue_assignments(state: AnalysisState) -> str:
    data = state.diagnostic_result
    if not data:
        return "请先运行 glue_diagnose"
    ra = data.get("recent_assignments", {})
    events = ra.get("events", [])
    if not events:
        return "无最近赋值事件数据（需指定 target_time）"

    parts = []
    parts.append(f"### 最近赋值事件序列")
    parts.append(f"目标时间: {ra.get('target_time', '')}")
    parts.append("")
    for e in events:
        active = " *生效" if e.get("is_active") else ""
        label = f"{e['t_label']} (#{e['cycle_index']}){active}"
        layers = ", ".join(e.get("layers", [])) if e.get("layers") else "-"
        vals = " / ".join(e.get("values", [])) if e.get("values") else "-"
        parts.append(f"- **{label}** {e.get('end_time', '')}  {layers}  {e['material']}  `{vals}`")
        if e.get("anomalies"):
            parts.append(f"  - 异常: {', '.join(e['anomalies'])}")
        if e.get("error_detail") and e["error_detail"] != "-":
            parts.append(f"  - 错误: {e['error_detail']}")
        parts.append("")

    conclusion = ra.get("conclusion", {})
    if conclusion.get("has_errors"):
        parts.append("**结论：发现了问题**")
        parts.append("")
        for ec in conclusion.get("cycles_with_errors", []):
            sep = "；"
            parts.append(f"- 周期 #{ec['index']} 存在以下错误：{sep.join(ec['labels'])}")
    else:
        parts.append("**结论：** 这几次赋值都没有发现任何问题，数据正常")

    return "\n".join(parts)


def register_glue_nodes(engine_instance: AnalysisEngine):
    engine_instance.register_nodes(
        AnalysisNode(
            "glue_diagnose", "糊间隙诊断", "运行全部诊断检查并生成结构化结果",
            _glue_diagnose,
            next_nodes=["glue_detail", "glue_assignments", "glue_report"],
        ),
        AnalysisNode(
            "glue_detail", "周期详情", "每个周期的计算值、错误、警告、跨来源一致性",
            _glue_detail,
            next_nodes=["glue_report"],
        ),
        AnalysisNode(
            "glue_assignments", "最近赋值事件", "目标时间前最近的赋值事件序列及结论",
            _glue_assignments,
            next_nodes=["glue_report"],
        ),
        AnalysisNode(
            "glue_report", "诊断报告", "生成完整诊断报告",
            _glue_report,
        ),
        is_entry=True,
        tool_name="glue_gap_diagnostic",
    )
    engine_instance._entry_nodes["get_glue_set_func_call_in_log"] = "glue_diagnose"


register_glue_nodes(engine)


# ── FastAPI Router ──

router = APIRouter(prefix="/analysis")


@router.post("/init")
async def init_analysis(body: dict):
    tool_name = body.get("tool_name")
    tool_args = body.get("tool_args", {})
    args = body.get("args", {})
    start_time = body.get("start_time")
    end_time = body.get("end_time")
    target_time = body.get("target_time") or args.get("target_time")
    material = body.get("material") or args.get("material")

    # 兼容旧格式：从 tool_args 中提取 start_time/end_time
    if not start_time and tool_args.get("time"):
        from utils import parse_time_flexible
        from datetime import timedelta

        center = parse_time_flexible(tool_args["time"])
        start_time = (center - timedelta(minutes=60)).strftime("%Y-%m-%d %H:%M:%S")[:-3]
        end_time = (center + timedelta(minutes=60)).strftime("%Y-%m-%d %H:%M:%S")[:-3]

    if not material and tool_args.get("desire_material"):
        material = tool_args["desire_material"]

    if not tool_name or not start_time or not end_time:
        raise HTTPException(
            status_code=400, detail="tool_name, start_time, end_time are required"
        )

    state_id = engine.create_state(
        tool_name, start_time, end_time, target_time, material
    )
    entry = engine.get_entry_node(tool_name)

    return {
        "state_id": state_id,
        "context": {
            "start_time": start_time,
            "end_time": end_time,
            "material": material,
        },
        "available_nodes": [entry.to_dict()],
    }


@router.post("/step")
async def step_analysis(body: dict):
    state_id = body.get("state_id")
    node_id = body.get("node_id")
    args = body.get("args", {})
    if not state_id or not node_id:
        raise HTTPException(status_code=400, detail="state_id and node_id are required")

    state = engine.get_state(state_id)

    if args.get("target_time"):
        state.target_time = args["target_time"]

    if node_id == "glue_assignments" and state.target_time:
        diagnostic = _build_diagnostic(state.start_time, state.end_time)
        result = diagnostic.generate_json(target_time=state.target_time)
        state.diagnostic_result = result

    return engine.step(state_id, node_id)


@router.post("/stepback")
async def stepback_analysis(body: dict):
    state_id = body.get("state_id")
    if not state_id:
        raise HTTPException(status_code=400, detail="state_id is required")
    return engine.stepback(state_id)


@router.get("/state/{state_id}")
async def get_state(state_id: str):
    return engine.get_state(state_id)


@router.get("/nodes")
async def list_nodes():
    return {"nodes": engine.list_nodes()}
