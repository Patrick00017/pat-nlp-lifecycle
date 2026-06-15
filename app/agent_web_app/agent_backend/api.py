import json
import uuid
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    BaseMessage,
    AIMessageChunk,
    ToolMessageChunk,
)
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from langgraph.types import interrupt, Command
from langgraph.errors import GraphInterrupt
from ips_log_agent import graph as ips_log_agent
from constant import LLAMA_SERVER_URL, FIXED_TOOLS
from utils import parse_function_calls
from analysis_engine import router as analysis_router

# from rag_agent import rag_tool_agent
from fastapi.sse import EventSourceResponse, ServerSentEvent

# from rag_agent import response as rag_response

# ------------------ FastAPI 应用 ------------------
app = FastAPI(title="LangGraph Agent with Interrupt")

# 设置允许的源（Origin）
origins = ["http://localhost:5173", "http://localhost", "http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 1. 允许的源列表
    allow_credentials=True,  # 2. 是否允许携带Cookie
    allow_methods=["*"],  # 3. 允许的HTTP方法（*代表全部）
    allow_headers=["*"],  # 4. 允许的请求头（*代表全部）
)

# 会话存储（线程 ID -> 配置）
sessions: Dict[str, Dict] = {}


class ChatRequest(BaseModel):
    thread_id: Optional[str] = None  # 会话 ID，若为空则新建
    message: str


class ResumeRequest(BaseModel):
    thread_id: str
    approved: bool = True
    modified_args: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    thread_id: str
    response: str
    interrupt: Optional[Dict] = None  # 如果中断发生，返回中断信息


class FuncQueryRequest(BaseModel):
    messages: list
    max_tokens: Optional[int] = 128


class FuncCallRequest(BaseModel):
    tool_calls: list


class SimpleRequest(BaseModel):
    message: str


class SimpleResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """发送消息到 Agent。"""
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # 准备输入
    input_state = {"messages": [HumanMessage(content=request.message)]}

    # 运行图，可能因 interrupt 暂停
    try:
        # stream 方式获取最终状态
        events = list(
            ips_log_agent.stream(input_state, config=config, stream_mode="values")
        )
        if not events:
            raise HTTPException(status_code=500, detail="No output from graph")
        final_state = events[-1]
        last_message = final_state["messages"][-1]

        if "__interrupt__" in final_state:
            return ChatResponse(
                thread_id=thread_id,
                response="",
                interrupt=final_state["__interrupt__"][0].value,
            )
        else:
            return ChatResponse(
                thread_id=thread_id,
                response=(
                    last_message.content if hasattr(last_message, "content") else ""
                ),
                interrupt=None,
            )
    except Exception as e:
        # 检查是否是中断异常（LangGraph 的 GraphInterrupt）
        print(e)


@app.post("/resume", response_model=ChatResponse)
async def resume(request: ResumeRequest):
    print(request)
    """继续被中断的会话。"""
    thread_id = request.thread_id
    config = {"configurable": {"thread_id": thread_id}}

    # 构造恢复命令
    resume_value = {
        "approved": request.approved,
        "modified_args": request.modified_args,
    }

    try:
        # 使用 Command(resume=...) 恢复
        events = list(
            ips_log_agent.stream(
                Command(resume=resume_value), config=config, stream_mode="values"
            )
        )
        if not events:
            raise HTTPException(status_code=500, detail="No output from graph")
        final_state = events[-1]
        last_message = final_state["messages"][-1]

        return ChatResponse(
            thread_id=thread_id,
            response=last_message.content if hasattr(last_message, "content") else "",
            interrupt=None,
        )
    except GraphInterrupt as e:
        interrupt_data = e.args[0] if e.args else {}
        return ChatResponse(thread_id=thread_id, response="", interrupt=interrupt_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/rag", response_model=SimpleResponse)
# async def rag(request: SimpleRequest):
#     """发送消息到 Agent。"""

#     # 运行图，可能因 interrupt 暂停
#     try:
#         content, no_think_content = rag_response(request.message)
#         return SimpleResponse(response=no_think_content)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error.{e}")


# @app.post("/rag/tool", response_class=EventSourceResponse)
# async def rag_tool(request: SimpleRequest):
#     """发送消息到 Agent，使用 Server-Sent Events 流式返回。"""
#     input_state = {"messages": [HumanMessage(content=request.message)]}
#     try:
#         for event in rag_tool_agent.stream(
#             input_state, stream_mode=["messages", "values"]
#         ):
#             # identify the event type
#             if isinstance(event[1][0], AIMessageChunk):
#                 event_type = event[0]  # can be messages or values
#                 if event_type == "messages":
#                     # go yield this token
#                     is_reason = event[1][0].additional_kwargs.get("reason", False)
#                     ai_msg_content = event[1][0].content
#                     if is_reason:
#                         yield f"data: {json.dumps({'type': 'reason', 'content': ai_msg_content})}\n\n"
#                     else:
#                         yield f"data: {json.dumps({'type': 'message', 'content': ai_msg_content})}\n\n"
#             elif isinstance(event[1][0], ToolMessage):
#                 ai_msg_content = event[1][0].content
#                 yield f"data: {json.dumps({'type': 'docs', 'content': ai_msg_content})}\n\n"
#         yield f"data: {json.dumps({'type': 'done'})}\n\n"
#     except Exception as e:
#         yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"


@app.post("/chat/stream", response_class=EventSourceResponse)
async def chat_stream(request: ChatRequest):
    """发送消息到 Agent，使用 Server-Sent Events 流式返回。"""
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    input_state = {"messages": [HumanMessage(content=request.message)]}

    yield f"data: {json.dumps({'type': 'thread_id', 'value': thread_id})}\n\n"
    try:
        for event in ips_log_agent.stream(
            input_state, config=config, stream_mode=["messages", "values"]
        ):
            # identify the event type
            event_type = event[0]  # can be messages or values
            if event_type == "messages":
                # go yield this token
                is_reason = event[1][0].additional_kwargs.get("reason", False)
                ai_msg_content = event[1][0].content
                if is_reason:
                    yield f"data: {json.dumps({'type': 'reason', 'content': ai_msg_content})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'message', 'content': ai_msg_content})}\n\n"
            elif event_type == "values":
                ai_msg = event[1]
                if "__interrupt__" in ai_msg:
                    interrupt_data = ai_msg["__interrupt__"][0].value
                    yield f"data: {json.dumps({'type': 'interrupt', 'value': interrupt_data})}\n\n"
                    return
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"


@app.post("/resume/stream", response_class=EventSourceResponse)
async def resume_stream(request: ResumeRequest):
    """继续被中断的会话。"""
    thread_id = request.thread_id
    config = {"configurable": {"thread_id": thread_id}}

    # 构造恢复命令
    resume_value = {
        "approved": request.approved,
        "modified_args": request.modified_args,
    }

    try:
        # 使用 Command(resume=...) 恢复
        for event in ips_log_agent.stream(
            Command(resume=resume_value),
            config=config,
            stream_mode=["messages", "values"],
        ):
            # identify the event type
            event_type = event[0]  # can be messages or values
            if event_type == "messages":
                # go yield this token
                is_reason = event[1][0].additional_kwargs.get("reason", False)
                ai_msg_content = event[1][0].content
                if is_reason:
                    yield f"data: {json.dumps({'type': 'reason', 'content': ai_msg_content})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'message', 'content': ai_msg_content})}\n\n"
            elif event_type == "values":
                ai_msg = event[1]
                if "__interrupt__" in ai_msg:
                    interrupt_data = ai_msg["__interrupt__"][0].value
                    yield f"data: {json.dumps({'type': 'interrupt', 'value': interrupt_data})}\n\n"
                    return
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"


@app.get("/tool/list")
async def list_tools():
    """返回所有可用工具列表"""
    from ips_log_agent import tool_map

    tools = []
    for name, tool in tool_map.items():
        schema = {}
        if tool.args_schema and hasattr(tool.args_schema, "model_fields"):
            schema = {
                field_name: str(field_info.annotation)
                .replace("<class '", "")
                .replace("'>", "")
                .replace("typing.", "")
                for field_name, field_info in tool.args_schema.model_fields.items()
            }
            tools.append(
                {"name": name, "description": tool.description, "schema": schema}
            )

    return {"tools": tools}


@app.post("/func/query")
async def func_query(request: FuncQueryRequest):
    import httpx

    payload = {
        "messages": request.messages,
        "tools": FIXED_TOOLS,
        "tool_choice": "auto",
        "max_tokens": request.max_tokens,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(LLAMA_SERVER_URL, json=payload)

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    llm_response = resp.json()
    content = llm_response["choices"][0]["message"]["content"]
    tool_calls = parse_function_calls(content)

    return {"tool_calls": tool_calls, "content": content}


@app.post("/func/call")
async def func_call(request: FuncCallRequest):
    from tools_definition import tools

    tool_map = {t.name: t for t in tools}
    results = []
    for tc in request.tool_calls:
        name = tc["name"]
        args = tc.get("arguments", {})
        tool = tool_map.get(name)
        if tool is None:
            results.append({"name": name, "error": f"Tool '{name}' not found"})
        else:
            try:
                result = tool.invoke(args)
                results.append({"name": name, "result": str(result)})
            except Exception as e:
                results.append({"name": name, "error": str(e)})
    return {"results": results}


app.include_router(analysis_router)

from opencode_router import router as opencode_router
app.include_router(opencode_router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
