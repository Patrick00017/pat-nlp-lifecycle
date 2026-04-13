import json
import uuid
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

from langgraph.errors import GraphInterrupt


# ------------------ 状态定义 ------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    # 可以添加其他需要持久化的字段


# ------------------ 工具定义 ------------------
# @tool
# def search(query: str) -> str:
#     """搜索互联网，返回相关信息。"""
#     return f"搜索结果：关于 '{query}' 的模拟信息。"


class CodeInput(BaseModel):
    """Input for weather queries."""

    code: str = Field(description="Code to execute")


@tool(args_schema=CodeInput)
def execute_code(code: str) -> str:
    tool_args = {"code": code}
    # interrupt 会暂停图执行，并将数据返回给客户端
    user_decision = interrupt(
        {"type": "tool_approval", "tool_name": "execute_code", "tool_args": tool_args}
    )
    if user_decision["approved"]:
        # 使用用户修改后的参数（或原参数）
        new_args = user_decision.get("modified_args", tool_args)
        # 执行工具
        # Execute python code.
        return f"执行代码：{new_args}"
    else:
        raise ValueError(f"Unknown response type: {user_decision['approved']}")


tools = [execute_code]
tool_map = {tool.name: tool for tool in tools}

# ------------------ 初始化模型 ------------------
llm = ChatOpenAI(
    temperature=0.5,
    # model="models/mistral-7b-openorca.Q8_0.gguff",
    openai_api_base="http://127.0.0.1:8080/v1",
    openai_api_key="ed",
)

checkpointer = InMemorySaver()
agent = create_agent(llm, [execute_code], checkpointer=checkpointer)


# ------------------ 定义节点 ------------------
# def call_model(state: AgentState) -> Dict[str, Any]:
#     """调用 LLM，决定下一步动作。"""
#     messages = state["messages"]
#     response = model_with_tools.invoke(messages)
#     return {"messages": [response]}


# def should_continue(state: AgentState) -> str:
#     """判断是否调用工具，还是直接结束。"""
#     last_message = state["messages"][-1]
#     if isinstance(last_message, AIMessage) and last_message.tool_calls:
#         return "ask_user_approval"  # 进入中断节点
#     return END


# def ask_user_approval(state: AgentState) -> Dict[str, Any]:
#     """
#     中断节点：向用户展示待调用的工具，等待用户确认/修改后恢复。
#     使用 LangGraph 内置的 interrupt() 实现暂停。
#     """
#     last_message = state["messages"][-1]
#     tool_call = last_message.tool_calls[0]

#     # interrupt 会暂停图执行，并将数据返回给客户端
#     user_decision = interrupt(
#         {
#             "type": "tool_approval",
#             "tool_name": tool_call["name"],
#             "tool_args": tool_call["args"],
#             "tool_id": tool_call["id"],
#         }
#     )

#     # user_decision 由客户端通过 /resume 端点提供，格式为 {"approved": True, "modified_args": {...}}
#     if user_decision["approved"]:
#         # 使用用户修改后的参数（或原参数）
#         new_args = user_decision.get("modified_args", tool_call["args"])
#         # 执行工具
#         tool_instance = tool_map[tool_call["name"]]
#         result = tool_instance.invoke(new_args)
#         # 返回 ToolMessage
#         return {"messages": [ToolMessage(content=result, tool_call_id=tool_call["id"])]}
#     else:
#         # 用户拒绝执行工具
#         return {
#             "messages": [
#                 ToolMessage(content="用户取消了工具调用", tool_call_id=tool_call["id"])
#             ]
#         }


# def call_tool(state: AgentState) -> Dict[str, Any]:
#     """实际执行工具（备用，本例通过 ask_user_approval 直接执行）。"""
#     # 可根据需要调整
#     pass


# ------------------ 构建图 ------------------
# builder = StateGraph(AgentState)
# builder.add_node("call_model", call_model)
# builder.add_node("ask_user_approval", ask_user_approval)
# builder.set_entry_point("call_model")
# builder.add_conditional_edges(
#     "call_model", should_continue, {"ask_user_approval": "ask_user_approval", END: END}
# )
# builder.add_edge("ask_user_approval", "call_model")  # 工具执行后继续调用模型

# # 使用内存检查点保存器（生产环境可换成数据库）
# memory = MemorySaver()
# graph = builder.compile(checkpointer=memory)

# ------------------ FastAPI 应用 ------------------
app = FastAPI(title="LangGraph Agent with Interrupt")

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
        events = list(agent.stream(input_state, config=config, stream_mode="values"))
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
            agent.stream(
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


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
