import json
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from tools_definition import tools


# ------------------ 状态定义 ------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    # 可以添加其他需要持久化的字段


# ------------------ 工具定义 ------------------
# @tool
# def search(query: str) -> str:
#     """搜索互联网，返回相关信息。"""
#     return f"搜索结果：关于 '{query}' 的模拟信息。"


# class CodeInput(BaseModel):
#     """Input for weather queries."""

#     code: str = Field(description="Code to execute")


# @tool(args_schema=CodeInput)
# def execute_code(code: str) -> str:
#     return code
#     # else:
#     #     raise ValueError(f"Unknown response type: {user_decision['approved']}")


# tools = [execute_code]
tool_map = {tool.name: tool for tool in tools}

# ------------------ 初始化模型 ------------------
llm = ChatOpenAI(
    temperature=0,
    # model="models/mistral-7b-openorca.Q8_0.gguff",
    openai_api_base="http://127.0.0.1:8080/v1",
    openai_api_key="ed",
    streaming=True,
)
llm_with_tools = llm.bind_tools(tools)


# checkpointer = InMemorySaver()
# agent = create_agent(llm, [execute_code], checkpointer=checkpointer)


# ------------------ 定义节点 ------------------
def call_model(state: AgentState) -> Dict[str, Any]:
    """调用 LLM，决定下一步动作。"""
    messages = [state["messages"][-1]]
    response = llm_with_tools.invoke(messages)
    # print(f"response: {response}")
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """判断是否调用工具，还是直接结束。"""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "ask_user_approval"  # 进入中断节点
    return END


def ask_user_approval(state: AgentState) -> Dict[str, Any]:
    """
    中断节点：向用户展示待调用的工具，等待用户确认/修改后恢复。
    使用 LangGraph 内置的 interrupt() 实现暂停。
    """
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]

    # interrupt 会暂停图执行，并将数据返回给客户端
    user_decision = interrupt(
        {
            "type": "tool_approval",
            "tool_name": tool_call["name"],
            "tool_args": tool_call["args"],
        }
    )

    # user_decision 由客户端通过 /resume 端点提供，格式为 {"approved": True, "modified_args": {...}}
    if user_decision["approved"]:
        # 使用用户修改后的参数（或原参数）
        new_args = user_decision.get("modified_args", tool_call["args"])
        # 执行工具
        tool_instance = tool_map[tool_call["name"]]
        result = tool_instance.invoke(new_args)
        # 返回 AIMessage，因为tool call直接对结果进行分析
        return {"messages": [AIMessage(content=result)]}
    else:
        # 用户拒绝执行工具
        return {"messages": [AIMessage(content="用户取消了工具调用")]}


# ------------------ 构建图 ------------------
builder = StateGraph(AgentState)
builder.add_node("call_model", call_model)
builder.add_node("ask_user_approval", ask_user_approval)
builder.set_entry_point("call_model")
builder.add_conditional_edges(
    "call_model", should_continue, {"ask_user_approval": "ask_user_approval", END: END}
)
builder.add_edge("ask_user_approval", END)  # 工具执行后直接结束

# # 使用内存检查点保存器（生产环境可换成数据库）
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    input_state = {
        "messages": [
            HumanMessage(
                content="query glue set function events in the system and return events list, start time is 2026-01-08 14:03:50.690, and end time is 2026-01-08 15:03:50.690, and desire material is P.-.-.8.J"
            )
        ]
    }
    for event in graph.stream(
        input_state, config=config, stream_mode=["messages", "values"]
    ):
        print(f"event: {event} \n\n")
        # identify the event type
        event_type = event[0]  # can be messages or values
        if event_type == "messages":
            # go yield this token
            ai_msg_content = event[1][0].content
            print(
                f"data: {json.dumps({'type': 'message', 'content': ai_msg_content})}\n\n"
            )
        elif event_type == "values":
            ai_msg = event[1]["messages"][-1]
            if "__interrupt__" in ai_msg:
                interrupt_data = ai_msg["__interrupt__"][0].value
                print(
                    f"data: {json.dumps({'type': 'interrupt', 'value': interrupt_data})}\n\n"
                )
