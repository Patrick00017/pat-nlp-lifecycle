from llama_tool_wrapper import (
    MyLlamaCppWithTools,
    improved_call_tools,
    improved_should_continue,
)
from langgraph.types import interrupt
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from tools_definition import tools
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated, Literal
import os
import re
import logging
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


MAX_ITERATIONS = 3


class State(TypedDict):
    """State structure for the chatbot with message history."""

    messages: Annotated[list, add_messages]
    approved: str


MODEL_PATH = "Qwen3-4B-Q4_K_M.gguf"
llm = MyLlamaCppWithTools(f"D:/code/gguf-models/qwen3-4b/{MODEL_PATH}")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    """
    Generate a response using the LLM with tool support.
    Args:
        state (State): The current conversation state, containing messages.
    Returns:
        dict: Updated state containing the assistant's message.
    """
    try:
        logger.debug(f"Chatbot processing: {len(state['messages'])} messages")
        response = llm_with_tools.invoke(state["messages"])
        logger.debug(f"LLM response: {response.content[:100]}...")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Chatbot error: {e}", exc_info=True)
        return {
            "messages": [AIMessage(content="I encountered an error. Please try again.")]
        }


def approval_node(state: State):
    # Pause and ask for approval
    approved = interrupt("Do you approve this action?")

    # When you resume, Command(resume=...) returns that value here
    return {"approved": approved}


def call_tools(state: State):
    """
    Execute any tool calls found in the last message.
    Args:
        state (State): The current conversation state.
    Returns:
        dict: Updated state after tool execution.
    """
    print(state)
    return improved_call_tools(state, llm_with_tools, tools)


def should_continue(state: State) -> str:
    """
    Determine if the conversation should continue with a tool call or end.
    Args:
        state (State): The current conversation state.
    Returns:
        str: 'tools' if tools need to be called, 'end' to finish conversation.
    """
    return improved_should_continue(
        state, llm_with_tools, max_iterations=MAX_ITERATIONS
    )


workflow = StateGraph(State)

workflow.add_node("chatbot", chatbot)
workflow.add_node("check", approval_node)
workflow.add_node("tools", call_tools)

workflow.add_edge(START, "chatbot")
workflow.add_edge("check", "tools")

workflow.add_conditional_edges(
    "chatbot", should_continue, {"check": "check", "end": END}
)

workflow.add_edge("tools", "chatbot")

# Set up memory
memory = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable

# We add in `interrupt_before=["action"]`
# This will add a breakpoint before the `action` node is called
app = workflow.compile(checkpointer=memory)

display(Image(app.get_graph().draw_mermaid_png()))

from langchain_core.messages import HumanMessage

configs = {"configurable": {"thread_id": "3"}}
inputs = [HumanMessage(content="1981乘9是多少，使用工具给出结果")]
# for event in app.stream({"messages": inputs}, configs, stream_mode="values"):
#     event["messages"][-1].pretty_print()

results = app.invoke({"messages": inputs}, configs, version="v2")

print(f"中断信息: {results.interrupts}")

ans = app.invoke(Command(resume=True), config=configs, version="v2")
print(ans)
