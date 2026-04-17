import ast
import json
from utils import load_config
from database_utils import SQLServerHelper
import pandas as pd
from fsm import SplicerLogStateMachineWrapper, KeyEventExtractor
from log_parser import test_ips_and_glue_template
import gradio as gr
import random
import time
from typing import Annotated, Any, Dict, List
from typing_extensions import TypedDict
from llama_tool_wrapper import (
    MyLlamaCppWithTools,
    improved_call_tools,
    improved_should_continue,
)
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from tools_definition import tools
from dotenv import load_dotenv
import os
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

# Get model path from environment variable with fallback
# Hermes-3-Llama-3.2-3B.Q6_K.gguf
# Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf
# qwen2.5-coder-1.5b-instruct-q8_0.gguf
MODEL_PATH = os.getenv("MODEL_PATH", "Hermes-3-Llama-3.2-3B.Q6_K.gguf")
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "20"))
ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", "false").lower() == "true"

# Initialize the local LLaMA model
logger.info(f"Initializing LLM with model: {MODEL_PATH}")
llm = MyLlamaCppWithTools(f"models/{MODEL_PATH}")
llm_with_tools = llm.bind_tools(tools)

conversation_state = {"messages": [], "tool_calls_and_results": []}
response_result = ''

class State(TypedDict):
    """State structure for the chatbot with message history."""
    messages: Annotated[list, add_messages]
    tool_calls_and_results: List[Dict[str, Any]]


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


def call_tools(state: State):
    """
    Execute any tool calls found in the last message.
    Args:
        state (State): The current conversation state.
    Returns:
        dict: Updated state after tool execution.
    """
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


# Build the state graph for managing conversation flow
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", call_tools)

graph_builder.add_edge(START, "chatbot")

graph_builder.add_conditional_edges(
    "chatbot", should_continue, {"tools": "tools", "end": END}
)

graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()


def clean_response(content: str) -> str:
    """
    Remove JSON tool call syntax from final response for clean output.
    Args:
        content (str): Raw response content.
    Returns:
        str: Cleaned response content.
    """
    # Remove JSON code blocks
    cleaned = re.sub(r"```json.*?```", "", content, flags=re.DOTALL).strip()
    # Remove tool_calls JSON structure
    cleaned = re.sub(r'"tool_calls":\s*\[.*?\]', "", cleaned, flags=re.DOTALL).strip()
    # Remove stray braces
    cleaned = re.sub(r"^\s*[{}]\s*$", "", cleaned, flags=re.MULTILINE).strip()

    return cleaned if cleaned else content

def response(message, history):
    global conversation_state
    global response_result

    # check and clean messages
    while len(conversation_state["messages"]) >= 5:
        # remove the oldest message
        conversation_state["messages"].pop(0)
    conversation_state["tool_calls_and_results"] = []
    response_result = ''

    # Exit commands
    if message.lower() in ["quit", "exit", "q"]:
        print("\nGoodbye!")
        response_result = 'Goodbye!'
    # Clear conversation history
    elif message.lower() == "clear":
        conversation_state = {"messages": [], "tool_calls_and_results": []}
        logger.info("Conversation history cleared")
        response_result = "Conversation history cleared"
    else:
        try:
            logger.info(f"Processing user input: {message}")
            # Add user message to conversation state
            conversation_state["messages"].append(HumanMessage(content=message))
            # Invoke the state graph
            result = graph.invoke(conversation_state)

            # for msg in result['messages']:
            #     if not isinstance(msg, HumanMessage):
            #         response_result += f"\r\n{msg.content}"
            response_result += '\r\n## function calls: '
            is_msg_tool_call = True
            for msg in result['tool_calls_and_results']:
                # if isinstance(msg, list):
                #     msg = msg[0]

                # if isinstance(msg, dict):
                #     formatted_msg = json.dumps(msg, indent=2, ensure_ascii=False)
                # elif isinstance(msg, str):
                #     msg = f"{{'result': {msg}}}"
                #     try:
                #         msg = msg.replace("'", '"')
                #         # msg = ast.literal_eval(msg)
                #         msg_dict = json.loads(msg)
                #         formatted_msg = json.dumps(msg_dict, indent=2, ensure_ascii=False)
                #     except:
                #         print("can not convert to json")
                #         formatted_msg = msg
                response_result += f"\r\nfunction call: \r\n ```json \n{msg}\n```" if is_msg_tool_call else f"\r\nfunction result: \r\n {msg}"
                # response_result += f"\r\nfunction call: \r\n {msg}\n" if is_msg_tool_call else f"\r\nfunction result: \r\n {msg}\n"
                is_msg_tool_call = False if is_msg_tool_call else True

            # Update conversation state with results
            conversation_state = result
            logger.debug(f"Graph result: {len(result['messages'])} messages")
            if result["messages"]:
                final_response = result["messages"][-1]
                if isinstance(final_response, AIMessage):
                    content = final_response.content
                    # final_text = clean_response(content)
                    final_text = content
                    # print(f"\nAssistant: {final_text}")
                    response_result += f"\r\n## Answer:"
                    response_result += f"\r\n{final_text}"
                else:
                    # Handle edge case where final message is not AIMessage
                    response_content = (
                        final_response.content
                        if hasattr(final_response, "content")
                        else str(final_response)
                    )
                    # print(f"\nAssistant: {response_content}")
                    response_result += f"\r\n{response_content}"
            else:
                logger.warning("No response received from graph")
                # print("\nNo response received")
                response_result = "No response received"
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            response_result = "Interrupted. Goodbye!"
            # break
        except Exception as e:
            logger.error(f"Error during conversation: {e}", exc_info=True)
            # print(f"\nError: {e}")
            response_result = f"Error: {e}"
            # print("Please try again or type 'quit' to exit")
        finally:
            for i in range(len(response_result)):
                time.sleep(0.0001)
                yield "" + response_result[: i+1]

def yes_man(message, history):
    if message.endswith("?"):
        return "Yes"
    else:
        return "Ask me anything!"




gr.ChatInterface(
    response,
    chatbot=gr.Chatbot(height=600),
    textbox=gr.Textbox(placeholder="Ask me anything.", container=False, scale=7),
    title="Log Analysis",
    description="Ask question about system log",
    examples=[
        "query glue set function events in the system and return events list, start time is 2026-01-08 14:03:50.690, and end time is 2026-01-08 15:03:50.690, and desire material is P.-.-.8.J",
        "track the material P.-.-.8.J lifecycle",
    ],
    save_history=False
    # cache_examples=True,
).launch(theme="ocean")