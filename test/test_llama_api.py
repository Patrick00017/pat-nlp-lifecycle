from langchain_openai import ChatOpenAI, custom_tool
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.types import Command


llm = ChatOpenAI(
    temperature=0.5,
    # model="models/mistral-7b-openorca.Q8_0.gguff",
    openai_api_base="http://127.0.0.1:8080/v1",
    openai_api_key="ed",
)


# class GetWeather(BaseModel):
#     """Get the current weather in a given location"""

#     location: str = Field(description="The city and state, e.g. San Francisco, CA")


class CodeInput(BaseModel):
    """Input for weather queries."""

    code: str = Field(description="Code to execute")


@tool(args_schema=CodeInput)
def execute_code(code: str) -> str:
    response = interrupt(
        f"Trying to call `execute_code`. Please approve or suggest edits."
    )
    if response["type"] == "accept":
        pass
    elif response["type"] == "edit":
        funcname = response["args"]["func"]
        print(f"user input funcname: {funcname}")
    else:
        raise ValueError(f"Unknown response type: {response['type']}")
    """Execute python code."""
    return "27"


# llm_with_tools = llm.bind_tools([GetWeather])
checkpointer = InMemorySaver()
agent = create_agent(llm, [execute_code], checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}

input_message = {"role": "user", "content": "Use the tool to calculate 3^3."}
for step in agent.stream({"messages": [input_message]}, config):
    print(step)
    print("\n")

# print(llm.predict("hi!"))

# ai_msg = llm_with_tools.invoke(
#     "what is the weather like in San Francisco",
# )
# print(ai_msg)

human_input = input("y or n or edit?\n")
if human_input == "y":
    for chunk in agent.stream(
        Command(resume={"type": "accept"}),
        # Command(resume={"type": "edit", "args": {"hotel_name": "McKittrick Hotel"}}),
        config,
    ):
        print(chunk)
        print("\n")
elif human_input == "edit":
    for chunk in agent.stream(
        Command(resume={"type": "edit", "args": {"func": "fuck"}}),
        # Command(resume={"type": "edit", "args": {"hotel_name": "McKittrick Hotel"}}),
        config,
    ):
        print(chunk)
        print("\n")
else:
    print("end")
