from langchain_openai import ChatOpenAI, custom_tool
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool

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
    """Execute python code."""
    return "27"


# llm_with_tools = llm.bind_tools([GetWeather])

agent = create_agent(llm, [execute_code])

input_message = {"role": "user", "content": "Use the tool to calculate 3^3."}
for step in agent.stream(
    {"messages": [input_message]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# print(llm.predict("hi!"))

# ai_msg = llm_with_tools.invoke(
#     "what is the weather like in San Francisco",
# )
# print(ai_msg)
