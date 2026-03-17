from agno.agent import Agent
from agno.models.llama_cpp import LlamaCpp
from agno.tools.hackernews import HackerNewsTools
from agno.tools.calculator import CalculatorTools

agent = Agent(
    model=LlamaCpp(
        id="qwen3-4b",
        base_url="http://localhost:8080/v1",  # Custom server URL
    ),
    tools=[CalculatorTools()],
    markdown=True,
)
agent.print_response("What is the answer of 3*19*100/5? Use the tool to answer.")
