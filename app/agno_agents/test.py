from agno.agent import Agent
from agno.models.llama_cpp import LlamaCpp

# Custom server configuration
agent = Agent(
    model=LlamaCpp(
        id="qwen3-4b",
        base_url="http://localhost:8080/v1",  # Custom server URL
    ),
    markdown=True,
)

agent.print_response("Share a 2 sentence horror story")
