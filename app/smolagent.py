from smolagents import TransformersModel
from smolagents import ToolCallingAgent, WebSearchTool, WikipediaSearchTool

model = TransformersModel(
    model_id="HuggingFaceTB/SmolLM-135M-Instruct", max_tokens=20480
)

agent = ToolCallingAgent(tools=[WikipediaSearchTool()], model=model)
agent.run("Who is Steve Jobs?")
# print(
#     model(
#         [{"role": "user", "content": [{"type": "text", "text": "Ok!"}]}],
#         stop_sequences=["great"],
#     )
# )
