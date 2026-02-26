from smolagents import TransformersModel
from smolagents import CodeAgent, ToolCallingAgent, WebSearchTool, WikipediaSearchTool

model = TransformersModel(
    model_id="Qwen/Qwen3-0.6B",
)

agent_prompt = """
你是专业的日志分析人员，尝试对用户输入的日志进行分析并判断存在哪些问题，
在回答之前进行步骤规划。

接纸机日志中会存在同材剩余米数，该项会在系统运行过程中慢慢变小，
直到需要换纸，换纸后会将新的纸卷剩余米数更新在这一栏中。
如下是用于判断接纸机日志存在问题的条件：
1. 若同材剩余米数在接近0时，没有换纸日志，数值直接跳变为新的剩余米数
2. 同材剩余米数不发生变化

若用户的问题不涉及日志分析，直接回答。
"""

agent = CodeAgent(
    name="Agent",
    description=agent_prompt,
    model=model,
    planning_interval=5,
    max_steps=10,
    tools=[],
)

question = """
同材剩余米数，是否换纸
100, False
83, False
61, False
40, False
29, False
18, False
1, False
0, True
100, False
56, False
19, False
2, False
200, False
180, False
130, False
100, False
100, False
100, False
100, False

回答日志记录是否有问题。
"""

# agent = CodeAgent(tools=[WikipediaSearchTool()], model=model)
result = agent.run(question, reset=True)
print(result)
# print(
#     model(
#         [{"role": "user", "content": [{"type": "text", "text": "Ok!"}]}],
#         stop_sequences=["great"],
#     )
# )
