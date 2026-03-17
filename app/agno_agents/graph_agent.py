from agno.agent import Agent
from agno.models.llama_cpp import LlamaCpp
from typing import List
from agno.run.agent import RunOutput
from pydantic import BaseModel, Field
from rich.pretty import pprint  # noqa

intent_agent_system_prompt = """你是一个用户意图分析助手。
用户的提问都是与系统的使用有关的，按照用户的问题对其意图进行分析，并接着判断询问的是哪个确切的模块。
用户意图可以分为三类，分别是界面、操作、流程。
关于界面的意图中，确切的模块有: 生管控制界面、生管总控界面、系统设置界面、QDM基础表界面、基础信息界面、报表管理界面、预警管理界面、工单管理界面、图形化控制界面、系统设置界面；
关于操作的意图中，确切的模块有：追加订单操作、停换操作、刷新操作、强换操作、重传操作；
关于流程的意图中，确切的模块有：停换操作流程

若没有可以匹配的模块，则意图与模块栏都填写为未知。
"""

# Custom server configuration
intent_agent = Agent(
    model=LlamaCpp(
        id="qwen3-4b",
        base_url="http://localhost:8080/v1",  # Custom server URL
    ),
    markdown=True,
    description=intent_agent_system_prompt,
)

intent_output: RunOutput = intent_agent.run("如何进行强换操作？")
print(intent_output.content)
