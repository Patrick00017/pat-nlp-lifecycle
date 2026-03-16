import json
import re
import os
from pathlib import Path
from typing import Dict, List, Optional
from collections import OrderedDict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_core.documents import Document


def extract_heading_to_text_map(markdown_text: str) -> Dict[str, str]:
    """
    从Markdown文本中提取###级标题及其对应的内容

    Args:
        markdown_text: Markdown格式的文本

    Returns:
        标题到文本的映射字典
    """
    if not markdown_text:
        return {}

    # 匹配###级标题（### 后面跟着标题内容）
    # 支持###后面可能有多个空格
    heading_pattern = re.compile(r"^###\s+(.+?)$", re.MULTILINE)

    # 找到所有标题及其位置
    headings = list(heading_pattern.finditer(markdown_text))

    if not headings:
        return {}

    result = OrderedDict()

    for i, heading_match in enumerate(headings):
        # 获取标题文本
        heading_text = heading_match.group(1).strip()

        # 确定内容开始位置（标题行之后）
        content_start = heading_match.end()

        # 确定内容结束位置
        if i < len(headings) - 1:
            # 到下一个标题开始前
            content_end = headings[i + 1].start()
        else:
            # 到最后
            content_end = len(markdown_text)

        # 提取内容并清理
        content = markdown_text[content_start:content_end].strip()

        # 存储到结果中
        result[heading_text] = content

    return dict(result)  # for title, content in result.items()


model = OllamaLLM(model="qwen3:8b")
embeddings = OllamaEmbeddings(model="llama3")
vector_store = Chroma(
    collection_name="bts_collection",
    embedding_function=embeddings,
    persist_directory="./bts_vector",
)
key2text = {}
intent_agent = None  # this agent is used to identify user's intention
doc_agent = None  # this agent is used to search doc vector and answer
graph_agent = None  # this agent is used to query the neo4j graph database

# system prompts
intent_agent_system_prompt = """你是一个用户意图分析助手。
用户的提问都是与系统的使用有关的，按照用户的问题对其意图进行分析，并接着判断询问的是哪个确切的模块。
用户意图可以分为两类，分别是界面、操作。
关于界面的意图中，确切的模块有: 生管控制界面、生管总控界面、系统设置界面、QDM基础表界面、基础信息界面、报表管理界面、预警管理界面、工单管理界面、图形化控制界面、系统设置界面
关于操作的意图中，确切的模块有：追加订单操作、停换操作、刷新操作、强换操作、重传操作

若没有完全匹配的模块，尽可能选择一个最相关的意图与模块

必须以严格的JSON格式返回所有响应。

## 响应格式要求
- 所有输出必须是有效的JSON对象
- 不要包含任何额外的文本、解释或markdown标记
- 不要使用代码块包裹JSON
- 确保JSON格式正确，可以被直接解析

## JSON结构
{
    "question": "直接摘抄用户问题",
    "intent": "界面或者操作", // 不能判断为其他类别
    "module": "确切的模块"  // 选择上述提到的一个确切的模块，不要输入其他不相关模块
}

## 示例
用户：强换操作如何使用？
助手：{"question": "强换操作如何使用？", "intent": "操作", "module": "强换操作"}

请始终遵循以上格式。"""


@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    # inject context into state messages
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query, k=5)
    docs_content = "\n\n".join(doc.content for doc in retrieved_docs)
    system_prompt = (
        "你是一个用户手册文档助手，该文档是关于BTS开发的IPS智能瓦楞纸板生产线的系统使用手册\n",
        "按照用户的问题，基于查询到的有关文档信息进行回答。\n"
        f"查询到的相关文档如下：{docs_content}",
    )
    return system_prompt


@dynamic_prompt
def prompt_with_neo4j_graph(request: ModelRequest) -> str:
    # as we all know this text is json format
    last_query = request.state["messages"][-1].text
    try:
        parsed_intent_output = json.loads(last_query)
        intent = parsed_intent_output["intent"]
        module = parsed_intent_output["module"]
    except json.JSONDecodeError as e:
        print(f"JSON解析失败，原文：{last_query}")
        intent = ""
        module = ""

    # try to get knowledge graph from neo4j
    # todo

    # create system prompt
    system_prompt = (
        "你是一个用户引导助手，需要基于给出的模块关系引导用户下一步可以进行什么询问。",
        f"模块关系如下: {1}",
        "回答用户，包含当前模块的界面或者操作有哪些，以及当前模块包含的界面或者操作有哪些。",
        "比如：包含A的界面有B和C，在A中包含操作D，如果有其他问题可以继续进行询问。",
    )
    return system_prompt


def init_all():
    # check if chroma vector database is existed
    vector_store_path = Path("./bts_vector")
    if not vector_store_path.exists():
        # no vector database is existed, so create it
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
        markdown_user_manual_path = "user_manual.md"
        # read markdown file
        with open(markdown_user_manual_path, "r", encoding="utf-8") as f:
            content = f.read()

        base_doc = Document(page_content=content)
        user_manual_docs = text_splitter.split_documents([base_doc])
        # now, do embedding
        document_ids = vector_store.add_documents(documents=user_manual_docs)

    # read markdown file and
    key2text = extract_heading_to_text_map(markdown_user_manual_path)
    # create agents
    intent_agent = create_agent(
        model, tools=[], system_prompt=intent_agent_system_prompt
    )
    doc_agent = create_agent(model, tools=[], middleware=[prompt_with_context])
    graph_agent = create_agent(model, tools=[], middleware=[prompt_with_neo4j_graph])


def response(question):
    # 1. intent classification and parse the json output
    intent_output = intent_agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    intent_output = intent_output["messages"][-1].content
    try:
        parsed_intent_output = json.loads(intent_output)
    except json.JSONDecodeError as e:
        print(f"JSON解析失败，原文：{intent_output}")
        return
    intent = parsed_intent_output["intent"]
    module = parsed_intent_output["module"]
    # 2. try to get the right answer in the key2text first,
    #    if can not find the direct answer, use doc agent
    content = ""
    if module in key2text.keys():
        content += f"{key2text[module]}\n"
    else:
        # use doc agent
        doc_agent_output = doc_agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        doc_agent_output = doc_agent_output["messages"][-1].content
        content += doc_agent_output

    # 3. use graph agent to make the graph guide
    graph_agent_output = graph_agent.invoke(
        {"messages": [{"role": "user", "content": intent_output}]}
    )
    graph_agent_output = graph_agent_output["messages"][-1].content
    content += graph_agent_output
