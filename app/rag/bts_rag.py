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
from networkx_construct import load_graph, query_node_relationships


def extract_h3_headings_v2(markdown_file: str) -> Dict[str, str]:
    """
    逐行解析Markdown文件，提取###标题及其内容
    遇到任何标题（#、##、###等）都视为当前内容的结束
    """
    with open(markdown_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    result = {}
    current_title = None
    current_content = []

    for line in lines:
        line_stripped = line.lstrip()  # 去除开头的空格，但保留缩进

        # 检查是否是任何级别的标题（以#开头）
        if line_stripped.startswith("#"):
            # 如果是###标题，开始新的记录
            if line_stripped.startswith("### "):
                # 如果之前有正在处理的标题，保存它
                if current_title:
                    result[current_title] = "".join(current_content).strip()

                # 开始新的标题
                current_title = line_stripped[4:].strip()  # 去掉'### '前缀
                current_content = []
            else:
                # 如果是其他级别的标题（# 或 ##），且当前正在收集内容
                if current_title:
                    # 保存当前###标题的内容
                    result[current_title] = "".join(current_content).strip()
                    # 重置，不再收集内容（因为新标题不是###）
                    current_title = None
                    current_content = []
                # 如果当前没有在收集内容，什么也不做
        elif current_title:
            # 如果当前在某个###标题下，添加内容
            current_content.append(line)

    # 保存最后一个标题
    if current_title:
        result[current_title] = "".join(current_content).strip()

    return result


model = OllamaLLM(model="qwen3:4b")
embeddings = OllamaEmbeddings(model="llama3:8b")
vector_store = Chroma(
    collection_name="bts_collection",
    embedding_function=embeddings,
    persist_directory="./bts_vector",
)
key2text = {}
intent_agent = None  # this agent is used to identify user's intention
doc_agent = None  # this agent is used to search doc vector and answer
graph_agent = None  # this agent is used to query the neo4j graph database
loaded_graph = None
docs = ""

markdown_user_manual_path = "user_manual.md"
# system prompts
intent_agent_system_prompt = """你是一个用户意图分析助手。
用户的提问都是与系统的使用有关的，按照用户的问题对其意图进行分析，并接着判断询问的是哪个确切的模块。
用户意图可以分为三类，分别是界面、操作、完整流程。
关于界面的意图中，确切的模块有: 生管控制界面、生管总控界面、系统设置界面、QDM基础表界面、基础信息界面、报表管理界面、预警管理界面、工单管理界面、图形化控制界面、系统设置界面；
关于操作的意图中，确切的模块有：追加订单操作、停换操作、刷新操作、强换操作、重传操作；


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
    docs_content = ""
    last_query = request.state["messages"][-1].text
    if docs != "":
        docs_content += f"\n{docs}"

    retrieved_docs = vector_store.similarity_search(last_query, k=2)
    docs_content += "\n".join(doc.content for doc in retrieved_docs)
    system_prompt = (
        "你是一个用户手册文档助手，该文档是关于BTS开发的IPS智能瓦楞纸板生产线的系统使用手册\n",
        "按照用户的问题，严格按照查询到的有关文档信息进行回答。使用的语句尽可能专业。\n",
        "不要有任何在文档以外的内容，不要有任何在文档以外的内容，不要有任何在文档以外的内容。\n"
        f"查询到的相关文档如下：{docs_content}",
    )
    print(f"part2 system prompt: {system_prompt}")
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
    related_results = query_node_relationships(loaded_graph, module)
    # create system prompt
    system_prompt = (
        "你是一个用户引导助手，需要基于给出的模块关系引导用户下一步可以进行什么询问。",
        f"模块的包含与属于关系如下: {related_results}",
        "回答用户，包含当前模块的界面或者操作有哪些，以及当前模块包含的界面或者操作有哪些。",
        "比如：包含A的界面有B和C，在A中包含操作D。",
        "只说明包含与属于关系，不回答任何其他内容",
        "若关系为空，则回答如果有其他问题我可以进行帮助等类似语句，关系不为空时，严格按照给出的关系回答。",
    )
    return system_prompt


def init_all():
    global key2text, intent_agent, doc_agent, graph_agent, loaded_graph
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
        # read markdown file
        with open(markdown_user_manual_path, "r", encoding="utf-8") as f:
            content = f.read()

        base_doc = Document(page_content=content)
        user_manual_docs = text_splitter.split_documents([base_doc])
        # now, do embedding
        _ = vector_store.add_documents(documents=user_manual_docs)

    # load the graph
    loaded_graph = load_graph("./bts_graph.pkl")
    # read markdown file and
    key2text = extract_h3_headings_v2(markdown_user_manual_path)
    print(key2text)
    # create agents
    intent_agent = create_agent(
        model, tools=[], system_prompt=intent_agent_system_prompt
    )
    doc_agent = create_agent(model, tools=[], middleware=[prompt_with_context])
    graph_agent = create_agent(model, tools=[], middleware=[prompt_with_neo4j_graph])


def response(question):
    global docs
    content = ""  # final answer

    # 1. intent classification and parse the json output
    intent_output = intent_agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    intent_output = intent_output["messages"][-1].content
    try:
        parsed_intent_output = json.loads(intent_output)
    except json.JSONDecodeError as e:
        print(f"JSON解析失败，原文：{intent_output}")
        return "没有理解"
    intent = parsed_intent_output["intent"]
    module = parsed_intent_output["module"]
    print(f"part1: {intent} - {module}")
    # 2. try to get the right answer in the key2text first,
    #    if can not find the direct answer, use doc agent
    docs = ""
    if module in key2text.keys():
        print(f"part2 find useful docs: {key2text[module]}")
        docs = key2text[module]
    doc_agent_output = doc_agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    doc_agent_output = doc_agent_output["messages"][-1].content
    print(f"part2 output: {doc_agent_output}")
    content += doc_agent_output

    # 3. use graph agent to make the graph guide
    graph_agent_output = graph_agent.invoke(
        {"messages": [{"role": "user", "content": intent_output}]}
    )
    graph_agent_output = graph_agent_output["messages"][-1].content
    print(f"part3: {graph_agent_output}")
    content += graph_agent_output
    return content


init_all()
text = response("解释一下强换操作")
