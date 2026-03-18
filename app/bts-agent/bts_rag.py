import json
import re
import os
from langchain_community.chat_models import ChatLlamaCpp
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
from networkx_construct import (
    load_graph,
    query_node_relationships,
    query_all_nodes,
    extract_upward_subgraph,
)


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
# model = ChatLlamaCpp(
#     temperature=0.5,
#     model_path="D:/code/btsagent/models/Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf",
#     n_ctx=10000,
#     n_gpu_layers=8,
#     n_batch=300,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
#     max_tokens=512,
#     repeat_penalty=1.5,
#     top_p=0.5,
#     verbose=True,
# )
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
graph_fetch_agent = None
graph_path_summarize_agent = None
docs = ""

markdown_user_manual_path = "user_manual.md"
# system prompts
intent_agent_system_prompt = """你是一个用户意图分析助手。
用户的提问都是与系统的使用有关的，按照用户的问题对其意图进行分析，并接着判断询问的是哪个确切的模块。
用户意图可以分为三类，分别是界面、操作、流程。
关于界面的意图中，确切的模块有: 生管控制界面、生管总控界面、系统设置界面、QDM基础表界面、基础信息界面、报表管理界面、预警管理界面、工单管理界面、图形化控制界面、系统设置界面；
关于操作的意图中，确切的模块有：停换操作、刷新操作、强换操作、重传操作；
关于流程的意图中，确切的模块有：停换操作流程

若没有可以匹配的模块，则用于意图与确切的模块都填写为未知。
当用户的问题比较复杂时，意图与确切的模块都填写未知。

必须以严格的JSON格式返回所有响应。

## 响应格式要求
- 所有输出必须是有效的JSON对象
- 不要包含任何额外的文本、解释或markdown标记
- 不要使用代码块包裹JSON
- 确保JSON格式正确，可以被直接解析

## JSON结构
{
    "question": "直接摘抄用户问题",
    "intent": "界面、操作、流程或未知", // 不能判断为其他类别
    "module": "确切的模块或未知"  // 选择上述提到的一个确切的模块，不要输入其他不相关模块
}

## 示例1
用户：强换操作如何使用？
助手：{"question": "强换操作如何使用？", "intent": "操作", "module": "强换操作"}

## 示例2
用户：停换操作的完成流程是怎样的？
助手：{"question": "停换操作的完成流程是怎样的？", "intent": "流程", "module": "停换操作流程"}

请始终遵循以上格式。"""


@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """
    用于不用思考流程的RAG部分
    """
    # inject context into state messages
    docs_content = ""
    last_query = request.state["messages"][-1].text
    if docs != "":
        docs_content += f"\n{docs}"

    retrieved_docs = vector_store.similarity_search(last_query, k=2)
    print(f"使用向量数据库查询到的文档有：{len(retrieved_docs)}")
    docs_content += "\n".join(doc.page_content for doc in retrieved_docs)
    system_prompt = f"""
你是一个用户手册文档助手，该文档是关于BTS开发的IPS智能瓦楞纸板生产线的系统使用手册。
按照用户的问题，严格按照查询到的有关文档信息进行回答。使用的语句尽可能专业。
不要有任何在文档以外的内容，不要有任何在文档以外的内容，不要有任何在文档以外的内容。
查询到的相关文档如下：{docs_content}
"""
    print(f"part2 system prompt: {system_prompt}")
    return system_prompt


@dynamic_prompt
def prompt_with_neo4j_graph(request: ModelRequest) -> str:
    """
    用于不用思考流程的Graph部分
    """
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

    related_results = query_node_relationships(loaded_graph, module)
    # {
    #     "node_name": node_name,
    #     "contains_arr": sorted(contains_arr),  # 排序使输出更整齐
    #     "related_arr": sorted(related_arr),
    # }
    # create system prompt
    system_prompt = f"""
你是一个用户引导助手，需要基于给出的模块关系进行总结。
模块的包含与属于关系如下: {related_results}
其中，contains_arr中为当前模块包含的子模块，related_arr中为包含当前模块的上层模块。
只对上述给出的关系进行总结，不要有任何其他内容，若无任何关系，则说明无任何关系
    """
    return system_prompt


@dynamic_prompt
def prompt_fetch_useful_graph(request: ModelRequest) -> str:
    """
    用于头脑风暴流程的抓取有用知识图节点
    """
    all_nodes = query_all_nodes(loaded_graph)
    # {
    #     "node_name": node_name,
    #     "contains_arr": sorted(contains_arr),  # 排序使输出更整齐
    #     "related_arr": sorted(related_arr),
    # }
    # create system prompt
    system_prompt = f"""
你是一个用户问题分析助手。
用户的提问都是与系统的使用有关的，对用户的问题进行分析，并接着判断问题需要使用到系统的哪个部分。
现已经使用图谱的形式将系统之间的关系进行了建模，你只需要对节点进行抽取，判断用户的问题需要使用哪些节点，将所有节点名称填入nodes字段中。
填入的节点名称不可以重复。图中所有节点名称如下：{all_nodes}

必须以严格的JSON格式返回所有响应。

## 响应格式要求
- 所有输出必须是有效的JSON对象
- 不要包含任何额外的文本、解释或markdown标记
- 不要使用代码块包裹JSON
- 确保JSON格式正确，可以被直接解析

## JSON结构
{{
    "nodes": ["node1", "node2"], // 不能判断为其他类别
}}

请始终遵循以上格式。
    """
    return system_prompt


def init_all():
    global key2text, intent_agent, doc_agent, graph_agent, loaded_graph, graph_fetch_agent, graph_path_summarize_agent

    # check if chroma vector database is existed
    vector_store_path = Path("./bts_vector")
    if not vector_store_path.exists() or len(os.listdir(vector_store_path)) == 0:
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
        base_doc = [Document(page_content=content)]
        print(f"Total characters: {len(base_doc[0].page_content)}")
        user_manual_docs = text_splitter.split_documents(base_doc)
        print(f"Split blog post into {len(user_manual_docs)} sub-documents.")
        # now, do embedding
        _ = vector_store.add_documents(documents=user_manual_docs)

    # load the graph
    loaded_graph = load_graph("./bts_graph.pkl")
    # read markdown file and
    key2text = extract_h3_headings_v2(markdown_user_manual_path)
    # create agents
    intent_agent = create_agent(
        model, tools=[], system_prompt=intent_agent_system_prompt
    )
    doc_agent = create_agent(model, tools=[], middleware=[prompt_with_context])
    graph_agent = create_agent(model, tools=[], middleware=[prompt_with_neo4j_graph])

    # brain storm agent
    graph_fetch_agent = create_agent(
        model, tools=[], middleware=[prompt_fetch_useful_graph]
    )
    graph_path_summarize_agent = create_agent(
        model,
        tools=[],
        system_prompt="""
        你是一个系统使用培训助手。
        现已将操作相关的部分以图的方式呈现，使用通俗易懂的语句，对图结构进行总结，只需要总结图，不要说明图以外的任何信息，以免对用户产生误导。
        可以适当说明操作的前后顺序，几大板块的操作优先级如下：
        生管控制界面 --> 低 
        系统设置界面 --> 高
        QDM基础表界面 --> 中
        基础信息界面 --> 中
        报表管理界面 --> 低
        预警管理界面 --> 低
        工单管理界面 --> 低
        图形化控制界面 --> 低

        板块中的子节点都遵顼这一优先级。
    """,
    )


def response(question):
    """
                            unknown intent or complicated question
                        --------------------------------------------> plan agent(brain storm agent) -----> fetch useful knowledge graph ------> summarize the graph based on manual
                        |
                        |            known intent
    question -----> intent router ------------------> rag agent(normal agent) ----------> graph agent: get adj node to show the little part of the knowledge graph

    """
    global docs
    content = ""  # final answer

    #  1. intent classification and parse the json output
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
    if intent != "未知":
        # Normal stage: 2. try to get the right answer in the key2text first,
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
        # Normal stage: 3. use graph agent to make the graph guide
        graph_agent_output = graph_agent.invoke(
            {"messages": [{"role": "user", "content": intent_output}]}
        )
        graph_agent_output = graph_agent_output["messages"][-1].content
        print(f"part3: {graph_agent_output}")
        content += graph_agent_output
        return content
    else:
        # brain storm stage 1: fetch the useful knowledge graph nodes
        graph_nodes_output = graph_fetch_agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        graph_nodes_output = graph_nodes_output["messages"][-1].content
        try:
            parsed_graph_nodes_output = json.loads(graph_nodes_output)
        except json.JSONDecodeError as e:
            print(f"JSON解析失败，原文：{graph_nodes_output}")
            return "没有理解"
        print(f"brain storm stage 1: {parsed_graph_nodes_output}")
        content += f"\n抽取到的可能的操作为：{parsed_graph_nodes_output}"
        # brain storm stage 2: make the summary for useful knowledge graph nodes
        # query the useful sub graph
        subgraph_content = extract_upward_subgraph(
            loaded_graph, parsed_graph_nodes_output["nodes"], relation_type="属于"
        )
        print(f"brain storm stage 2 subgraph: {subgraph_content}")
        graph_path_summary = graph_path_summarize_agent.invoke(
            {"messages": [{"role": "user", "content": subgraph_content}]}
        )
        print(f"brain storm stage 2 output: {graph_path_summary}")
        content += f"\n{graph_path_summary}"
        return content


init_all()

# question1 = "解释一下强换操作"
# ans1 = response(question1)
# print(f"问题：{question1}")
# print(f"回答：{ans1}")

# question2 = "生管总控界面是什么"
# ans2 = response(question2)
# print(f"问题：{question2}")
# print(f"回答：{ans2}")

question1 = "如何进行订单修改"
ans1 = response(question1)
print(f"问题：{question1}")
print(f"回答：{ans1}")
