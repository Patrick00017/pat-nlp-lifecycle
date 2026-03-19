from collections import deque
import pickle
import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any


# ==================== 保存图 ====================
def save_graph(graph, filename="my_graph.pkl"):
    """
    保存图到文件（最简单的方法）

    Args:
        graph: Networkx图对象
        filename: 保存的文件名
    """
    with open(filename, "wb") as f:
        pickle.dump(graph, f)
    print(f"✅ 图已保存到: {filename}")


# ==================== 加载图 ====================
def load_graph(filename="my_graph.pkl"):
    """
    从文件加载图（最简单的方法）

    Args:
        filename: 要加载的文件名

    Returns:
        加载的图对象
    """
    with open(filename, "rb") as f:
        graph = pickle.load(f)
    print(f"✅ 已从 {filename} 加载图")
    print(f"   节点数: {graph.number_of_nodes()}")
    print(f"   边数: {graph.number_of_edges()}")
    return graph


def query_node_relationships(DG: nx.DiGraph, node_name: str) -> Dict[str, List[str]]:
    """
    查询节点的包含关系（contains_arr）和属于关系（related_arr）

    Args:
        DG: Networkx有向图
        node_name: 要查询的节点名称

    Returns:
        包含 contains_arr 和 related_arr 的字典
    """
    if node_name not in DG:
        return {
            "node_name": node_name,
            "contains_arr": [],
            "related_arr": [],
            "error": f"节点 '{node_name}' 不存在",
        }

    contains_arr = []
    related_arr = []

    # 查询所有出边（节点指向其他节点）
    for _, target, data in DG.out_edges(node_name, data=True):
        if data.get("type") == "包含":
            contains_arr.append(target)

    # 查询所有入边（其他节点指向该节点）
    for source, _, data in DG.in_edges(node_name, data=True):
        if data.get("type") == "包含":
            related_arr.append(source)

    return {
        "node_name": node_name,
        "contains_arr": sorted(contains_arr),  # 排序使输出更整齐
        "related_arr": sorted(related_arr),
    }


def query_all_nodes(DG: nx.DiGraph):
    return DG.nodes()


def extract_upward_subgraph(G, start_nodes, relation_type="属于"):
    """
    从起始节点向上抽取子图（沿着关系类型的反方向）

    Args:
        G: 原始有向图
        start_nodes: 起始节点列表
        relation_type: 关系类型

    Returns:
        向上抽取的子图
    """
    # 收集所有需要包含的节点
    nodes_to_include = set(start_nodes)

    # 使用队列进行 BFS
    queue = deque(start_nodes)
    visited = set(start_nodes)

    while queue:
        current = queue.popleft()
        # print(current)

        # 查找所有指向当前节点的边（即出边）
        for successor in G.successors(current):
            # print(successor)
            # 检查这条边的类型是否为指定的关系类型
            edge_data = G.get_edge_data(current, successor)
            if edge_data and edge_data.get("type") == relation_type:
                if successor not in visited:
                    visited.add(successor)
                    nodes_to_include.add(successor)
                    queue.append(successor)

    # 返回子图
    # 打印结果
    subgraph = G.subgraph(nodes_to_include).copy()
    content = ""
    content += "子图中的节点:\n"
    for node in subgraph.nodes():
        node_type = subgraph.nodes[node].get("type", "未知")
        content += f"  - {node} ({node_type})\n"

    content += "子图中的边:\n"
    for u, v, data in subgraph.edges(data=True):
        content += f"  {u} -> {v} (type: {data.get('type')})\n"
    return content


def convert_to_networkx_digraph(json_data):
    """
    将给定的JSON结构转换为NetworkX有向图

    参数:
    json_data: 字典格式的JSON数据，包含nodes和relationships

    返回:
    nx.DiGraph: NetworkX有向图对象
    """
    # 创建有向图
    G = nx.DiGraph()
    id2Caption = {}
    # 记录json中的id2Caption和caption2Id
    for node in json_data["nodes"]:
        node_id = node["id"]
        node_caption = node["caption"]
        id2Caption[node_id] = node_caption
    # 添加节点
    for node in json_data["nodes"]:
        node_id = node["id"]
        # 提取节点属性
        node_attrs = {
            "labels": node.get("labels", []),
            "position": node.get("position", {}),
            "style": node.get("style", {}),
            "properties": node.get("properties", {}),
        }
        G.add_node(id2Caption[node_id], **node_attrs)

    # 添加边（关系）
    for rel in json_data["relationships"]:
        from_id = rel["fromId"]
        to_id = rel["toId"]

        # 提取关系属性
        edge_attrs = {
            "type": rel.get("type", ""),
            "style": rel.get("style", {}),
            "properties": rel.get("properties", {}),
        }
        G.add_edge(id2Caption[from_id], id2Caption[to_id], **edge_attrs)

    return G


def init_graph():
    DG = nx.DiGraph()
    DG.add_node("IPS系统", type="系统")
    DG.add_node("生管控制界面", type="界面")
    DG.add_node("系统设置界面", type="界面")
    DG.add_node("QDM基础表界面", type="界面")
    DG.add_node("基础信息界面", type="界面")
    DG.add_node("报表管理界面", type="界面")
    DG.add_node("预警管理界面", type="界面")
    DG.add_node("工单管理界面", type="界面")
    DG.add_node("图形化控制界面", type="界面")
    DG.add_node("生管总控界面", type="界面")
    DG.add_node("SF1生管界面", type="界面")
    DG.add_node("SF2生管界面", type="界面")
    DG.add_node("追加订单操作", type="操作")
    DG.add_node("停换操作", type="操作")
    DG.add_node("刷新操作", type="操作")
    DG.add_node("强换操作", type="操作")
    DG.add_node("重传操作", type="操作")

    # add edges
    DG.add_edge("IPS系统", "生管控制界面", type="包含")
    DG.add_edge("生管控制界面", "IPS系统", type="属于")
    DG.add_edge("IPS系统", "系统设置界面", type="包含")
    DG.add_edge("系统设置界面", "IPS系统", type="属于")
    DG.add_edge("IPS系统", "QDM基础表界面", type="包含")
    DG.add_edge("QDM基础表界面", "IPS系统", type="属于")
    DG.add_edge("IPS系统", "基础信息界面", type="包含")
    DG.add_edge("基础信息界面", "IPS系统", type="属于")
    DG.add_edge("IPS系统", "报表管理界面", type="包含")
    DG.add_edge("报表管理界面", "IPS系统", type="属于")
    DG.add_edge("IPS系统", "预警管理界面", type="包含")
    DG.add_edge("预警管理界面", "IPS系统", type="属于")
    DG.add_edge("IPS系统", "工单管理界面", type="包含")
    DG.add_edge("工单管理界面", "IPS系统", type="属于")
    DG.add_edge("IPS系统", "图形化控制界面", type="包含")
    DG.add_edge("图形化控制界面", "IPS系统", type="属于")
    DG.add_edge("生管控制界面", "生管总控界面", type="包含")
    DG.add_edge("生管总控界面", "生管控制界面", type="属于")
    DG.add_edge("生管总控界面", "SF1生管界面", type="包含")
    DG.add_edge("SF1生管界面", "生管总控界面", type="属于")
    DG.add_edge("生管总控界面", "SF2生管界面", type="包含")
    DG.add_edge("SF2生管界面", "生管总控界面", type="属于")

    DG.add_edge("生管总控界面", "追加订单操作", type="包含")
    DG.add_edge("追加订单操作", "生管总控界面", type="属于")
    DG.add_edge("生管总控界面", "停换操作", type="包含")
    DG.add_edge("停换操作", "生管总控界面", type="属于")
    DG.add_edge("生管总控界面", "刷新操作", type="包含")
    DG.add_edge("刷新操作", "生管总控界面", type="属于")
    DG.add_edge("生管总控界面", "强换操作", type="包含")
    DG.add_edge("强换操作", "生管总控界面", type="属于")
    DG.add_edge("生管总控界面", "重传操作", type="包含")
    DG.add_edge("重传操作", "生管总控界面", type="属于")
    # save_graph(DG, filename="./bts_graph.pkl")
    start_nodes = ["重传操作", "SF2生管界面"]
    subgraph = extract_upward_subgraph(DG, start_nodes, "属于")
    print(subgraph)
    # 打印结果
    print("子图中的节点:")
    for node in subgraph.nodes():
        node_type = subgraph.nodes[node].get("type", "未知")
        print(f"  - {node} ({node_type})")

    print("\n子图中的边:")
    for u, v, data in subgraph.edges(data=True):
        print(f"  {u} -> {v} (type: {data.get('type')})")


# init_graph()

# nx.draw(DG, with_labels=True)
# plt.show()

# loaded_graph = load_graph("./bts_graph.pkl")
# results = query_node_relationships(loaded_graph, "IPS系统")
# print(results["contains_arr"])
# print(results["related_arr"])


with open("graph.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)
G = convert_to_networkx_digraph(json_data)
# 查看图信息
print(f"节点数: {G.number_of_nodes()}")
print(f"边数: {G.number_of_edges()}")

# 访问节点属性
for node_id, attrs in G.nodes(data=True):
    print(f"节点 {node_id}: {attrs['labels']}")

# 访问边属性
for u, v, attrs in G.edges(data=True):
    print(f"边 {u} -> {v}: {attrs.get('type', '')}")
