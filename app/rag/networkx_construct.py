import pickle

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
    save_graph(DG, filename="./bts_graph.pkl")


# init_graph()

# nx.draw(DG, with_labels=True)
# plt.show()

loaded_graph = load_graph("./bts_graph.pkl")
results = query_node_relationships(loaded_graph, "IPS系统")
print(results["contains_arr"])
print(results["related_arr"])
