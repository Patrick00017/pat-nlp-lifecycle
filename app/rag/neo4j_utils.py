from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional, Union


class Neo4jConnection:
    """
    Neo4j数据库连接管理器
    """

    def __init__(
        self,
        uri: str = "neo4j+s://b66832ee.databases.neo4j.io",
        user: str = "b66832ee",
        password: str = "zEBeiOJTatBJ1MPKq_N7Onj_G8Xz6XIcbJ9UMZCGCBs",
        database: str = "b66832ee",
    ):
        """
        初始化Neo4j连接

        Args:
            uri: 连接URI
            user: 用户名
            password: 密码
            database: 数据库名
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self._connect()

    def _connect(self):
        """建立连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            # 测试连接
            with self.driver.session() as session:
                session.run("RETURN 1")
            print(f"成功连接到Neo4j数据库: {self.uri}")
        except Exception as e:
            print(f"连接失败: {e}")
            raise

    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            print("Neo4j连接已关闭")

    def run_query(self, query: str, parameters: dict = None) -> List[Dict[str, Any]]:
        """
        执行Cypher查询

        Args:
            query: Cypher查询语句
            parameters: 查询参数

        Returns:
            查询结果列表
        """
        if not self.driver:
            self._connect()

        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Neo4jWrapper:
    def __init__(self):
        self.connector = Neo4jConnection()

    def close(self):
        self.connector.close()

    def get_node_relationships(self, node_name):
        with self.driver.session() as session:
            result = session.run(
                f"""
MATCH (n:界面 {{name: '{node_name}'}})
OPTIONAL MATCH (n)-[:CONTAINS]->(contains_node)
OPTIONAL MATCH (n)<-[:RELATE]-(relate_node)
RETURN n.name AS node_name,
collect(DISTINCT contains_node.name) AS contains,
collect(DISTINCT relate_node.name) AS relate
            """
            )

            data = []
            for record in result:
                data.append(
                    {
                        "node_name": record["node_name"],
                        "node_type": record["node_type"],
                        "contains": record["contains"],
                        "relate": record["relate"],
                    }
                )
            return data


# 使用示例
wrapper = Neo4jWrapper()
results = wrapper.get_node_relationships("生管总控界面")
for item in results:
    print(f"{item['node_name']} ({item['node_type']}):")
    print(f"  contains: {item['contains']}")
    print(f"  relate: {item['relate']}")
    print()
wrapper.close()
