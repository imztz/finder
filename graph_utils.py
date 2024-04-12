from disjoint_set import DisjointSet
from typing import List
from copy import deepcopy

class GraphUtil:
    def __init__(self):
        pass

    def delete_node(self, adj_list_graph: List[List[int]], node: int):
        for neighbour in adj_list_graph[node]:
            adj_list_graph[neighbour].remove(node)
        adj_list_graph[node].clear()

    @staticmethod       # [JYFIXED] change to static method
    def recover_add_node(backup_completed_adj_list_graph: List[List[int]], backup_all_vex: List[bool], adj_list_graph: List[List[int]], node:int, union_set: DisjointSet):
        for neighbour_node in backup_completed_adj_list_graph[node]:
            if backup_all_vex[neighbour_node]:
                GraphUtil.add_edge(adj_list_graph, node, neighbour_node)
                union_set.merge(node, neighbour_node)
        backup_all_vex[node] = True

    @staticmethod       # [JYFIXED] change to static method
    def add_edge(adj_list_graph: List[List[int]], node0: int, node1: int):
        max_index = max(node0, node1)
        while len(adj_list_graph) <= max_index:
            adj_list_graph.append([])
        adj_list_graph[node0].append(node1)
        adj_list_graph[node1].append(node0)

def main():
    # 初始化图的邻接矩阵表示
    adj_list_graph = [
        [1, 2],  # 节点 0 连接到节点 1 和 2
        [0, 2],  # 节点 1 连接到节点 0 和 2
        [0, 1]   # 节点 2 连接到节点 0 和 1
    ]

    # 初始化所有节点状态为未访问
    backup_all_vex = [False] * len(adj_list_graph)

    backup_completed_adj_list_graph = deepcopy(adj_list_graph)
    # 创建 GraphUtil 实例
    graph_util = GraphUtil()

    # 创建 DisjointSet 实例
    union_set = DisjointSet(len(adj_list_graph))

    # 打印原始图
    print("原始图的邻接表表示:", adj_list_graph)

    # 删除节点 1 并打印结果
    graph_util.delete_node(adj_list_graph, 1)
    print("删除节点 1 后的邻接表表示:", adj_list_graph)

    graph_util.delete_node(adj_list_graph, 2)
    print("删除节点 2 后的邻接表表示:", adj_list_graph)

    graph_util.recover_add_node(backup_completed_adj_list_graph, backup_all_vex, adj_list_graph, 1, union_set)
    print(f'恢复节点 1 后的邻接表表示: {adj_list_graph}')

    graph_util.recover_add_node(backup_completed_adj_list_graph, backup_all_vex, adj_list_graph, 2, union_set)
    print(f'恢复节点 2 后的邻接表表示: {adj_list_graph}')

    # 添加新的边 (2, 3) 并打印结果
    graph_util.add_edge(adj_list_graph, 2, 3)
    print("添加边 (2, 3) 后的邻接表表示:", adj_list_graph)

if __name__ == "__main__":
    main()
