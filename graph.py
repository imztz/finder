import numpy as np
import networkx as nx
import random
from typing import List
import csv

class Graph:
    def __init__(self, num_nodes=0, num_edges=0, edges_from=None, edges_to=None):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.edge_list = []
        self.adj_list = [[] for _ in range(num_nodes)]
        if edges_from is not None and edges_to is not None:
            for x, y in zip(edges_from, edges_to):
                self.adj_list[int(x)].append(int(y))
                self.adj_list[int(y)].append(int(x))
                self.edge_list.append((int(x), int(y)))


    def get_two_rank_neighbors_ratio(self, covered: List[int]):
        covered_set = set(covered)
        sum = 0.0
        for i in range(self.num_nodes):
            if i not in covered_set:
                for j in range(i + 1, self.num_nodes):
                    if j not in covered_set:
                        intersection = set(self.adj_list[i]) & set(self.adj_list[j])
                        if intersection:
                            sum += 1.0
        return sum

    # def read_graphs_from_csv(self, file_name):
    #     graph_list = []
    #
    #     with open(file_name, 'r') as f:
    #         reader = csv.reader(f)
    #         next(reader)  # 跳过标题行
    #         for row in reader:
    #             num_nodes = int(row[0])
    #             num_edges = int(row[1])
    #             aadj_list = [list(map(int, adj_list.split())) for adj_list in row[2].split(';') if adj_list]
    #             eedge_list = [tuple(map(int, edge_list.split())) for edge_list in row[3].split(';') if edge_list]
    #
    #             ggg = Graph(num_nodes, num_edges)
    #             ggg.adj_list = aadj_list
    #             ggg.edge_list = eedge_list
    #             graph_list.append(ggg)
    #
    #     return graph_list

class GSet:
    def __init__(self):
        self.graph_pool = {}

    def clear(self):
        self.graph_pool.clear()

    def insert_graph(self, gid: int, graph: Graph):
        assert gid not in self.graph_pool
        self.graph_pool[gid] = graph

    def get(self, gid: int):
        assert gid in self.graph_pool
        return self.graph_pool[gid]

    def sample(self):
        assert len(self.graph_pool) > 0
        gid = random.choice(list(self.graph_pool))
        return self.graph_pool[gid]

if __name__ == '__main__':
    import networkx as nx
    cur_n = 40
    g = nx.erdos_renyi_graph(n=cur_n, p=0.15)
    g2 = Graph(4,3,[0,0,0],[1,2,3])
    adjList = g2.adj_list
    print(adjList)
    print("\n")
    print(g2.edge_list)
    gs = GSet()
    gs.insert_graph(0, g)
    print("\ndone.")