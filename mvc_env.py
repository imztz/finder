# 3.16
import networkx as nx
import numpy as np
import random
from graph import Graph
from disjoint_set import DisjointSet

class MvcEnv:
    def __init__(self, norm=1.0):
        self.norm = norm
        self.graph = Graph()
        self.num_covered_edges = 0
        self.cc_num = 1.0
        self.state_seq = []
        self.act_seq = []
        self.action_list = []
        self.reward_seq = []
        self.sum_rewards = []
        self.covered_set = set()
        self.avail_list = []

    def s0(self, g: Graph):
        self.graph = g
        self.covered_set.clear()
        self.action_list.clear()
        self.num_covered_edges = 0
        self.cc_num = 1.0
        self.state_seq.clear()
        self.act_seq.clear()
        self.reward_seq.clear()
        self.sum_rewards.clear()

    def step(self, a: int):
        assert self.graph is not None
        assert a not in self.covered_set
        self.state_seq.append(list(self.action_list))
        self.act_seq.append(a)
        self.covered_set.add(a)
        self.action_list.append(a)

        for neigh in self.graph.adj_list[a]:
            if neigh not in self.covered_set:
                self.num_covered_edges += 1

        r_t = self.get_reward()
        self.reward_seq.append(r_t)
        self.sum_rewards.append(r_t)

        return r_t

    def step_without_reward(self, a: int):
        assert self.graph is not None
        assert a not in self.covered_set
        self.covered_set.add(a)
        self.action_list.append(a)

        for neigh in self.graph.adj_list[a]:
            if neigh not in self.covered_set:
                self.num_covered_edges += 1

    def random_action(self):
        assert self.graph is not None
        self.avail_list.clear()
        for i in range(self.graph.num_nodes):
            if i not in self.covered_set:
                useful = False
                for neigh in self.graph.adj_list[i]:
                    if neigh not in self.covered_set:
                        useful = True
                        break
                if useful:
                    self.avail_list.append(i)

        assert len(self.avail_list) > 0
        return random.choice(self.avail_list)

    def between_action(self):
        assert self.graph is not None
        G = self.nx_graph(self.graph)
        subgraph = G.subgraph([node for node in G.nodes if node not in self.covered_set])
        BC = self.betweenness(subgraph)
        max_bc_node = max(BC, key=BC.get)
        return max_bc_node

    def is_terminal(self):
        assert self.graph is not None
        return self.num_covered_edges == self.graph.num_edges

    def get_reward(self):
        return -self.get_max_connected_nodes_num() / float(self.graph.num_nodes ** 2)

    def print_graph(self):
        print("edge_list:")
        print([edge for edge in self.nx_graph(self.graph).edges()])
        print("\ncovered_set:")
        print(list(self.covered_set))

    def get_num_of_connected_components(self):
        assert self.graph is not None
        d_set = DisjointSet(self.graph.num_nodes)
        for i in range(self.graph.num_nodes):
            if i not in self.covered_set:
                for neigh in self.graph.adj_list[i]:
                    if neigh not in self.covered_set:
                        d_set.merge(i, neigh)
        lcc_ids = set(d_set.parent[i] for i in range(self.graph.num_nodes))
        return len(lcc_ids)

    def get_max_connected_nodes_num(self):
        assert self.graph is not None
        d_set = DisjointSet(self.graph.num_nodes)
        for i in range(self.graph.num_nodes):
            if i not in self.covered_set:
                for neigh in self.graph.adj_list[i]:
                    if neigh not in self.covered_set:
                        d_set.merge(i, neigh)
        return d_set.max_rank

    def betweenness(self, graph):
        G = self.nx_graph(graph)
        centrality = nx.betweenness_centrality(G)
        return centrality

    def nx_graph(self, graph: Graph):
        G = nx.Graph()
        for i, edges in enumerate(graph.adj_list):
            for j in edges:
                G.add_edge(i, j)
        return G

def main():
    # 创建一个简单的图结构
    g = Graph(6, 4)
    g.adj_list = [[1], [0], [], [4, 5], [3, 5], [3, 4]]  # 四节点的图的邻接矩阵

    env = MvcEnv()
    env.s0(g)

    print("初始图状态：")
    env.print_graph()

    print("\n图的介数中心性：")
    print(env.betweenness(g))

    while not env.is_terminal():
        action = env.random_action()
        reward = env.step(action)
        print(f"执行动作：{action}, 得到奖励：{reward}")

    print("\nnx图状态：")
    GG = env.nx_graph(g)
    print(GG.edges)

    env.print_graph()

    # 获取和打印连接组件信息
    num_connected_components = env.get_num_of_connected_components()
    max_connected_nodes_num = env.get_max_connected_nodes_num()
    print(f"连通子图的数量: {num_connected_components}, 最大连通子图节点数: {max_connected_nodes_num}")

if __name__ == "__main__":
    main()