import pickle
import numpy as np
import tensorflow as tf
import networkx as nx
import graph
from graph import Graph
import disjoint_set
from decrease_strategy import DecreaseComponentStrategy
import graph_utils
import decrease_strategy
from typing import List
from PrepareBatchGraph import SparseMatrix
from copy import deepcopy

class Utils:
    def __init__(self):
        self.max_wcc_sz_list = []

    def re_insert(self, g_graph: Graph, solution: List[int], all_vex: List[int], decrease_strategy_id: int, reinsert_each_step: int):
        if decrease_strategy_id == 1:
            d_decrease_strategy = decrease_strategy.DecreaseComponentCount()
        elif decrease_strategy_id == 2:
            d_decrease_strategy = decrease_strategy.DecreaseComponentRank()
        elif decrease_strategy_id == 3:
            d_decrease_strategy = decrease_strategy.DecreaseComponentMultiple()
        else:
            d_decrease_strategy = decrease_strategy.DecreaseComponentRank()

        return self.re_insert_inner(solution, g_graph, all_vex, d_decrease_strategy, reinsert_each_step)

    def re_insert_inner(self, before_output: List[int], g_graph: Graph, all_vex: List[int], d_decrease_strategy: DecreaseComponentStrategy, reinsert_each_step: int):
        current_adj_list_graph = []
        backup_completed_adj_list_graph = deepcopy(g_graph.adj_list)
        current_all_vex = [False] * g_graph.num_nodes
        for v in all_vex:
            current_all_vex[v] = True

        left_output = set(before_output)
        final_output = []

        d_disjoint_set = disjoint_set.DisjointSet(g_graph.num_nodes)

        while left_output:
            batch_list = []

            for each_node in left_output:
                decrease_value = d_decrease_strategy.decrease_component_num_if_add_node(current_adj_list_graph, current_all_vex, d_disjoint_set, each_node)
                batch_list.append((decrease_value, each_node))

            if reinsert_each_step >= len(batch_list):
                reinsert_each_step = len(batch_list)
            else:
                batch_list.sort(key=lambda x: x[0])
                batch_list = batch_list[:reinsert_each_step]

            for _, each_node in batch_list:
                final_output.append(each_node)
                left_output.remove(each_node)
                graph_utils.GraphUtil.recover_add_node(backup_completed_adj_list_graph, current_all_vex, current_adj_list_graph, each_node, d_disjoint_set)

        final_output.reverse()
        return final_output

    def get_robustness(self, g_graph: Graph, solution: List[int]):
        self.max_wcc_sz_list.clear()
        current_adj_list_graph = []
        backup_completed_adj_list_graph = deepcopy(g_graph.adj_list)
        d_disjoint_set = disjoint_set.DisjointSet(g_graph.num_nodes)
        total_max_num = 0.0
        current_all_vex = [False] * g_graph.num_nodes
        for node in reversed(solution):
            graph_utils.GraphUtil.recover_add_node(backup_completed_adj_list_graph, current_all_vex,
                                                    current_adj_list_graph, node, d_disjoint_set)
            total_max_num += float(d_disjoint_set.max_rank)
            self.max_wcc_sz_list.append(d_disjoint_set.max_rank / g_graph.num_nodes)
        total_max_num -= d_disjoint_set.max_rank
        self.max_wcc_sz_list.reverse()

        return total_max_num / (g_graph.num_nodes * g_graph.num_nodes)       # [JYFIXED] g_graph.num_nodes is already a number, no need to len()

    def get_mx_wcc_sz(self, g_graph: Graph):
        assert graph is not None
        d_disjoint_set = disjoint_set.DisjointSet(g_graph.num_nodes)
        for i in range(len(g_graph.adj_list)):
            for j in range(len(g_graph.adj_list[i])):
                d_disjoint_set.merge(i, g_graph.adj_list[i][j])

        return d_disjoint_set.max_rank

    def betweenness(self, g_graph: Graph):
        G = self.nx_graph(g_graph)
        # centrality = nx.betweenness_centrality(G, normalized=True)
        centrality = nx.betweenness_centrality(G)
        return centrality

    def nx_graph(self, g_graph: Graph):
        G = nx.Graph()
        for i, edges in enumerate(g_graph.adj_list):
            for j in edges:
                G.add_edge(i, j)
        return G

def sparseMatrix2TFSparseMatrixValue(sparseMatrix: SparseMatrix):
    """
    参考: https://blog.csdn.net/baidu_24536755/article/details/88261936
    :param sparseMatrix:
    :return:
    """
    rowNum = sparseMatrix.rowNum
    colNum = sparseMatrix.colNum
    indices = np.mat([sparseMatrix.rowIndex, sparseMatrix.colIndex]).transpose()
    dense_shape = (rowNum, colNum)
    tfResult = tf.SparseTensorValue(indices=indices, values=sparseMatrix.value, dense_shape=dense_shape)
    return tfResult

if __name__ == '__main__':
    with open("data/sparseMatrix.pkl", "rb") as fin:
        info = pickle.load(fin)
    output1d = sparseMatrix2TFSparseMatrixValue(info["1d"])
    output2d = sparseMatrix2TFSparseMatrixValue(info["2d"])
    print("done.")