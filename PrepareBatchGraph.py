import graph_struct
from graph_struct import GraphStruct
import math
from graph import Graph
from typing import List, Tuple
from copy import deepcopy

class SparseMatrix:
    def __init__(self):
        self.rowIndex = []
        self.colIndex = []
        self.value = []
        self.rowNum = 0
        self.colNum = 0


class PrepareBatchGraph:
    def __init__(self, aggregatorID: int):
        self.act_select = SparseMatrix()
        self.aggregatorID = aggregatorID
        self.rep_global = SparseMatrix()
        self.n2nsum_param = SparseMatrix()
        self.laplacian_param = SparseMatrix()
        self.subgsum_param = SparseMatrix()
        self.idx_map_list = []
        self.subgraph_id_span = []
        self.aux_feat = []
        self.graph = graph_struct.GraphStruct()
        self.avail_act_cnt = []

    def get_status_info(self, g: Graph, num: int, covered: List[int]):
        c = set()
        for i in range(num):
            c.add(covered[i])
        idx_map = [-1] * g.num_nodes
        counter = 0
        twohop_number = 0
        threehop_number = 0
        node_twohop_counter = {}

        n = 0
        for p in g.edge_list:
            if p[0] in c or p[1] in c:
                counter += 1
            else:
                if idx_map[p[0]] < 0:
                    n += 1
                if idx_map[p[1]] < 0:
                    n += 1
                idx_map[p[0]] = 0
                idx_map[p[1]] = 0
                node_twohop_counter[p[0]] = node_twohop_counter.get(p[0], 0) + 1
                node_twohop_counter[p[1]] = node_twohop_counter.get(p[1], 0) + 1
                twohop_number += node_twohop_counter[p[0]] + node_twohop_counter[p[1]] - 2

        return n, counter, twohop_number, threehop_number, idx_map

    def setup_graph_input(self, idxes: List[int], g_list: List[Graph], covered: List[List[int]], actions: List[int]):
        self.act_select = SparseMatrix()
        self.rep_global = SparseMatrix()
        self.idx_map_list = [[] for _ in idxes]
        self.avail_act_cnt = [0 for _ in idxes]
        node_cnt = 0

        for i, idx in enumerate(idxes):
            temp_feat = []
            g = g_list[idx]
            if g.num_nodes > 0:
                temp_feat.append(len(covered[idx]) / g.num_nodes)

            avail_act, counter, twohop_number, threehop_number, idx_map = self.get_status_info(g, len(covered[idx]),
                                                                                               covered[idx])
            self.idx_map_list[i] = idx_map
            self.avail_act_cnt[i] = avail_act

            if len(g.edge_list) > 0:
                temp_feat.append(counter / len(g.edge_list))

            temp_feat.append(twohop_number / (g.num_nodes * g.num_nodes))
            temp_feat.append(1.0)

            node_cnt += self.avail_act_cnt[i]
            self.aux_feat.append(temp_feat)

        self.graph.resize(len(idxes), node_cnt)

        if actions is not None:
            self.act_select.rowNum = len(idxes)
            self.act_select.colNum = node_cnt
        else:
            self.rep_global.rowNum = node_cnt
            self.rep_global.colNum = len(idxes)

        node_cnt = 0
        edge_cnt = 0
        for i, idx in enumerate(idxes):
            g = g_list[idx]
            idx_map = deepcopy(self.idx_map_list[i])

            t = 0
            for n_idx in range(g.num_nodes):
                if idx_map[n_idx] < 0:
                    continue
                idx_map[n_idx] = t
                self.graph.add_node(i, node_cnt + t)
                if actions is None:
                    self.rep_global.rowIndex.append(node_cnt + t)
                    self.rep_global.colIndex.append(i)
                    self.rep_global.value.append(1.0)
                t += 1
            assert t == self.avail_act_cnt[i]

            if actions is not None:
                act = actions[idx]
                assert idx_map[act] >= 0 and 0 <= act < g.num_nodes
                self.act_select.rowIndex.append(i)
                self.act_select.colIndex.append(node_cnt + idx_map[act])
                self.act_select.value.append(1.0)

            for x, y in g.edge_list:
                if idx_map[x] < 0 or idx_map[y] < 0:
                    continue
                self.graph.add_edge(edge_cnt, idx_map[x] + node_cnt, idx_map[y] + node_cnt)
                edge_cnt += 1
                self.graph.add_edge(edge_cnt, idx_map[y] + node_cnt, idx_map[x] + node_cnt)
                edge_cnt += 1

            node_cnt += self.avail_act_cnt[i]

        n2n, laplacian = n2n_construct(self.graph, self.aggregatorID)
        self.n2nsum_param = deepcopy(n2n)
        self.laplacian_param = deepcopy(laplacian)
        self.subgsum_param = subg_construct(self.graph, self.subgraph_id_span)

    def setup_train(self, idxes, g_list, covered, actions):
        self.setup_graph_input(idxes, g_list, covered, actions)

    def setup_pred_all(self, idxes, g_list, covered):
        self.setup_graph_input(idxes, g_list, covered, None)


def n2n_construct(graph: GraphStruct, aggregatorID: int):
    result = SparseMatrix()
    result.rowNum = graph.num_nodes
    result.colNum = graph.num_nodes

    result_laplacian = SparseMatrix()
    result_laplacian.rowNum = graph.num_nodes
    result_laplacian.colNum = graph.num_nodes

    for i in range(graph.num_nodes):
        in_list = graph.in_edges.head[i]

        if len(in_list) > 0:
            result_laplacian.value.append(len(in_list))
            result_laplacian.rowIndex.append(i)
            result_laplacian.colIndex.append(i)

        for j in range(len(in_list)):
            if aggregatorID == 0:  # sum
                value = 1.0
            elif aggregatorID == 1:  # mean
                value = 1.0 / len(in_list)
            elif aggregatorID == 2:  # GCN
                neighborDegree = len(graph.in_edges.head[in_list[j][1]])
                selfDegree = len(in_list)
                norm = math.sqrt(neighborDegree + 1) * math.sqrt(selfDegree + 1)
                value = 1.0 / norm
            else:
                value = 0

            result.value.append(value)
            result.rowIndex.append(i)
            result.colIndex.append(in_list[j][1])

            result_laplacian.value.append(-1.0)
            result_laplacian.rowIndex.append(i)
            result_laplacian.colIndex.append(in_list[j][1])

    return result, result_laplacian


def e2n_construct(graph: GraphStruct):
    result = SparseMatrix()
    result.rowNum = graph.num_nodes
    result.colNum = graph.num_edges

    for i in range(graph.num_nodes):
        for (idx, _) in graph.in_edges.head[i]:
            result.value.append(1.0)
            result.rowIndex.append(i)
            result.colIndex.append(idx)

        return result


def n2e_construct(graph: GraphStruct):
    result = SparseMatrix()
    result.rowNum = graph.num_edges
    result.colNum = graph.num_nodes

    for i, (x, _) in enumerate(graph.edge_list):
        result.value.append(1.0)
        result.rowIndex.append(i)
        result.colIndex.append(x)

    return result


def e2e_construct(graph: GraphStruct):
    result = SparseMatrix()
    result.rowNum = graph.num_edges
    result.colNum = graph.num_edges

    for i, (node_from, node_to) in enumerate(graph.edge_list):
        for (idx, second_node) in graph.in_edges.head[node_from]:
            if second_node == node_to:
                continue
            result.value.append(1.0)
            result.rowIndex.append(i)
            result.colIndex.append(idx)

    return result


def subg_construct(graph: GraphStruct, subgraph_id_span: List[Tuple[int, int]]):
    result = SparseMatrix()
    result.rowNum = graph.num_subgraph
    result.colNum = graph.num_nodes

    subgraph_id_span.clear()
    start = 0
    for i in range(graph.num_subgraph):
        list_subg = graph.subgraph.head[i]
        end = start + len(list_subg) - 1
        for j in range(len(list_subg)):
            result.value.append(1.0)
            result.rowIndex.append(i)
            result.colIndex.append(list_subg[j])
        if len(list_subg) > 0:
            subgraph_id_span.append((start, end))
        else:
            subgraph_id_span.append((graph.num_nodes, graph.num_nodes))
        start = end + 1

    return result
