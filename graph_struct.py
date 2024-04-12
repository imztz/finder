# ...
class LinkedTable:
    def __init__(self):
        self.n = 0
        self.head = []

    def add_entry(self, head_id: int, content):
        if head_id >= self.n:
            while len(self.head) <= head_id:
                self.head.append([])
            self.n = head_id + 1
        self.head[head_id].append(content)

    def resize(self, new_n):
        if new_n > len(self.head):
            while len(self.head) < new_n:
                self.head.append([])
        self.n = new_n
        for i in range(len(self.head)):
            self.head[i].clear()

class GraphStruct:
    def __init__(self):
        self.out_edges = LinkedTable()
        self.in_edges = LinkedTable()
        self.subgraph = LinkedTable()
        self.edge_list = []
        self.node_set = set()
        self.num_nodes = 0
        self.num_edges = 0
        self.num_subgraph = 0

    def add_edge(self, idx, x, y):
        # Add edge information to out_edges and in_edges
        self.out_edges.add_entry(x, (idx, y))
        self.in_edges.add_entry(y, (idx, x))
        # Increment number of edges and update edge_list
        self.num_edges += 1
        self.edge_list.append((x, y))
        assert self.num_edges == len(self.edge_list) #"Edge list size mismatch"
        assert self.num_edges - 1 == idx #"Edge index mismatch"
        self.node_set.add(x)
        self.node_set.add(y)
        self.num_nodes = len(self.node_set)


    def add_node(self, subg_id, n_idx):
        # Add a node to a subgraph
        self.subgraph.add_entry(subg_id, n_idx)
        self.node_set.add(n_idx)
        self.num_nodes = len(self.node_set)

    def resize(self, num_subgraph, num_nodes=0):
        # Adjust sizes of structures based on new graph dimensions
        self.num_nodes = num_nodes
        self.num_edges = 0
        self.edge_list = []
        self.num_subgraph = num_subgraph
        self.out_edges.resize(num_nodes)
        self.in_edges.resize(num_nodes)
        self.subgraph.resize(num_subgraph)

def main():
    # 创建 GraphStruct 实例
    graph = GraphStruct()

    # 添加一些节点和边到图中
    graph.add_node(0, 0)  # 将节点 0 添加到子图 0
    graph.add_node(1, 1)  # 将节点 1 添加到子图 1

    # 添加一些边
    graph.add_edge(0, 0, 1)  # 添加边 从节点 0 到节点 1
    graph.add_edge(1, 1, 2)  # 添加边 从节点 1 到节点 2
    graph.add_edge(2, 2, 0)  # 添加边 从节点 2 到节点 0

    # 检查节点和边的数量
    print(f"节点数量: {graph.num_nodes}")
    print(f"边的数量: {graph.num_edges}")

    # 打印边的列表
    print("边的列表:")
    for edge in graph.edge_list:
        print(edge)

    # 打印出边和入边的邻接列表
    print("出边的邻接列表:")
    for i, edges in enumerate(graph.out_edges.head):
        print(f"节点 {i}: {edges}")
    
    print("入边的邻接列表:")
    for i, edges in enumerate(graph.in_edges.head):
        print(f"节点 {i}: {edges}")

    # 调整图的大小并检查变化
    print("调整图的大小...")
    graph.resize(2, 5)  # 假设有2个子图，5个节点
    print(f"调整后的节点数量: {graph.num_nodes}")
    print(f"调整后的子图数量: {graph.num_subgraph}")

if __name__ == "__main__":
    main()
