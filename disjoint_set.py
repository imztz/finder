class DisjointSet:
    def __init__(self, graph_size: int):
        self.parent = list(range(graph_size))
        self.rank = [1] * graph_size
        self.max_rank = 1

    def find_root(self, node: int):
        if self.parent[node] != node:
            self.parent[node] = self.find_root(self.parent[node])
        return self.parent[node]

    def merge(self, node1: int, node2: int):
        root1 = self.find_root(node1)
        root2 = self.find_root(node2)

        if root1 != root2:
            if self.rank[root2] > self.rank[root1]:
                self.parent[root1] = root2
                self.rank[root2] += self.rank[root1]
                self.max_rank = max(self.max_rank, self.rank[root2])
            else:
                self.parent[root2] = root1
                self.rank[root1] += self.rank[root2]
                self.max_rank = max(self.max_rank, self.rank[root1])

    def get_biggest_component_current_ratio(self):
        return self.max_rank / float(len(self.rank))

    def get_rank(self, node: int):
        return self.rank[node]

def main():
    # 假设有一个大小为 10 的图
    graph_size = 10
    disjoint_set = DisjointSet(graph_size)
    
    # 进行一些合并操作
    disjoint_set.merge(0, 1)
    disjoint_set.merge(1, 2)
    disjoint_set.merge(2, 3)
    disjoint_set.merge(3, 4)
    disjoint_set.merge(4, 5)
    disjoint_set.merge(5, 6)
    disjoint_set.merge(6, 7)

    # 打印每个节点的根节点和所在集合的大小
    for i in range(graph_size):
        root = disjoint_set.find_root(i)
        rank = disjoint_set.get_rank(root)
        print(f"节点 {i} 的根节点是 {root}, 集合大小是 {rank}")

    # 打印最大连通分量的大小和比例
    biggest_component_ratio = disjoint_set.get_biggest_component_current_ratio()
    print(f"最大连通分量的大小比例是: {biggest_component_ratio:.2f}")

if __name__ == "__main__":
    main()