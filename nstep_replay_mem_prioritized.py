import graph
import numpy as np
import random
import time
from mvc_env import MvcEnv
from typing import List

class Data:
    def __init__(self):
        self.g = graph.Graph()
        self.s_t = []
        self.s_prime = []
        self.a_t = 0
        self.r_t = 0.0
        self.term_t = False

class LeafResult:
    def __init__(self):
        self.leaf_idx = 0
        self.p = 0.0
        self.data = Data()

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = [0.0] * (2 * capacity - 1)
        self.data = [None] * capacity
        self.data_pointer = 0
        self.minElement = float('inf')
        self.maxElement = float('-inf')

    def add(self, p: float, _data: Data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = _data
        self.update(tree_idx, p)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx: int, p: float):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
        self.minElement = min(self.minElement, p)
        self.maxElement = max(self.maxElement, p)

    def get_leaf(self, v: float):
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
        data_idx = leaf_idx - self.capacity + 1
        result = LeafResult()
        result.leaf_idx = leaf_idx
        result.p = self.tree[leaf_idx]
        result.data = self.data[data_idx]
        return result
    
class ReplaySample:
    def __init__(self, batch_size: int):
        self.b_idx = [0] * batch_size  
        self.ISWeights = [0.0] * batch_size  
        self.g_list = [graph.Graph()] * batch_size  
        self.list_st = [[] for _ in range(batch_size)] 
        self.list_s_primes = [[] for _ in range(batch_size)] 
        self.list_at = [0] * batch_size  
        self.list_rt = [0.0] * batch_size 
        self.list_term = [False] * batch_size

class Memory:
    def __init__(self, epsilon: float, alpha: float, beta: float, beta_increment_per_sampling: float, abs_err_upper: float, capacity: int):
        self.tree = SumTree(capacity)
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.abs_err_upper = abs_err_upper

    def store(self, transition: Data):
        max_p = self.tree.maxElement
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def add(self, env: MvcEnv, n_step: int):
        assert env.is_terminal()
        num_steps = len(env.state_seq)
        assert num_steps > 0

        env.sum_rewards[num_steps - 1] = env.reward_seq[num_steps - 1]
        for i in range(num_steps - 2, -1, -1):
            env.sum_rewards[i] = env.sum_rewards[i + 1] + env.reward_seq[i]

        for i in range(num_steps):
            if i + n_step >= num_steps:
                cur_r = env.sum_rewards[i]
                s_prime = env.action_list
                term_t = True
            else:
                cur_r = env.sum_rewards[i] - env.sum_rewards[i + n_step]
                s_prime = env.state_seq[i + n_step]

            transition = Data()
            transition.g = env.graph
            transition.s_t = env.state_seq[i]
            transition.s_prime = s_prime
            transition.a_t = env.act_seq[i]
            transition.r_t = cur_r
            transition.term_t = term_t
            self.store(transition)

    def sampling(self, n: int):
        result = ReplaySample(n)
        total_p = self.tree.tree[0]
        pri_seg = total_p / n
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        min_prob = self.tree.minElement / total_p

        for i in range(n):
            a = pri_seg * i
            b = pri_seg * (i + 1)
            random.seed(time.time())
            v = random.uniform(a, b)
            leafResult = self.tree.get_leaf(v)
            result.b_idx[i] = leafResult.leaf_idx
            prob = leafResult.p / total_p
            result.ISWeights[i] = np.power(prob / min_prob, -self.beta)
            result.g_list[i] = leafResult.data.g
            result.list_st[i] = leafResult.data.s_t
            result.list_s_primes[i] = leafResult.data.s_prime
            result.list_at[i] = leafResult.data.a_t
            result.list_rt[i] = leafResult.data.r_t
            result.list_term[i] = leafResult.data.term_t

        return result

    def batch_update(self, tree_idx: List[int], abs_errors: List[float]):
        for i in range(len(tree_idx)):
            abs_errors[i] += self.epsilon
            clipped_error = min(abs_errors[i], self.abs_err_upper)
            ps = np.power(clipped_error, self.alpha)
            self.tree.update(tree_idx[i], ps)  