import random
from graph import Graph
from typing import List
from mvc_env import MvcEnv
from copy import deepcopy
class ReplaySample:
    def __init__(self, batch_size: int):
        self.g_list = [Graph() for _ in range(batch_size)]
        self.list_st = [[] for _ in range(batch_size)]
        self.list_s_primes = [[] for _ in range(batch_size)]
        self.list_at = [0 for _ in range(batch_size)]
        self.list_rt = [0.0 for _ in range(batch_size)]
        self.list_term = [False for _ in range(batch_size)]

class NStepReplayMem:
    def __init__(self, memory_size: int):
        self.memory_size = memory_size
        self.graphs = [Graph() for _ in range(memory_size)]
        self.actions = [0 for _ in range(memory_size)]
        self.rewards = [0.0 for _ in range(memory_size)]
        self.states = [[] for _ in range(memory_size)]
        self.s_primes = [[] for _ in range(memory_size)]
        self.terminals = [False for _ in range(memory_size)]
        self.current = 0
        self.count = 0

    def add(self, g: Graph, s_t: List[int], a_t: int, r_t: float, s_prime: List[int], terminal: bool):
        self.graphs[self.current] = g
        self.actions[self.current] = a_t
        self.rewards[self.current] = r_t
        self.states[self.current] = s_t
        self.s_primes[self.current] = s_prime
        self.terminals[self.current] = terminal

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def add_env(self, env: MvcEnv, n_step: int):
        assert env.is_terminal()
        num_steps = len(env.state_seq)
        assert num_steps > 0

        env.sum_rewards[num_steps - 1] = deepcopy(env.reward_seq[num_steps - 1])
        for i in range(num_steps - 2, -1, -1):
            env.sum_rewards[i] = env.sum_rewards[i + 1] + env.reward_seq[i]

        for i in range(num_steps):
            term_t = False
            if i + n_step >= num_steps:
                cur_r = env.sum_rewards[i]
                s_prime = deepcopy(env.action_list)
                term_t = True
            else:
                cur_r = env.sum_rewards[i] - env.sum_rewards[i + n_step]
                s_prime = deepcopy(env.state_seq[i + n_step])
            s_seq = deepcopy(env.state_seq[i])
            a_seq = deepcopy(env.act_seq[i])

            # self.add(env.graph, env.state_seq[i], env.act_seq[i], cur_r, s_prime, term_t)
            self.add(env.graph, s_seq, a_seq, cur_r, s_prime, term_t)

    def sampling(self, batch_size: int):
        assert self.count >= batch_size

        samples = ReplaySample(batch_size)

        for i in range(batch_size):
            idx = random.randint(0, self.count - 1)
            samples.g_list[i] = self.graphs[idx]
            samples.list_st[i] = self.states[idx]
            samples.list_at[i] = self.actions[idx]
            samples.list_rt[i] = self.rewards[idx]
            samples.list_s_primes[i] = self.s_primes[idx]
            samples.list_term[i] = self.terminals[idx]

        return samples