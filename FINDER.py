import numpy as np
import networkx as nx
import tensorflow as tf
import tqdm
import os
import time
import random
from omegaconf import OmegaConf
from graph import GSet, Graph
from utils import Utils, sparseMatrix2TFSparseMatrixValue
from nstep_replay_mem_prioritized import Memory
from nstep_replay_mem import NStepReplayMem
from mvc_env import MvcEnv
from PrepareBatchGraph import PrepareBatchGraph

class FINDER:
    def __init__(self):
        self.configPath = "configs/FINDER_ND.yaml"
        conf = OmegaConf.load(self.configPath)
        self.conf = conf
        self.embedding_size = conf.HyperParameters.EMBEDDING_SIZE
        self.learning_rate = conf.HyperParameters.LEARNING_RATE
        self.g_type = 'barabasi_albert'  # erdos_renyi, powerlaw, small-world

        self.TrainSet = GSet()
        self.TestSet = GSet()
        self.inputs = dict()
        self.reg_hidden = conf.HyperParameters.REG_HIDDEN
        self.utils = Utils()

        ############----------------------------- variants of DQN(start) ------------------- ###################################
        self.IsHuberloss = False
        self.IsDoubleDQN = False
        self.IsPrioritizedSampling = False
        self.IsDuelingDQN = False
        self.IsMultiStepDQN = True     ##(if IsNStepDQN=False, N_STEP==1)
        self.IsDistributionalDQN = False
        self.IsNoisyNetDQN = False
        self.Rainbow = False
        ############----------------------------- variants of DQN(end) ------------------- ###################################

        # Simulator
        self.ngraph_train = 0
        self.ngraph_test = 0
        self.env_list = []
        self.g_list = []
        self.pred = []
        if self.IsPrioritizedSampling:
            self.nStepReplayMem = Memory(conf.Memory.epsilon, conf.Memory.alpha, conf.Memory.beta,
                                         conf.Memory.beta_increment_per_sampling, conf.Memory.TD_err_upper,
                                         conf.HyperParameters.MEMORY_SIZE)
        else:
            self.nStepReplayMem = NStepReplayMem(conf.HyperParameters.MEMORY_SIZE)

        for i in range(conf.Simulation.num_env):
            self.env_list.append(MvcEnv(conf.Simulation.NUM_MAX))
            self.g_list.append(Graph())

        self.test_env = MvcEnv(conf.Simulation.NUM_MAX)

        # [batch_size, node_cnt]
        self.action_select = tf.sparse_placeholder(tf.float32, name="action_select")
        # [node_cnt, batch_size]
        self.rep_global = tf.sparse_placeholder(tf.float32, name="rep_global")
        # [node_cnt, node_cnt]
        self.n2nsum_param = tf.sparse_placeholder(tf.float32, name="n2nsum_param")
        # [node_cnt, node_cnt]
        self.laplacian_param = tf.sparse_placeholder(tf.float32, name="laplacian_param")
        # [batch_size, node_cnt]
        self.subgsum_param = tf.sparse_placeholder(tf.float32, name="subgsum_param")
        # [batch_size,1]
        self.target = tf.placeholder(tf.float32, [conf.Simulation.BATCH_SIZE, 1], name="target")
        # [batch_size, aux_dim]
        self.aux_input = tf.placeholder(tf.float32, name="aux_input")

        # [batch_size, 1]
        if self.IsPrioritizedSampling:
            self.ISWeights = tf.placeholder(tf.float32, [conf.Simulation.BATCH_SIZE, 1], name='IS_weights')

        # init Q network
        self.loss, self.trainStep, self.q_pred, self.q_on_all, self.Q_param_list = self.BuildNet()  # [loss,trainStep,q_pred, q_on_all, ...]
        # init Target Q Network
        self.lossT, self.trainStepT, self.q_predT, self.q_on_allT, self.Q_param_listT = self.BuildNet()
        # takesnapsnot
        self.copyTargetQNetworkOperation = [a.assign(b) for a, b in zip(self.Q_param_listT, self.Q_param_list)]

        self.UpdateTargetQNetwork = tf.group(*self.copyTargetQNetworkOperation)
        # saving and loading networks
        self.saver = tf.train.Saver(max_to_keep=None)
        # self.session = tf.InteractiveSession()
        config = tf.ConfigProto(device_count={"CPU": 8},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=100,
                                intra_op_parallelism_threads=100,
                                log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        # self.session = tf_debug.LocalCLIDebugWrapperSession(self.session)
        self.session.run(tf.global_variables_initializer())
        print("init done.")

    def BuildNet(self):
        # [2, embed_dim]
        w_n2l = tf.Variable(tf.truncated_normal([2, self.embedding_size], stddev=self.conf.Simulation.initialization_stddev), tf.float32)
        # [embed_dim, embed_dim]
        p_node_conv = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=self.conf.Simulation.initialization_stddev), tf.float32)
        if self.conf.Embedding.embeddingMethod == 1:    #'graphsage'
            # [embed_dim, embed_dim]
            p_node_conv2 = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=self.conf.Simulation.initialization_stddev), tf.float32)
            # [2*embed_dim, embed_dim]
            p_node_conv3 = tf.Variable(tf.truncated_normal([2*self.embedding_size, self.embedding_size], stddev=self.conf.Simulation.initialization_stddev), tf.float32)

        #[reg_hidden+aux_dim, 1]
        if self.reg_hidden > 0:
            #[2*embed_dim, reg_hidden]
        #    h1_weight = tf.Variable(tf.truncated_normal([2 * self.embedding_size, self.reg_hidden], stddev=self.conf.Simulation.initialization_stddev), tf.float32)
            # [embed_dim, reg_hidden]
            h1_weight = tf.Variable(tf.truncated_normal([self.embedding_size, self.reg_hidden], stddev=self.conf.Simulation.initialization_stddev), tf.float32)
            #[reg_hidden1, reg_hidden2]
            # h2_weight = tf.Variable(tf.truncated_normal([self.reg_hidden1, self.reg_hidden2], stddev=self.conf.Simulation.initialization_stddev), tf.float32)
            #[reg_hidden+aux_dim, 1]
            h2_weight = tf.Variable(tf.truncated_normal([self.reg_hidden + self.conf.Simulation.aux_dim, 1], stddev=self.conf.Simulation.initialization_stddev), tf.float32)
            #[reg_hidden2 + aux_dim, 1]
            last_w = h2_weight
        else:
            #[2*embed_dim, reg_hidden]
            h1_weight = tf.Variable(tf.truncated_normal([2 * self.embedding_size, self.reg_hidden], stddev=self.conf.Simulation.initialization_stddev), tf.float32)
            # [embed_dim, reg_hidden]
            # h1_weight = tf.Variable(tf.truncated_normal([self.embedding_size, self.reg_hidden], stddev=self.conf.Simulation.initialization_stddev), tf.float32)
            #[2*embed_dim, reg_hidden]
            last_w = h1_weight

        ## [embed_dim, 1]
        cross_product = tf.Variable(tf.truncated_normal([self.embedding_size, 1], stddev=self.conf.Simulation.initialization_stddev), tf.float32)

        #[node_cnt, 2]
        nodes_size = tf.shape(self.n2nsum_param)[0]
        node_input = tf.ones((nodes_size,2))

        y_nodes_size = tf.shape(self.subgsum_param)[0]
        # [batch_size, 2]
        y_node_input = tf.ones((y_nodes_size,2))

        #[node_cnt, 2] * [2, embed_dim] = [node_cnt, embed_dim]
        input_message = tf.matmul(tf.cast(node_input,tf.float32), w_n2l)

        #[node_cnt, embed_dim]  # no sparse
        input_potential_layer = tf.nn.relu(input_message)

        # no sparse
        # [batch_size, embed_dim]
        y_input_message = tf.matmul(tf.cast(y_node_input,tf.float32), w_n2l)
        #[batch_size, embed_dim]  # no sparse
        y_input_potential_layer = tf.nn.relu(y_input_message)

        #input_potential_layer = input_message
        lv = 0
        #[node_cnt, embed_dim], no sparse
        cur_message_layer = input_potential_layer
        cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)

        #[batch_size, embed_dim], no sparse
        y_cur_message_layer = y_input_potential_layer
        # [batch_size, embed_dim]
        y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)

        while lv < self.conf.Embedding.max_bp_iter:
            lv = lv + 1
            #[node_cnt, node_cnt] * [node_cnt, embed_dim] = [node_cnt, embed_dim], dense
            n2npool = tf.sparse_tensor_dense_matmul(tf.cast(self.n2nsum_param,tf.float32), cur_message_layer)

            #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
            node_linear = tf.matmul(n2npool, p_node_conv)

            # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
            y_n2npool = tf.sparse_tensor_dense_matmul(tf.cast(self.subgsum_param,tf.float32), cur_message_layer)
            #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
            y_node_linear = tf.matmul(y_n2npool, p_node_conv)

            if self.conf.Embedding.embeddingMethod == 0: # 'structure2vec'
                #[node_cnt, embed_dim] + [node_cnt, embed_dim] = [node_cnt, embed_dim], return tensed matrix
                merged_linear = tf.add(node_linear,input_message)
                #[node_cnt, embed_dim]
                cur_message_layer = tf.nn.relu(merged_linear)

                #[batch_size, embed_dim] + [batch_size, embed_dim] = [batch_size, embed_dim], return tensed matrix
                y_merged_linear = tf.add(y_node_linear, y_input_message)
                #[batch_size, embed_dim]
                y_cur_message_layer = tf.nn.relu(y_merged_linear)
            else:   # 'graphsage'
                #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
                cur_message_layer_linear = tf.matmul(tf.cast(cur_message_layer, tf.float32), p_node_conv2)

                #[[node_cnt, embed_dim] [node_cnt, embed_dim]] = [node_cnt, 2*embed_dim], return tensed matrix
                merged_linear = tf.concat([node_linear, cur_message_layer_linear], 1)
                #[node_cnt, 2*embed_dim]*[2*embed_dim, embed_dim] = [node_cnt, embed_dim]
                cur_message_layer = tf.nn.relu(tf.matmul(merged_linear, p_node_conv3))

                #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
                y_cur_message_layer_linear = tf.matmul(tf.cast(y_cur_message_layer, tf.float32), p_node_conv2)
                #[[batch_size, embed_dim] [batch_size, embed_dim]] = [batch_size, 2*embed_dim], return tensed matrix
                y_merged_linear = tf.concat([y_node_linear, y_cur_message_layer_linear], 1)
                #[batch_size, 2*embed_dim]*[2*embed_dim, embed_dim] = [batch_size, embed_dim]
                y_cur_message_layer = tf.nn.relu(tf.matmul(y_merged_linear, p_node_conv3))

            cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)
            y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)

        self.node_embedding = cur_message_layer
        #[batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim], dense
       # y_potential = tf.sparse_tensor_dense_matmul(tf.cast(self.subgsum_param,tf.float32), cur_message_layer)
        y_potential = y_cur_message_layer
        #[batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
        action_embed = tf.sparse_tensor_dense_matmul(tf.cast(self.action_select, tf.float32), cur_message_layer)

      #  embed_s_a = tf.concat([action_embed,y_potential],1)

        # # [batch_size, embed_dim, embed_dim]
        temp = tf.matmul(tf.expand_dims(action_embed, axis=2),tf.expand_dims(y_potential, axis=1))
        # [batch_size, embed_dim]
        Shape = tf.shape(action_embed)
       # [batch_size, embed_dim], first transform
        embed_s_a = tf.reshape(tf.matmul(temp, tf.reshape(tf.tile(cross_product,[Shape[0],1]),[Shape[0],Shape[1],1])),Shape)

        #[batch_size, embed_dim]
        last_output = embed_s_a

        if self.reg_hidden > 0:
            #[batch_size, 2*embed_dim] * [2*embed_dim, reg_hidden] = [batch_size, reg_hidden], dense
            hidden = tf.matmul(embed_s_a, h1_weight)
            #[batch_size, reg_hidden]
            last_output = tf.nn.relu(hidden)

        # if reg_hidden == 0: ,[[batch_size, 2*embed_dim], [batch_size, aux_dim]] = [batch_size, 2*embed_dim+aux_dim]
        # if reg_hidden > 0: ,[[batch_size, reg_hidden], [batch_size, aux_dim]] = [batch_size, reg_hidden+aux_dim]
        last_output = tf.concat([last_output, self.aux_input], 1)
        #if reg_hidden == 0: ,[batch_size, 2*embed_dim+aux_dim] * [2*embed_dim+aux_dim, 1] = [batch_size, 1]
        #if reg_hidden > 0: ,[batch_size, reg_hidden+aux_dim] * [reg_hidden+aux_dim, 1] = [batch_size, 1]
        q_pred = tf.matmul(last_output, last_w)

        ## first order reconstruction loss
        loss_recons = 2 * tf.trace(tf.matmul(tf.transpose(cur_message_layer), tf.sparse_tensor_dense_matmul(tf.cast(self.laplacian_param,tf.float32), cur_message_layer)))
        edge_num = tf.sparse_reduce_sum(tf.cast(self.n2nsum_param, tf.float32))
        loss_recons = tf.divide(loss_recons, edge_num)

        if self.IsPrioritizedSampling:
            self.TD_errors = tf.reduce_sum(tf.abs(self.target - q_pred), axis=1)    # for updating Sumtree
            if self.IsHuberloss:
                loss_rl = tf.losses.huber_loss(self.ISWeights * self.target, self.ISWeights * q_pred)
            else:
                loss_rl = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.target, q_pred))
        else:
            if self.IsHuberloss:
                loss_rl = tf.losses.huber_loss(self.target, q_pred)
            else:
                loss_rl = tf.losses.mean_squared_error(self.target, q_pred)

        loss = loss_rl + self.conf.HyperParameters.Alpha * loss_recons

        trainStep = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        #[node_cnt, batch_size] * [batch_size, embed_dim] = [node_cnt, embed_dim]
        rep_y = tf.sparse_tensor_dense_matmul(tf.cast(self.rep_global, tf.float32), y_potential)

      #  embed_s_a_all = tf.concat([cur_message_layer,rep_y],1)

        # # [node_cnt, embed_dim, embed_dim]
        temp1 = tf.matmul(tf.expand_dims(cur_message_layer, axis=2),tf.expand_dims(rep_y, axis=1))
        # [node_cnt embed_dim]
        Shape1 = tf.shape(cur_message_layer)
        # [batch_size, embed_dim], first transform
        embed_s_a_all = tf.reshape(tf.matmul(temp1, tf.reshape(tf.tile(cross_product,[Shape1[0],1]),[Shape1[0],Shape1[1],1])),Shape1)

        #[node_cnt, 2 * embed_dim]
        last_output = embed_s_a_all
        if self.reg_hidden > 0:
            #[node_cnt, 2 * embed_dim] * [2 * embed_dim, reg_hidden] = [node_cnt, reg_hidden1]
            hidden = tf.matmul(embed_s_a_all, h1_weight)
            #Relu, [node_cnt, reg_hidden1]
            last_output = tf.nn.relu(hidden)
            #[node_cnt, reg_hidden1] * [reg_hidden1, reg_hidden2] = [node_cnt, reg_hidden2]

        #[node_cnt, batch_size] * [batch_size, aux_dim] = [node_cnt, aux_dim]
        rep_aux = tf.sparse_tensor_dense_matmul(tf.cast(self.rep_global, tf.float32), self.aux_input)

        #if reg_hidden == 0: , [[node_cnt, 2 * embed_dim], [node_cnt, aux_dim]] = [node_cnt, 2*embed_dim + aux_dim]
        #if reg_hidden > 0: , [[node_cnt, reg_hidden], [node_cnt, aux_dim]] = [node_cnt, reg_hidden + aux_dim]
        last_output = tf.concat([last_output,rep_aux],1)

        #if reg_hidden == 0: , [node_cnt, 2 * embed_dim + aux_dim] * [2 * embed_dim + aux_dim, 1] = [node_cnt，1]
        #f reg_hidden > 0: , [node_cnt, reg_hidden + aux_dim] * [reg_hidden + aux_dim, 1] = [node_cnt，1]
        q_on_all = tf.matmul(last_output, last_w)

        return loss, trainStep, q_pred, q_on_all, tf.trainable_variables()

    def gen_graph(self, num_min, num_max):
        max_n = num_max
        min_n = num_min
        cur_n = np.random.randint(max_n - min_n + 1) + min_n
        if self.g_type == 'erdos_renyi':
            g = nx.erdos_renyi_graph(n=cur_n, p=0.15)
        elif self.g_type == 'powerlaw':
            g = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
        elif self.g_type == 'small-world':
            g = nx.connected_watts_strogatz_graph(n=cur_n, k=8, p=0.1)
        elif self.g_type == 'barabasi_albert':
            g = nx.barabasi_albert_graph(n=cur_n, m=4)
        return g

    def train(self):
        print("I am in train.")
        self.PrepareValidData()
        self.gen_new_graphs(self.conf.Simulation.NUM_MIN, self.conf.Simulation.NUM_MAX)

        for i in range(10):
            self.PlayGame(100, 1)
        self.TakeSnapShot()
        eps_start = 1.0
        eps_end = 0.05
        eps_step = 10000.0
        loss = 0

        save_dir = './models/Model_%s' % (self.g_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        VCFile = '%s/ModelVC_%d_%d.csv' % (save_dir, self.conf.Simulation.NUM_MIN, self.conf.Simulation.NUM_MAX)
        f_out = open(VCFile, 'w')
        t_train_start = time.time()
        for iter in range(self.conf.HyperParameters.MAX_ITERATION):
            start = time.time()
            ###########-----------------------normal training data setup(start) -----------------##############################
            if iter and iter % 5000 == 0:
                self.gen_new_graphs(self.conf.Simulation.NUM_MIN, self.conf.Simulation.NUM_MAX)
            eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - iter) / eps_step)

            if iter % 10 == 0:
                self.PlayGame(10, eps)
            if iter % 300 == 0:
                if (iter == 0):
                    N_start = start
                else:
                    N_start = N_end
                frac = 0.0
                test_start = time.time()
                for idx in range(self.conf.Simulation.n_valid):
                    frac += self.Test(idx)
                test_end = time.time()
                f_out.write('%.8f, %.4f\n' % (frac / self.conf.Simulation.n_valid, test_end - t_train_start))  # write vc into the file
                f_out.flush()
                print('iter %d, eps %.4f, average size of vc:%.6f' % (iter, eps, frac / self.conf.Simulation.n_valid))
                print('testing 200 graphs time: %.2fs' % (test_end - test_start))
                N_end = time.time()
                print('300 iterations total time: %.2fs\n' % (N_end - N_start))
                model_path = '%s/nrange_%d_%d_iter_%d.ckpt' % (save_dir, self.conf.Simulation.NUM_MIN, self.conf.Simulation.NUM_MAX, iter)
                self.SaveModel(model_path)
            if iter % self.conf.HyperParameters.UPDATE_TIME == 0:
                self.TakeSnapShot()
            self.Fit()
        f_out.close()

    def PrepareValidData(self):
        print('\ngenerating validation graphs...')
        result_degree = 0.0
        result_betweeness = 0.0
        for i in tqdm.tqdm(range(self.conf.Simulation.n_valid)):
            g = self.gen_graph(self.conf.Simulation.NUM_MIN, self.conf.Simulation.NUM_MAX)
            g_degree = g.copy()
            g_betweenness = g.copy()
            val_degree, sol = self.HXA(g_degree, 'HDA')
            result_degree += val_degree
            val_betweenness, sol = self.HXA(g_betweenness, 'HBA')
            result_betweeness += val_betweenness
            self.InsertGraph(g, is_test=True)
        print('Validation of HDA: %.6f'%(result_degree / self.conf.Simulation.n_valid))
        print('Validation of HBA: %.6f'%(result_betweeness / self.conf.Simulation.n_valid))

    def HXA(self, g, method):
        # 'HDA', 'HBA', 'HPRA', ''
        sol = []
        G = g.copy()
        while (nx.number_of_edges(G)>0):
            if method == 'HDA':
                dc = nx.degree_centrality(G)
            elif method == 'HBA':
                dc = nx.betweenness_centrality(G)
            elif method == 'HCA':
                dc = nx.closeness_centrality(G)
            elif method == 'HPRA':
                dc = nx.pagerank(G)
            keys = list(dc.keys())
            values = list(dc.values())
            maxTag = np.argmax(values)
            node = keys[maxTag]
            sol.append(int(node))
            G.remove_node(node)
        solution = sol + list(set(g.nodes()) ^ set(sol))
        solutions = [int(i) for i in solution]
        Robustness = self.utils.get_robustness(self.GenNetwork(g), solutions)
        return Robustness, sol

    def GenNetwork(self, g):    #networkx2four
        edges = g.edges()
        if len(edges) > 0:
            a, b = zip(*edges)
            A = np.array(a)
            B = np.array(b)
        else:
            A = np.array([0])
            B = np.array([0])
        return Graph(len(g.nodes()), len(edges), A, B)

    def InsertGraph(self, g, is_test):
        if is_test:
            t = self.ngraph_test
            self.ngraph_test += 1
            self.TestSet.insert_graph(t, self.GenNetwork(g))
        else:
            t = self.ngraph_train
            self.ngraph_train += 1
            self.TrainSet.insert_graph(t, self.GenNetwork(g))

    def gen_new_graphs(self, num_min, num_max):
        print('\ngenerating new training graphs...')
        self.ClearTrainGraphs()
        for i in tqdm.tqdm(range(1000)):
            g = self.gen_graph(num_min, num_max)
            self.InsertGraph(g, is_test=False)

    def ClearTrainGraphs(self):
        self.ngraph_train = 0
        self.TrainSet.clear()

    def ClearTestGraphs(self):
        self.ngraph_test = 0
        self.TestSet.clear()

    def PlayGame(self,n_traj, eps):
        self.Run_simulator(n_traj, eps, self.TrainSet, self.conf.Simulation.N_STEP)

    def Run_simulator(self, num_seq, eps, TrainSet, n_step):
        num_env = len(self.env_list)
        n = 0
        while n < num_seq:
            for i in range(num_env):
                if self.env_list[i].graph.num_nodes == 0 or self.env_list[i].is_terminal():
                    if self.env_list[i].graph.num_nodes > 0 and self.env_list[i].is_terminal():
                        n = n + 1
                        self.nStepReplayMem.add_env(self.env_list[i], n_step)

                    g_sample = TrainSet.sample()
                    self.env_list[i].s0(g_sample)
                    self.g_list[i] = self.env_list[i].graph
            if n >= num_seq:
                break

            Random = False
            if random.uniform(0, 1) >= eps:
                pred = self.PredictWithCurrentQNet(self.g_list, [env.action_list for env in self.env_list])
            else:
                Random = True

            for i in range(num_env):
                if (Random):
                    a_t = self.env_list[i].random_action()
                else:
                    a_t = self.argMax(pred[i])
                self.env_list[i].step(a_t)

    def argMax(self, scores):
        n = len(scores)
        pos = -1
        best = -10000000.0
        for i in range(n):
            if pos == -1 or scores[i] > best:
                pos = i
                best = scores[i]
        return pos

    def Max(self, scores):
        n = len(scores)
        pos = -1
        best = -10000000.0
        for i in range(n):
            if pos == -1 or scores[i] > best:
                pos = i
                best = scores[i]
        return best

    def SetupTrain(self, idxes, g_list, covered, actions, target):
        self.m_y = target
        self.inputs['target'] = self.m_y
        prepareBatchGraph = PrepareBatchGraph(self.conf.Embedding.aggregatorID)
        prepareBatchGraph.setup_train(idxes, g_list, covered, actions)
        self.inputs['action_select'] = prepareBatchGraph.act_select
        self.inputs['rep_global'] = prepareBatchGraph.rep_global
        self.inputs['n2nsum_param'] = prepareBatchGraph.n2nsum_param
        self.inputs['laplacian_param'] = prepareBatchGraph.laplacian_param
        self.inputs['subgsum_param'] = prepareBatchGraph.subgsum_param
        self.inputs['aux_input'] = prepareBatchGraph.aux_feat

    def SetupPredAll(self, idxes, g_list, covered):
        prepareBatchGraph = PrepareBatchGraph(self.conf.Embedding.aggregatorID)
        prepareBatchGraph.setup_pred_all(idxes, g_list, covered)
        self.inputs['rep_global'] = prepareBatchGraph.rep_global
        self.inputs['n2nsum_param'] = prepareBatchGraph.n2nsum_param
        # self.inputs['laplacian_param'] = prepareBatchGraph.laplacian_param
        self.inputs['subgsum_param'] = prepareBatchGraph.subgsum_param
        self.inputs['aux_input'] = prepareBatchGraph.aux_feat
        return prepareBatchGraph.idx_map_list

    def Predict(self, g_list, covered, isSnapSnot):
        n_graphs = len(g_list)
        for i in range(0, n_graphs, self.conf.Simulation.BATCH_SIZE):
            bsize = self.conf.Simulation.BATCH_SIZE
            if (i + self.conf.Simulation.BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            for j in range(i, i + bsize):
                batch_idxes[j - i] = j
            batch_idxes = np.int32(batch_idxes)

            idx_map_list = self.SetupPredAll(batch_idxes, g_list, covered)
            my_dict = {}
            my_dict[self.rep_global] = sparseMatrix2TFSparseMatrixValue(self.inputs['rep_global'])
            my_dict[self.n2nsum_param] = sparseMatrix2TFSparseMatrixValue(self.inputs['n2nsum_param'])
            my_dict[self.subgsum_param] = sparseMatrix2TFSparseMatrixValue(self.inputs['subgsum_param'])
            my_dict[self.aux_input] = self.inputs['aux_input']

            if isSnapSnot:
                result = self.session.run([self.q_on_allT], feed_dict=my_dict)
            else:
                result = self.session.run([self.q_on_all], feed_dict=my_dict)
            raw_output = result[0]
            pos = 0
            pred = []
            for j in range(i, i + bsize):
                idx_map = idx_map_list[j-i]
                cur_pred = np.zeros(len(idx_map))
                for k in range(len(idx_map)):
                    if idx_map[k] < 0:
                        cur_pred[k] = -self.conf.Simulation.inf
                    else:
                        cur_pred[k] = raw_output[pos]
                        pos += 1
                for k in covered[j]:
                    cur_pred[k] = -self.conf.Simulation.inf
                pred.append(cur_pred)
            assert (pos == len(raw_output))
        return pred

    def PredictWithCurrentQNet(self, g_list, covered):
        result = self.Predict(g_list, covered, False)
        return result

    def PredictWithSnapshot(self, g_list, covered):
        result = self.Predict(g_list, covered, True)
        return result

    def TakeSnapShot(self):
        self.session.run(self.UpdateTargetQNetwork)

    def Fit(self):
        sample = self.nStepReplayMem.sampling(self.conf.Simulation.BATCH_SIZE)
        ness = False
        for i in range(self.conf.Simulation.BATCH_SIZE):
            if (not sample.list_term[i]):
                ness = True
                break
        if ness:
            if self.IsDoubleDQN:
                double_list_pred = self.PredictWithCurrentQNet(sample.g_list, sample.list_s_primes)
                double_list_predT = self.PredictWithSnapshot(sample.g_list, sample.list_s_primes)
                list_pred = []
                # list_pred = [a[self.argMax(b)] for a, b in zip(double_list_predT, double_list_pred)]
                for i in range(len(double_list_pred)):
                    max_index = self.argMax(double_list_pred[i])
                    list_pred[i].append(double_list_predT[max_index])
            else:
                list_pred = self.PredictWithSnapshot(sample.g_list, sample.list_s_primes)

        list_target = np.zeros([self.conf.Simulation.BATCH_SIZE, 1])

        for i in range(self.conf.Simulation.BATCH_SIZE):
            q_rhs = 0
            if (not sample.list_term[i]):
                if self.IsDoubleDQN:
                    q_rhs = self.conf.Simulation.GAMMA * list_pred[i]
                else:
                    q_rhs = self.conf.Simulation.GAMMA * self.Max(list_pred[i])
            q_rhs += sample.list_rt[i]
            list_target[i] = q_rhs
            # list_target.append(q_rhs)

        return self.fit(sample.g_list, sample.list_st, sample.list_at, list_target)

    def fit(self, g_list, covered, actions, list_target):
        loss = 0.0
        n_graphs = len(g_list)
        for i in range(0, n_graphs, self.conf.Simulation.BATCH_SIZE):
            bsize = self.conf.Simulation.BATCH_SIZE
            if i + self.conf.Simulation.BATCH_SIZE > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            for j in range(i, i + bsize):
                batch_idxes[j - i] = j
            batch_idxes = np.int32(batch_idxes)

            self.SetupTrain(batch_idxes, g_list, covered, actions, list_target)
            my_dict = {}
            my_dict[self.action_select] = sparseMatrix2TFSparseMatrixValue(self.inputs['action_select'])
            my_dict[self.rep_global] = sparseMatrix2TFSparseMatrixValue(self.inputs['rep_global'])
            my_dict[self.n2nsum_param] = sparseMatrix2TFSparseMatrixValue(self.inputs['n2nsum_param'])
            my_dict[self.laplacian_param] = sparseMatrix2TFSparseMatrixValue(self.inputs['laplacian_param'])
            my_dict[self.subgsum_param] = sparseMatrix2TFSparseMatrixValue(self.inputs['subgsum_param'])
            my_dict[self.aux_input] = self.inputs['aux_input']
            my_dict[self.target] = self.inputs['target']

            result = self.session.run([self.loss, self.trainStep], feed_dict=my_dict)
            loss += result[0] * bsize
        return loss / len(g_list)

    def SaveModel(self, model_path):
        self.saver.save(self.session, model_path)
        print('model has been saved success!')

    def LoadModel(self, model_path):
        self.saver.restore(self.session, model_path)
        print('restore model from file successfully')

    def EvaluateRealData(self, model_file, data_test, save_dir, stepRatio=0.0025):  #测试真实数据
        test_name = data_test.split('/')[-1]
        save_dir_local = save_dir+'/StepRatio_%.4f' % stepRatio
        if not os.path.exists(save_dir_local):  # make dir
            os.makedirs(save_dir_local)
        result_file = '%s/%s' % (save_dir_local, test_name)
        g = nx.read_edgelist(data_test)
        with open(result_file, 'w') as f_out:
            print('testing')
            print('number of nodes:%d' % (nx.number_of_nodes(g)))
            print('number of edges:%d' % (nx.number_of_edges(g)))
            if stepRatio > 0:
                step = np.max([int(stepRatio*nx.number_of_nodes(g)), 1])  # step size
            else:
                step = 1
            self.InsertGraph(g, is_test=True)
            t1 = time.time()
            solution = self.GetSolution(0, step)
            t2 = time.time()
            solution_time = (t2 - t1)
            for i in range(len(solution)):
                f_out.write('%d\n' % solution[i])
        self.ClearTestGraphs()
        return solution, solution_time

    def GetSolution(self, gid, step=1):
        g_list = []
        self.test_env.s0(self.TestSet.get(gid))
        g_list.append(self.test_env.graph)
        sol = []
        sum_sort_time = 0
        it = 0
        while (not self.test_env.is_terminal()):
            print('Iteration:%d' % it)
            it += 1
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            start_time = time.time()
            batchSol = np.argsort(-list_pred[0])[:step]
            end_time = time.time()
            sum_sort_time += (end_time-start_time)
            for new_action in batchSol:
                if not self.test_env.is_terminal():
                    self.test_env.step_without_reward(new_action)
                    sol.append(new_action)
                else:
                    continue
        return sol

    def EvaluateSol(self, data_test, sol_file, strategyID=0, reInsertStep=20):
        #evaluate the robust given the solution, strategyID:0,count;2:rank;3:multipy
        g = nx.read_edgelist(data_test)
        g_inner = self.GenNetwork(g)
        print('number of nodes:%d' % nx.number_of_nodes(g))
        print('number of edges:%d' % nx.number_of_edges(g))
        nodes = list(range(nx.number_of_nodes(g)))
        sol = []
        for line in open(sol_file):
            sol.append(int(line))
        print('number of sol nodes:%d' % len(sol))
        sol_left = list(set(nodes) ^ set(sol))
        if strategyID > 0:
            start = time.time()
            if reInsertStep > 0 and reInsertStep < 1:
                step = np.max([int(reInsertStep*nx.number_of_nodes(g)),1]) #step size
            else:
                step = reInsertStep
            sol_reinsert = self.utils.re_insert(g_inner, sol, sol_left, strategyID, step)
            end = time.time()
            print('reInsert time:%.6f'%(end-start))
        else:
            sol_reinsert = sol
        solution = sol_reinsert + sol_left
        print('number of solution nodes:%d'%len(solution))
        Robustness = self.utils.get_robustness(g_inner, solution)
        MaxCCList = self.utils.max_wcc_sz_list
        return Robustness, MaxCCList

    def Test(self, gid):
        g_list = []
        self.test_env.s0(self.TestSet.get(gid))
        g_list.append(self.test_env.graph)
        sol = []
        while (not self.test_env.is_terminal()):
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            new_action = self.argMax(list_pred[0])
            self.test_env.step_without_reward(new_action)
            sol.append(new_action)
        nodes = list(range(g_list[0].num_nodes))
        solution = sol + list(set(nodes) ^ set(sol))
        Robustness = self.utils.get_robustness(g_list[0], solution)
        return Robustness

if __name__ == '__main__':
    f = FINDER()
    f.train()
