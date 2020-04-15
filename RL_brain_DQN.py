import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)
num_uav = 10

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_feature_sum,
            n_actions_sum,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=0.1,
            output_graph=False,
    ):
        self.n_actions_sum = n_actions_sum
        self.n_features_sum = n_feature_sum
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size_sim = memory_size*10
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, 51),dtype=object)

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features_sum], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions_sum], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, n_l2, n_l3, n_l4, n_l5, n_l6, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 50, 50, 50, 50, 50, 50, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features_sum, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, n_l3], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, n_l2], initializer=b_initializer, collections=c_names)
                l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)

            with tf.variable_scope('l4'):
                w4 = tf.get_variable('w4', [n_l3, n_l4], initializer=w_initializer, collections=c_names)
                b4 = tf.get_variable('b4', [1, n_l4], initializer=b_initializer, collections=c_names)
                l4 = tf.nn.relu(tf.matmul(l3, w4) + b4)

            with tf.variable_scope('l5'):
                w5 = tf.get_variable('w5', [n_l4, n_l5], initializer=w_initializer, collections=c_names)
                b5 = tf.get_variable('b5', [1, n_l5], initializer=b_initializer, collections=c_names)
                l5 = tf.nn.relu(tf.matmul(l4, w5) + b5)

            with tf.variable_scope('l6'):
                w6 = tf.get_variable('w6', [n_l5, n_l6], initializer=w_initializer, collections=c_names)
                b6 = tf.get_variable('b6', [1, n_l6], initializer=b_initializer, collections=c_names)
                l6 = tf.nn.relu(tf.matmul(l5, w6) + b6)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l7'):
                w7 = tf.get_variable('w7', [n_l1, self.n_actions_sum], initializer=w_initializer, collections=c_names)
                b7 = tf.get_variable('b7', [1, self.n_actions_sum], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l6, w7) + b7

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features_sum], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features_sum, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, n_l3], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, n_l2], initializer=b_initializer, collections=c_names)
                l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)

            with tf.variable_scope('l4'):
                w4 = tf.get_variable('w4', [n_l3, n_l4], initializer=w_initializer, collections=c_names)
                b4 = tf.get_variable('b4', [1, n_l4], initializer=b_initializer, collections=c_names)
                l4 = tf.nn.relu(tf.matmul(l3, w4) + b4)

            with tf.variable_scope('l5'):
                w5 = tf.get_variable('w5', [n_l4, n_l5], initializer=w_initializer, collections=c_names)
                b5 = tf.get_variable('b5', [1, n_l5], initializer=b_initializer, collections=c_names)
                l5 = tf.nn.relu(tf.matmul(l4, w5) + b5)

            with tf.variable_scope('l6'):
                w6 = tf.get_variable('w6', [n_l5, n_l6], initializer=w_initializer, collections=c_names)
                b6 = tf.get_variable('b6', [1, n_l6], initializer=b_initializer, collections=c_names)
                l6 = tf.nn.relu(tf.matmul(l5, w6) + b6)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l7'):
                w7 = tf.get_variable('w7', [n_l1, self.n_actions_sum], initializer=w_initializer, collections=c_names)
                b7 = tf.get_variable('b7', [1, self.n_actions_sum], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l6, w7) + b7

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0


        index = self.memory_counter % self.memory_size
        transition = np.hstack((s,a,r,s_))
        self.memory[index, : ] = transition

#        transition = [s,[a,r],s_]
#        print('transition: ',transition)
#        print('transition_shape: ', np.array(transition).shape)

#        transition = np.hstack((s, [a, r], s_))

#        print('s_list:  ', s)
#        print('[a,r]_list ; ', [a,r])

#        print('transition_shape: ', transition.shape)

        # replace the old memory with new memory
#        index = self.memory_counter % self.memory_size
#        print(index)
#        self.memory[index, :] = transition
#        print('memory: ',self.memory)

        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
#        print('observation: ',observation)
#        print('observation_type: ',type(observation))
        actions = []
#        print(observation)
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
#        print('actions_value: ',actions_value)
        for num in range(num_uav):
            if np.random.uniform() < self.epsilon:
                # forward feed the observation and get q value for every actions,通过状态算出动作值
#                print('num: ',num,'  max_action: ',np.argmax(actions_value[num]))
#              print('num is: ',num)
#                print('actions_value: ', actions_value.shape)
                actions_uav = actions_value[: ,num*5: num*5 + 5]
                actions.append(np.argmax(actions_uav))
#               print('here')
            else:
#                print('num: ', num, '  max_action: ', np.random.randint(0, self.n_actions))
                actions.append(np.random.randint(0, self.n_actions))
#        print(actions)
        return actions
        '''
        actions_value = []
        action = []
        print('observation_list: ',observation)
        '''
        '''
        for num in range(num_uav):
            print('old_observation: ',observation[num])
            obs = observation[num]
            obs = obs[np.newaxis, :]
            observation[num] = np.array(observation[num])
            print('obs: ', obs)

            if np.random.uniform() < self.epsilon:
                # forward feed the observation and get q value for every actions
                print('observ_num: ',observation[num])
                actions_value.append(self.sess.run(self.q_eval, feed_dict={self.s: obs}))
                print(actions_value)
                print(np.argmax(actions_value[num]))
                action.append(np.argmax(actions_value[num]))
                print(action)
            else:
                actions_value.append(-9999)
                action.append(np.random.randint(0, self.n_actions_inde))
        '''

        # to have batch dimension when feed into tf placeholder
        '''        
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        '''

    def learn(self):
#        print('epsilon: ', self.epsilon)
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
#            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
#        print(len(batch_memory))
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features_sum:],  # fixed params
                self.s: batch_memory[:, :self.n_features_sum],  # newest params
            })
        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()
        q_calu = q_next.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
#        print('batch_memo: ', batch_memory[:,self.n_features_sum:self.n_features_sum+num_uav])
        eval_act_index = batch_memory[:, self.n_features_sum: self.n_features_sum+num_uav].astype(int)
#        print('eval_act_index: ', eval_act_index)
        reward = batch_memory[:, self.n_features_sum + num_uav + 1]

        for num in range(num_uav):
#            print('q_tar_shape: ',q_target.shape)
            q_target_uav = q_target[:, num*5 : num*5 + 5 ]
            q_next_uav = q_calu[:, num*5 : num*5 +5]
#            print('q_tar_u_shape: ',q_target_uav.shape)
#            print('q_next_shape:', q_next_uav.shape)
#            print(type(q_target_uav))
#            print(q_target_uav[:,eval_act_index[num]])
#          print('eval_act_index[num: ]',eval_act_index[:,num])
            q_target_uav[:, eval_act_index[:,num]] = reward + self.gamma * np.max(q_next_uav, axis=1)
#            print('q_target_shape: ', q_target[:, num*5 : num*5 +5])

            q_target[:, num*5 : num*5 +5] = q_target_uav

#        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features_sum],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        '''
                sample_index_sim = []
        for n in range(len(sample_index)):
            si_10 = sample_index[n]
            for m in range(10):
                sample_index_sim.append(si_10 // 10 * 10 + m)
        print('memo_counter: ', self.memory_counter)
        print('sample_index: ', len(sample_index))
        print('sample_index_sim: ', len(sample_index_sim))
        batch_memory = self.memory[sample_index_sim, :]
        print('batch_memory: ', batch_memory, '     lens of bm: ', len(batch_memory))
        batch_memory_s_ = batch_memory[:, -self.n_features:]
        batch_memory_s = batch_memory[:, :self.n_features]
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        print('sample_index_sim: ',sample_index_sim)

        print(self.s)
        print('batch: ',batch_memory)
        print('batch_shape: ',np.shape(batch_memory_s_))
        print('batch_memory_s_ shape: ',batch_memory[:, -1:])
        print('batch_memory_s shape', batch_memory_s_)
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory_s_,  # fixed params
                self.s: batch_memory_s,  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size*10, dtype=np.int32)

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        '''

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.title('Cost')
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()