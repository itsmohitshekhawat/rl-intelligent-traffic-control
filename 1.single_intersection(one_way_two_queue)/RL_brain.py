import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.001,
            reward_decay=0.99,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment else self.epsilon_max

        self.save_file = './weights/model.ckpt'

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self._build_net()

        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        self.cost_his = []

    def _build_net(self):
        n_l1 = 100
        w_initializer = tf.random_normal_initializer(0., 0.3)
        b_initializer = tf.constant_initializer(0.1)

        # EVAL NET
        self.s = tf.placeholder(tf.float32, [None, self.n_features])
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions])

        with tf.variable_scope('eval_net'):
            w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer)
            b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer)
            l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer)
            b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer)
            self.q_eval = tf.matmul(l1, w2) + b2

        self.loss = tf.reduce_mean(tf.square(self.q_target - self.q_eval))
        self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # TARGET NET
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features])

        with tf.variable_scope('target_net'):
            w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer)
            b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer)
            l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer)
            b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer)
            self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        s = s.reshape(-1)
        s_ = s_.reshape(-1)

        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # ✅ FIXED SHAPE
        observation = observation.reshape(1, -1).astype(np.float32)

        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def learn(self):
        if not hasattr(self, 'memory_counter') or self.memory_counter < self.batch_size:
            return

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        sample_index = np.random.choice(
            min(self.memory_counter, self.memory_size),
            size=self.batch_size
        )

        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],
                self.s: batch_memory[:, :self.n_features],
            })

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        self.sess.run(self._train_op, feed_dict={
            self.s: batch_memory[:, :self.n_features],
            self.q_target: q_target
        })

        self.learn_step_counter += 1

    def store(self):
        self.saver.save(self.sess, self.save_file)
        print("✅ Model saved")

    def restore(self):
        import os
        if os.path.exists(self.save_file + '.index'):
            self.saver.restore(self.sess, self.save_file)
            print("✅ Model loaded successfully")
        else:
            print("⚠️ No model found, running fresh model")