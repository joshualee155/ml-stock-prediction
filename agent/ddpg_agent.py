import tensorflow as tf
import numpy as np
import random
import math 

class DDPGAgent():
    def __init__(self, env, discount_rate = 0.99, batch_size = 128, tau = 0.001, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.1, 
                 actor_lr = 1e-5, critic_lr = 1e-4, quiet = True):
        
        self.actor = Actor(env, learning_rate = actor_lr, quiet = quiet, 
                           epsilon=epsilon, epsilon_min=epsilon_min, epsilon_log_decay=epsilon_log_decay, tau = tau)
        self.critic = Critic(env, self.actor, learning_rate = critic_lr, quiet = quiet, tau = tau)
        
        self._batch_size = batch_size
        self._discount_rate = discount_rate
        
        # Memory
        self._state_buffer  = []
        self._action_buffer = []
        self._q_buffer = []
        
    
    def store_step(self, state, action, reward, next_state, done):
        self._state_buffer.append(state)
        self._action_buffer.append(action)
        
        next_action = self.actor.act_target([next_state])
        if not done:
            q_next = self.critic.predict_target_q([next_state], [[next_action]])
            q_expected = reward + self._discount_rate * q_next
        else:
            q_expected = reward
        self._q_buffer.append([q_expected])
    
    def train(self):
        self._action_buffer = [[a] for a in self._action_buffer]
        samples = []
        for t in range(len(self._state_buffer)):
            samples.append([self._state_buffer[t], self._action_buffer[t], self._q_buffer[t]])
        np.random.shuffle(samples)
        batches = []
        
        for i in range(0, len(samples), self._batch_size):
            batches.append(samples[i:i + self._batch_size])
        
        for batch in batches:
            states_batch = [row[0] for row in batch]
            actions_batch = [row[1] for row in batch]
            q_batch = [row[2] for row in batch]
            
            self.critic.train(states_batch, q_batch, actions_batch)
            action_grads_batch = self.critic.get_action_grads(states_batch, actions_batch)
            action_grads_batch = [[a] for a in action_grads_batch]
            self.actor.train(states_batch, action_grads_batch)
            
        self.actor.update_target_network()
        self.critic.update_target_network()
        
        #After applying gradients
        self._state_buffer  = []
        self._action_buffer = []
        self._q_buffer = []
            
class Actor():

    def __init__(self, env, learning_rate = 0.0001, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.1, tau = 0.001, quiet = True):
        
        self._optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        self._sess = tf.Session()
        self.env = env
        self._quiet = quiet
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self._tau = tau
        
        self.state_dim = np.prod(np.array(env.observation_space.shape))
        
        self.init = tf.contrib.layers.xavier_initializer()

        #Actor Network
        with tf.variable_scope('actor') as actor_vs:
            self._state, self.action = self.create_actor_network()
        
        self.network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=actor_vs.name)
        
        #Target Network
        with tf.variable_scope('actor_target') as actor_target_vs:
            self._target_state, self.target_action = self.create_actor_network()
        
        self.target_network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=actor_target_vs.name)
        
        
        
        self.update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self.network_params[i],self._tau)
                                                                                  + tf.multiply(self.target_network_params[i],
                                                                                                1. - self._tau))
                                             for i in range(len(self.target_network_params))]
        
        #Computing training op
        
        self.trainable_vars = tf.trainable_variables()
        
        self._action_gradients = tf.placeholder(tf.float32, [None, 1], name="action_grad")
        
        self._var_grads = tf.gradients(self.action, self.network_params, -self._action_gradients, )
        
        self._grad_norm = tf.global_norm( self._var_grads )

        self._train_op = self._optimizer.apply_gradients(zip(self._var_grads,self.network_params))      
        
        # Add summary place holders
        self.add_summary()
        self.init_averages()

        #Initializing
        self._sess.run(tf.global_variables_initializer())
        self._sess.run([self.target_network_params[i].assign(self.network_params[i]) 
                        for i in range(len(self.target_network_params))])

    def add_summary(self):
        """
        Tensorboard stuff.

        You don't have to change or use anything here.
        """
        # extra placeholders to log stuff from python
        self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
        self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
        self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

        self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg Reward", self.avg_reward_placeholder)
        tf.summary.scalar("Max Reward", self.max_reward_placeholder)
        tf.summary.scalar("Std Reward", self.std_reward_placeholder)
        tf.summary.scalar("Eval Reward", self.eval_reward_placeholder)

        # logging
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter( 'result/actor_model/', self._sess.graph)    

    def init_averages(self):
        """
        Defines extra attributes for tensorboard.

        You don't have to change or use anything here.
        """
        self.avg_reward = 0.
        self.max_reward = 0.
        self.std_reward = 0.
        self.eval_reward = 0.

    def update_averages(self, rewards, scores_eval):
        """
        Update the averages.

        You don't have to change or use anything here.

        Args:
            rewards: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def record_summary(self, t):
        """
        Add summary to tensorboard

        You don't have to change or use anything here.
        """

        fd = {
        self.avg_reward_placeholder: self.avg_reward,
        self.max_reward_placeholder: self.max_reward,
        self.std_reward_placeholder: self.std_reward,
        self.eval_reward_placeholder: self.eval_reward,
        }
        summary = self._sess.run(self.merged, feed_dict=fd)
        # tensorboard stuff
        self.file_writer.add_summary(summary, t)  

    def create_actor_network(self):
        
        # neural featurizer parameters
        h1 = 256
        h2 = 128
        h3 = 128
        
        state = tf.placeholder(tf.float32,
                                     shape=(None, self.state_dim))
        
        action_hidden = tf.layers.dense(state, h1, 
                                    activation = tf.tanh,
                                    kernel_initializer=self.init)
        action_hidden_2 = tf.layers.dense(action_hidden, h2, 
                                      activation = tf.tanh,
                                      kernel_initializer=self.init)
        action_hidden_3 = tf.layers.dense(action_hidden_2, h3, 
                                      activation = tf.tanh, 
                                      kernel_initializer=self.init)
        action = tf.layers.dense(action_hidden_3, 1,
                                   activation = tf.tanh,
                                   kernel_initializer=self.init)
        action = tf.squeeze(action)
        
        return state, action
    
    def act(self, state,  step = None):
        
        # epsilon greedy
        if step is not None:
            epsilon = max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((step + 1) * self.epsilon_decay)))
            action = self.env.action_space.sample()[0].item() if (np.random.random() <= epsilon) else self._sess.run(self.action, feed_dict={self._state: state})
        else:
            action = self._sess.run(self.action, feed_dict={self._state: state})
        
        if not self._quiet:
            print("Action: {}".format(action))
        
        return action

    def act_target(self, state):

        action = self._sess.run(self.target_action, feed_dict={self._target_state: state})
        
        return action
    
    def train(self, states_batch, actions_grads_batch): 

        feed_dict={
            self._state: states_batch,
            self._action_gradients: actions_grads_batch}
        grad_norm = self._sess.run( self._grad_norm, feed_dict = feed_dict )
        print( "Actor gradadient norm: {}".format( grad_norm ) )
        self._sess.run([self._train_op], feed_dict=feed_dict)
        
    def update_target_network(self):
        self._sess.run(self.update_target_network_params)

class Critic():

    def __init__(self, env, actor, learning_rate = 0.001, tau = 0.001, quiet = True):
        
        self._optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        self._sess = tf.Session()
        self._env = env
        self._quiet = quiet
        self._tau = tau
        
        self._state_dim = np.prod(np.array(env.observation_space.shape))
        
        self.init = tf.contrib.layers.xavier_initializer()
        
        #Q function
        with tf.variable_scope('q') as q_vs:
            self._state, self._action, self.q = self.create_actor_network()
        
        self.network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=q_vs.name)
        
        #Setting up target function
        with tf.variable_scope('q_target') as q_target_vs:
            self._target_state, self._target_action, self.target_q = self.create_actor_network()
        
        self.target_network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=q_target_vs.name)
        
        self.update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self.network_params[i],self._tau)
                                                                                  + tf.multiply(self.target_network_params[i],
                                                                                                1. - self._tau))
                                             for i in range(len(self.target_network_params))]
        
        #Computing training op
        
        self._q_expected = tf.placeholder(tf.float32,shape=[None,1], name="q_expected")
        
        self._loss = tf.losses.mean_squared_error(self._q_expected,self.q)
        
        self._train_op = self._optimizer.minimize(self._loss)
        
        self._action_gradients = tf.squeeze(tf.gradients(self.q ,self._action))
        
        #Initializing
        self._sess.run(tf.global_variables_initializer())
        
        self._sess.run([self.target_network_params[i].assign(self.network_params[i]) 
                        for i in range(len(self.target_network_params))])
        
    def create_actor_network(self):
        
        state = tf.placeholder(tf.float32,
                               shape=(None, self._state_dim))

        action = tf.placeholder(tf.float32,shape=[None,1])
        
        # neural featurizer parameters
        h1 = 256
        h2 = 128
        h3 = 128
        
        
        q_hidden = tf.layers.dense(tf.concat([state, action], 1), h1, 
                                    activation = tf.nn.relu,
                                    kernel_initializer=self.init)
        q_hidden_2 = tf.layers.dense(q_hidden, h2, 
                                      activation = tf.nn.relu,
                                      kernel_initializer=self.init)
        q_hidden_3 = tf.layers.dense(q_hidden_2, h3, 
                                      activation = tf.nn.relu,
                                      kernel_initializer=self.init)
        q = tf.layers.dense(q_hidden_3, 1,
                            activation = None,
                            kernel_initializer=self.init)
        
        q = tf.squeeze(q)
        
        return state, action, q
    
    def predict_q(self, state, action):
        
        q = self._sess.run(self.q, feed_dict={
            self._state: state,
            self._action: action})
        return q
    
    def predict_target_q(self, state, action):
        
        q = self._sess.run(self.target_q, feed_dict={
            self._target_state: state,
            self._target_action: action})
        return q
    
    def get_action_grads(self, state, action):
        grads = self._sess.run(self._action_gradients, feed_dict={
            self._state: state,
            self._action: action})
        return grads
    
    def train(self, states_batch, q_batch, actions_batch): 
        feed_dict={
            self._state: states_batch,
            self._q_expected: q_batch,
            self._action: actions_batch}
        pre_train_loss = self._sess.run( self._loss, feed_dict = feed_dict )
        self._sess.run([self._train_op], feed_dict=feed_dict)
        post_train_loss = self._sess.run( self._loss, feed_dict = feed_dict )
        print( "Pre-train loss: {:.2f}|Post-train loss: {:.2f}".format( pre_train_loss, post_train_loss ) )
        
    def update_target_network(self):
        self._sess.run(self.update_target_network_params)        
        