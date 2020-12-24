import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, state_dim):  
        super(Model, self).__init__()  

        h1 = 256
        h2 = 128
        h3 = 128
        
        self.hidden_1 = nn.Linear(state_dim, h1)
        self.hidden_2 = nn.Linear(h1, h2)
        self.hidden_3 = nn.Linear(h2, h3)

        self.output = nn.Linear(h3, 1)

    def forward(self, x):
        x = F.tanh(self.hidden_1(x))
        x = F.tanh(self.hidden_2(x))
        x = F.tanh(self.hidden_3(x))
        return F.tanh(self.output(x))  

class StochasticPolicyGradientAgent():
    """
    A Gaussian Policy Gradient based agent implementation
    """
    def __init__(self, env, learning_rate = 0.001, discount_rate = 0.99, batch_size = 1, quiet = True):
        
        self._env = env
        self._batch_size = batch_size
        self._discount_rate = discount_rate
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []
        self._log_prob_buffer = []
        self._quiet = quiet
        
        state_dim = np.prod(np.array(env.observation_space.shape))

        self._mu_model = Model(state_dim).to(device)
        self._sigma_model = Model(state_dim).to(device)
        
        self._optimizer = torch.optim.Adam(params=[ { 'params' : self._mu_model.parameters() },
                                                    { 'params' : self._sigma_model.parameters() }
                                                    ])

        #Sampling action from distribuition
        
        # self._normal_dist = tf.contrib.distributions.Normal(self._mu, self._sigma)
        # self._action = self._normal_dist.sample()
        
        # #Computing loss function
        
        # self._discounted_rewards = tf.placeholder(tf.float32, (None, 1), name="discounted_rewards")
        # self._taken_actions = tf.placeholder(tf.float32, (None, 1), name="taken_actions")
        
        # self._loss = -tf.reduce_mean(tf.log(1e-5 + self._normal_dist.prob(self._taken_actions)) * self._discounted_rewards,0)
         
        # self._train_op = self._optimizer.minimize(self._loss)        
        
        # self._sess.run(tf.global_variables_initializer())
                
    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        mu = self._mu_model(state)
        log_sigma = self._sigma_model(state)

        dist = Normal(mu, torch.exp(log_sigma))
        action = dist.sample()
        action = torch.clamp(action, self._env.action_space.low[0], self._env.action_space.high[0])
        log_prob = dist.log_prob(action)

        self._log_prob_buffer.append(log_prob)

        if not self._quiet:
            print("Sigma: {}, Mu: {}, Action: {}".format(sigma, mu, action))
        
        return action.item()
    
    def train(self): 
        rewards = self._discount_rewards().tolist()
        #rewards -= np.mean(rewards)
        samples = []
        for t in range(len(rewards)):
            samples.append([self._log_prob_buffer[t], rewards[t]])
            
            
        np.random.shuffle(samples)
        batches = []
        for i in range(0, len(samples), self._batch_size):
            batches.append(samples[i:i + self._batch_size])
            
        for batch in batches:
            log_probs_batch = [row[0] for row in batch]
            rewards_batch = [row[1] for row in batch]

            loss = [log_prob*r for log_prob, r in batch ]
            loss = torch.cat(loss).sum()
            
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            
        
        #After applying gradients
        self._state_buffer  = []
        self._reward_buffer = []
        self._log_prob_buffer = []
    
    def store_step(self, state, reward):
        self._state_buffer.append(state)
        self._reward_buffer.append(np.array(reward))
        
    def _discount_rewards(self):
        r = 0
        N = len(self._reward_buffer)
        discounted_rewards = np.zeros(N)
        for t in reversed(range(N)):
            r = r + self._reward_buffer[t] * self._discount_rate
            discounted_rewards[t] = r
        return discounted_rewards