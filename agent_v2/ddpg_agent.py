import torch
from torch import nn
import numpy as np
import random
import math 
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor():

    def __init__(self, env, learning_rate = 0.0001, tau = 0.001, quiet = True):
        
        self.env = env
        self._quiet = quiet
        self._tau = tau

        self.state_dim = np.prod(np.array(env.observation_space.shape))
        
        # Actor network
        self.model = self.create_actor_network().to(device)
        # Target network
        self.target = self.create_actor_network().to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)

        self.update_target_network()

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
        
        model = nn.Sequential(
            nn.Linear(self.state_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, h3),
            nn.Tanh(),
            nn.Linear(h3, 1),
            nn.Tanh(),
        )
        
        return model 
    
    def act(self, state):
        state = torch.from_numpy(state).float().to(device)      
        action = self.model(state).detach().cpu().item()
        return action

    def act_target(self, state):
        state = torch.from_numpy(state).float().to(device)
        action = self.target(state).detach().cpu().item()
        return action

    def update_target_network(self):
        
        for var1, var2 in zip(self.model.parameters(), self.target.parameters()):
            var2 = self._tau * var1 + (1-self._tau) * var2


class Critic():

    def __init__(self, env, learning_rate = 0.001, tau = 0.001, quiet = True):
        
        self._env = env
        self._quiet = quiet
        self._tau = tau
        
        self._state_dim = np.prod(np.array(env.observation_space.shape))

        # critic network
        self.model = self.create_critic_network().to(device)

        # critic target network
        self.target = self.create_critic_network().to(device)

        # optimizer only applies to model network
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        
        self.update_target_network()
        
    def create_critic_network(self):
        
        # neural featurizer parameters
        h1 = 256
        h2 = 128
        h3 = 128
        
        model = nn.Sequential(
            nn.Linear(self._state_dim+1, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, 1),
        )

        return model

    def get_q_values(self, states, actions):
        inputs = torch.cat([states, actions], 1)

        return self.model(inputs).squeeze()

    def get_target_q_values(self, states, actions):
        states = torch.Tensor(states).float().to(device)
        actions = torch.Tensor(actions).float().to(device)
        inputs = torch.cat([states, actions.unsqueeze(1)], 1)

        return self.target(inputs).squeeze().item()
    
    def update_target_network(self):
        
        for var1, var2 in zip(self.model.parameters(), self.target.parameters()):
            var2 = self._tau * var1 + (1-self._tau) * var2
        

class DDPGAgent():
    def __init__(self, env, buffer_size = 1000000, discount_rate = 0.99, batch_size = 128, tau = 0.001, 
                 actor_lr = 1e-5, critic_lr = 1e-4, quiet = True):
        
        self.actor = Actor(env, learning_rate = actor_lr, quiet = quiet, tau = tau)
        self.critic = Critic(env, learning_rate = critic_lr, quiet = quiet, tau = tau)
        
        self._batch_size = batch_size
        self._discount_rate = discount_rate
        
        # Memory
        self.memory = deque( maxlen = buffer_size )
        
    def get_gradient_norm(self, model):

        total_norm = 0.0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        return total_norm

    def store_step(self, state, action, reward, next_state, done):
        
        next_action = self.actor.act_target(next_state)
        if not done:
            q_next = self.critic.get_target_q_values([next_state], [next_action])
            q_expected = reward + self._discount_rate * q_next
        else:
            q_expected = reward

        self.memory.append([state, action, q_expected])
    
    def train(self):
        
        batch = random.sample(self.memory, self._batch_size)
        
        states, actions, qs_expected = zip(*batch)

        states = torch.Tensor(states).float().to(device)
        actions = torch.Tensor(actions).float().to(device).unsqueeze(1)
        qs_expected = torch.Tensor(qs_expected).float().to(device)
        
        mse_loss = nn.MSELoss()
        
        q_values = self.critic.get_q_values(states, actions)
        critic_loss = mse_loss( q_values, qs_expected )

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        critic_grad = self.get_gradient_norm(self.critic.model)
        self.critic.optimizer.step()

        actor_loss = -self.critic.get_q_values( states, self.actor.model(states) ).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        actor_grad = self.get_gradient_norm(self.actor.model)
        self.actor.optimizer.step()

        self.actor.update_target_network()
        self.critic.update_target_network()
        
        return actor_grad, critic_grad
