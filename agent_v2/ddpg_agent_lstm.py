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

        # syncronize actor and target
        for var1, var2 in zip(self.model.parameters(), self.target.parameters()):
            var2 = var1

    def create_actor_network(self):
        
        # neural featurizer parameters
        h1 = 256
        h2 = 128
        h3 = 128
        
        class LSTM_Last(nn.Module):
            
            def __init__(self, *args, **kwargs):
                super(LSTM_Last, self).__init__()
                # self.lstm = nn.LSTM(*args, **kwargs)
                self.lstm = nn.RNN(*args, **kwargs)
            
            def forward(self, x):
                x, _ = self.lstm(x)
                return x[:, -1, :]
        
        model = nn.Sequential(
            LSTM_Last(2, h1, 1, nonlinearity='relu', batch_first=True),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, 1),
            nn.Tanh(),
        )
        
        return model 
    
    def act(self, state):
        state = torch.Tensor([state]).float().to(device)      
        action = self.model(state).detach().cpu().item()
        return action

    def act_target(self, states):
        # called at calculating q_targets
        action = self.target(states)
        return action

    def update_target_network(self):
        
        with torch.no_grad():
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
        
        # synchronize critic and target
        with torch.no_grad():
            for var1, var2 in zip(self.model.parameters(), self.target.parameters()):
                var2 = var1
        
    def create_critic_network(self):
        
        # neural featurizer parameters
        h1 = 256
        h2 = 128
        h3 = 128
        
        class LSTM_Last(nn.Module):
            
            def __init__(self, *args, **kwargs):
                super(LSTM_Last, self).__init__()
                # self.lstm = nn.LSTM(*args, **kwargs)
                self.lstm = nn.RNN(*args, **kwargs)
                self.model = nn.Sequential(
                                nn.Linear(h1+1, h2),
                                nn.ReLU(),
                                nn.Linear(h2, h3),
                                nn.ReLU(),
                                nn.Linear(h3, 1),
                            )
            
            def forward(self, x, a):
                x, _ = self.lstm(x)
                x = torch.cat([x[:,-1,:], a], 1)
                return self.model(x)
        
        model = LSTM_Last(2, h1, 1, nonlinearity='relu', batch_first=True)

        return model

    def get_q_values(self, states, actions):
        
        return self.model(states, actions).squeeze()

    def get_target_q_values(self, states, actions):

        return self.target(states, actions).squeeze()
    
    def update_target_network(self):
        
        with torch.no_grad():
            for var1, var2 in zip(self.model.parameters(), self.target.parameters()):
                var2 = self._tau * var1 + (1-self._tau) * var2

class DDPGAgent():
    def __init__(self, env, buffer_size = 1000000, discount_rate = 0.99, batch_size = 128, tau = 0.001, 
                 actor_lr = 1e-5, critic_lr = 1e-4, update_steps = 100,
                 quiet = True):
        
        self.actor = Actor(env, learning_rate = actor_lr, quiet = quiet, tau = tau)
        self.critic = Critic(env, learning_rate = critic_lr, quiet = quiet, tau = tau)
        
        self._batch_size = batch_size
        self._discount_rate = discount_rate
        self._update_steps = update_steps
        
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
        
        self.memory.append([state, action, reward, next_state, done])
    
    def train(self):
        
        actor_grad = 0
        critic_grad = 0
        for _ in range(self._update_steps):
            batch = random.sample(self.memory, self._batch_size)
            
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.Tensor(states).float().to(device)
            actions = torch.Tensor(actions).float().to(device).unsqueeze(1)
            rewards = torch.Tensor(rewards).float().to(device)
            next_states = torch.Tensor(next_states).float().to(device)
            dones = torch.Tensor(dones).float().to(device)

            target_actions = self.actor.act_target(next_states)
            q_targets = rewards + (1.0 - dones) * self._discount_rate * self.critic.get_target_q_values(next_states, target_actions)
            
            mse_loss = nn.MSELoss()
            
            self.critic.optimizer.zero_grad()
            q_values = self.critic.get_q_values(states, actions)
            critic_loss = mse_loss( q_values, q_targets )
            critic_loss.backward()
            critic_grad += self.get_gradient_norm(self.critic.model)
            self.critic.optimizer.step()

            # freeze critic network for actor update, to avoid unnecessary grad calculation
            for p in self.critic.model.parameters():
                p.requires_grad = False

            self.actor.optimizer.zero_grad()
            actor_loss = -self.critic.get_q_values( states, self.actor.model(states) ).mean()
            # print("actor loss before update: {}".format(actor_loss.item()))
            actor_loss.backward()
            actor_grad += self.get_gradient_norm(self.actor.model)
            self.actor.optimizer.step()
            # actor_loss = -self.critic.get_q_values( states, self.actor.model(states) ).mean()
            # print("actor loss after update: {}".format(actor_loss.item()))

            # unfreeze critic network after actor update
            for p in self.critic.model.parameters():
                p.requires_grad = True

            self.actor.update_target_network()
            self.critic.update_target_network()
        
        return actor_grad / self._update_steps, critic_grad / self._update_steps
