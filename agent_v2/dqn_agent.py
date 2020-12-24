import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, state_dim):  
        super(Model, self).__init__()

        h1 = 512
        h2 = 256
        h3 = 128
        
        self.hidden_1 = nn.Linear(state_dim, h1)
        self.hidden_2 = nn.Linear(h1, h2)
        self.hidden_3 = nn.Linear(h2, h3)

        self.output = nn.Linear(h3, 3)

    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        return self.output(x)

class DQNAgent():
    def __init__(self, env, gamma=0.99, buffer_size = 1000000,
        epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.999, 
        alpha=1e-4, alpha_decay=0.001, batch_size=128, quiet=False):
        
        self.env = env
        self.memory = deque(maxlen = buffer_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self._batch_size = batch_size
        self.quiet = quiet
        self._state_dim = np.prod(np.array(env.observation_space.shape))
        
        self.model = Model(self._state_dim).to(device)
        self.target = Model(self._state_dim).to(device)
        # Align model network and target network
        self.copy_target(1.0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha)

    def store_step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, step = None):
        state = torch.from_numpy(state).float().to(device)
        if step is not None:
            # epsilon = max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((step + 1) * self.epsilon_decay)))
            epsilon = self.epsilon
            q_values = self.model(state)
            q_values = q_values.detach().cpu().numpy()
            random_action = np.random.choice(3) - 1 # action space: -1, 0, 1
            q_max_action = np.argmax(q_values) - 1
            action = random_action if (np.random.random() <= epsilon) else q_max_action
            return action
            print("here")
        else:
            q_values = self.model(state)
            q_values = q_values.detach().cpu().numpy()
            return np.argmax(q_values) - 1

    def train(self):
        loss_fn = nn.MSELoss()

        batch_size = self._batch_size
        x_batch, y_batch = [], []
        
        batch = random.sample(self.memory, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states).float().to(device)
        actions = torch.tensor(actions).long().to(device)
        rewards = torch.tensor(rewards).float().to(device)
        next_states = torch.tensor(next_states).float().to(device)
        dones = np.array(dones)
        not_dones = torch.from_numpy(1.0-dones).float().to(device)

        # need to convert actions to actions + 1: (-1, 0, 1) to (0, 1, 2)
        y_preds = self.model(states).gather(1, actions.unsqueeze(1)+1).squeeze()
        
        with torch.no_grad():
            y_targets = rewards + not_dones * self.gamma * self.target(next_states).detach().max(1)[0]

        # loss = torch.mean( (y_preds - y_targets)**2 )
        loss = loss_fn( y_preds, y_targets )
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.copy_target( 0.01 )
        # self.memory = []

    def copy_target(self, polyak=1.0):
        """copy model parameters to target network

        Args:
            polyak (float, optional): [Polyak averaging]. Defaults to 1.0.
        """
        for var1, var2 in zip( self.model.parameters(), self.target.parameters() ):
            var2 = polyak * var1 + (1-polyak) * var2

        