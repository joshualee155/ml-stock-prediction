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
    def __init__(self, env, gamma=0.99, buffer_size = 1000000, update_steps = 100,
        epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.999, tau = 0.01, max_grad_norm = 10,
        alpha=1e-4, alpha_decay=0.001, batch_size=128, quiet=False):
        
        self.env = env
        self.memory = deque(maxlen = buffer_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.update_steps = update_steps
        self.max_grad_norm = max_grad_norm
        self._tau = tau
        self._batch_size = batch_size
        self.quiet = quiet
        self._state_dim = np.prod(np.array(env.observation_space.shape))
        
        self.model = Model(self._state_dim).to(device)
        self.target = Model(self._state_dim).to(device)
        # Align model network and target network
        self.copy_target(1.0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha)

    def get_gradient_norm(self, model):

        total_norm = 0.0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        return total_norm

    def store_step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, greedy=False):
        state = torch.from_numpy(state).float().to(device)
        if greedy:
            # epsilon = max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((step + 1) * self.epsilon_decay)))
            epsilon = self.epsilon
            q_values = self.model(state)
            q_values = q_values.detach().cpu().numpy()
            random_action = self.env.action_space.sample() # action space: 0, 1, 2
            q_max_action = np.argmax(q_values)
            action = random_action if (np.random.random() <= epsilon) else q_max_action
            return action
        else:
            q_values = self.model(state)
            q_values = q_values.detach().cpu().numpy()
            return np.argmax(q_values)

    def train(self):
        loss_fn = nn.MSELoss()

        batch_size = self._batch_size
        
        grad = 0.0
        for _ in range(self.update_steps):
            batch = random.sample(self.memory, batch_size)
            
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(states).float().to(device)
            actions = torch.tensor(actions).long().to(device)
            rewards = torch.tensor(rewards).float().to(device)
            next_states = torch.tensor(next_states).float().to(device)
            dones = torch.tensor(dones).float().to(device)

            with torch.no_grad():
                y_targets = rewards + (1 - dones) * self.gamma * self.target(next_states).detach().max(1)[0]

            # loss = torch.mean( (y_preds - y_targets)**2 )
            
            self.optimizer.zero_grad()
            # actions: (0, 1, 2)
            y_preds = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
            loss = loss_fn( y_preds, y_targets )

            # print("Loss before update: {}".format(loss.item()))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            grad += self.get_gradient_norm(self.model)
            self.optimizer.step()

            # re-evaluate loss function
            # y_preds = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
            # loss = loss_fn( y_preds, y_targets )
            # print("Loss after update: {}".format(loss.item()))
                
        self.copy_target( 1.0 )

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return grad / self.update_steps
        
    def copy_target(self, polyak=1.0):
        """copy model parameters to target network

        Args:
            polyak (float, optional): [Polyak averaging]. Defaults to 1.0.
        """
        with torch.no_grad():
            for var1, var2 in zip( self.model.parameters(), self.target.parameters() ):
                var2 = polyak * var1 + (1-polyak) * var2

        