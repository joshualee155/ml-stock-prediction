import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime

import gym
import trading_env

import os
from agent_v2.dqn_agent import DQNAgent
from utils import max_reward
import sys

def main():

    env_trading = gym.make('test_trading-v1')
    print(env_trading.action_space)

    NUM_EP = 400

    date = datetime.datetime( 2017, 7, 15, 0, 0 )
    date_test = datetime.datetime( 2017, 8, 15, 0, 0 )

    m = max_reward( env_trading, date )
    m_test = max_reward( env_trading, date_test )

    print("Max Reward for env {} : {:.2f}".format(date.date(), m))
    print("Max Reward for env {} : {:.2f}".format(date_test.date(), m_test))

    agentDQN = DQNAgent(env_trading, gamma=0.99, buffer_size = 100000,
                        epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.999, 
                        alpha=1e-4, alpha_decay=0.001, batch_size=128, quiet=False)

    rewards = []
    rewards_test = []
    portfolio = []
    grads = []

    while (len(agentDQN.memory) < 10000):
        state = env_trading.reset(date = date)
        state = np.reshape(state,200)
        while (True):
            action = env_trading.action_space.sample()
            next_state, reward, done, _ = env_trading.step(action)
            state = np.reshape(state,200)
            next_state = next_state.reshape(200)
            agentDQN.store_step(state, action, reward, next_state, done)
            state = next_state
            print("\rPopulating memory buffer: {:5d}/100000".format(len(agentDQN.memory)), end="")
            sys.stdout.flush()
            if done:
                break

    print("\n")

    for i in range( NUM_EP ):
        state = env_trading.reset(date = date)
        state = np.reshape(state,200)
        total_reward = 0
        actions = []
        
        while(True):
            action = agentDQN.act(state, True)
    #         print(action)
            actions.append(action)
            next_state, reward, done, _ = env_trading.step(action)
            state = np.reshape(state,200)
            next_state = next_state.reshape(200)
            agentDQN.store_step(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                rewards.append(total_reward)
    #             portfolio.append(env_trading.portfolio_value)
                #print("Episode: {}, Total reward: {}".format(i,total_reward))
                break

        grad = agentDQN.train()
        grads.append(grad)
        
        state_test = env_trading.reset( date = date_test )
        state_test = np.reshape( state_test, 200 )
        total_reward_test = 0
        actions_test = []

        while( True ):
            action = agentDQN.act(state_test)
    #         print(action)
            actions_test.append(action)
            state_test, reward_test, done_test, _ = env_trading.step(action)
            state_test = np.reshape(state_test,200)
            total_reward_test += reward_test
            if done_test:
                rewards_test.append(total_reward_test)
                portfolio.append(env_trading.portfolio_value)
                print("Episode: {}, Training reward: {:.4f}, Testing reward: {:.4f}, Grad: {:.4f}, Actions: {:.2f}+/-{:.2f}, Test Actions: {:.2f}+/-{:.2f}".format(i, total_reward, total_reward_test, grad, np.mean(actions), np.std(actions), np.mean(actions_test), np.std(actions_test)))
                break

if __name__ == "__main__":

    main()