import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime

import gym
import trading_env

import os
from agent_v2.dqn_agent import DQNAgent

def main():

    env_trading = gym.make('test_trading-v2')
    NUM_EP = 400

    date = datetime.datetime( 2017, 7, 15, 0, 0 )
    date_test = datetime.datetime( 2017, 7, 15, 0, 0 )

    agentDQN = DQNAgent(env_trading, gamma=0.99, buffer_size = 1000000,
                        epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.999, 
                        alpha=1e-4, alpha_decay=0.001, batch_size=128, quiet=False)

    rewards = []
    rewards_test = []
    portfolio = []

    print("Populating memory buffer...")

    while (len(agentDQN.memory) < 100000):
        state = env_trading.reset(date = date)
        state = np.reshape(state,200)
        while (True):
            action = agentDQN.act(state, True)
            next_state, reward, done, _ = env_trading.step(action)
            state = np.reshape(state,200)
            next_state = next_state.reshape(200)
            agentDQN.store_step(state, action, reward, next_state, done)
            if done:
                break

    for i in range( NUM_EP ):
        state = env_trading.reset(date = date)
        state = np.reshape(state,200)
        total_reward = 0
        
        while(True):
            action = agentDQN.act(state, step=i)
    #         print(action)
            next_state, reward, done, _ = env_trading.step(action)
            state = np.reshape(state,200)
            next_state = next_state.reshape(200)
            agentDQN.store_step(state, action, reward, next_state, done)
            total_reward += reward
            if done:
                rewards.append(total_reward)
    #             portfolio.append(env_trading.portfolio_value)
                #print("Episode: {}, Total reward: {}".format(i,total_reward))
                break
        if len( agentDQN.memory ) > agentDQN._batch_size:
            agentDQN.train()
        
        state_test = env_trading.reset( date = date_test )
        state_test = np.reshape( state_test, 200 )
        total_reward_test = 0
        
        while( True ):
            action = agentDQN.act(state_test)
    #         print(action)
            state_test, reward_test, done_test, _ = env_trading.step(action)
            state_test = np.reshape(state_test,200)
            total_reward_test += reward_test
            if done_test:
                rewards_test.append(total_reward_test)
                portfolio.append(env_trading.portfolio_value)
                print("Episode: {}, Training reward: {}, Testing reward: {}".format(i, total_reward, total_reward_test))
                break

if __name__ == "__main__":

    main()