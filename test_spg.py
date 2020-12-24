import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime

import gym
import trading_env

import os
from agent_v2.spg_agent import StochasticPolicyGradientAgent

def main():

    env_trading = gym.make('test_trading-v2')
    NUM_EP = 400
    date = datetime.datetime(2017, 7, 10, 0, 0)
    data = env_trading.historical_data["close"]
    env_trading.reset(date=date)
    # plt.plot(data[env_trading.start_index:env_trading.start_index + int(env_trading.episode_steps) 
    #             if env_trading.start_index + int(env_trading.episode_steps) < data.shape[0]
    #             else data.shape[0]])

    # plt.show()

    agentSPG = StochasticPolicyGradientAgent(env_trading, learning_rate = 1e-4, 
                                               discount_rate = 0.99, batch_size = 64)

    rewards = []
    rewards_test = []
    portfolio = []
    for i in range( NUM_EP ):
        state = env_trading.reset(date = datetime.datetime( 2017, 7, 15, 0, 0 ))
        state = np.reshape(state,200)
        total_reward = 0
        
        while(True):
            action = agentSPG.act(state)
    #         print(action)
            state, reward, done, _ = env_trading.step(action)
            state = np.reshape(state,200)
            agentSPG.store_step(state, reward)
            total_reward += reward
            if done:
                rewards.append(total_reward)
    #             portfolio.append(env_trading.portfolio_value)
                #print("Episode: {}, Total reward: {}".format(i,total_reward))
                break
        agentSPG.train()
        
        state_test = env_trading.reset( date = datetime.datetime(2017, 8, 15, 0, 0) )
        state_test = np.reshape( state_test, 200 )
        total_reward_test = 0
        
        while( True ):
            action = agentSPG.act(state_test)
    #         print(action)
            state_test, reward_test, done_test, _ = env_trading.step(action)
            state_test = np.reshape(state_test,200)
    #         agentSPG.store_step(action, state, reward)
            total_reward_test += reward_test
            if done_test:
                rewards_test.append(total_reward_test)
                portfolio.append(env_trading.portfolio_value)
                print("Episode: {}, Training reward: {}, Testing reward: {}".format(i, total_reward, total_reward_test))
                break

if __name__ == "__main__":

    main()