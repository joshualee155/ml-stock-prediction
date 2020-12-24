import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import datetime

import gym
import trading_env

import os
import agent 
from os import __file__

def main():

    env_trading = gym.make('test_trading-v0')
    NUM_EP = 400
    date = datetime.datetime(2017, 7, 10, 0, 0)
    data = env_trading.historical_data["close"]
    # env_trading.reset(date=date)
    # plt.plot(data[env_trading.start_index:env_trading.start_index + int(env_trading.episode_steps) 
    #             if env_trading.start_index + int(env_trading.episode_steps) < data.shape[0]
    #             else data.shape[0]])

    # plt.show()

    agentDDPG = agent.DDPGAgent(env_trading, 
                                epsilon_log_decay=0.99, 
                                tau = 0.001, 
                                actor_lr = 1e-4, 
                                critic_lr = 1e-4)

    # Ornstein-Uhlenbeck noise by lirnli/OpenAI-gym-solutions    
    def UONoise():
        theta = 0.15
        sigma = 0.2
        state = 0
        while True:
            yield state
            state += -theta*state+sigma*np.random.randn()

    date = datetime.datetime(2017, 7, 15, 0, 0)
    date_test = datetime.datetime(2017, 8, 15, 0, 0)
    noise = UONoise()
    scores = []
    scores_test = []
    sample_actions = [] # Keep track of actions every 100 episode
    portfolios = []

    for e in range(NUM_EP):
        state = np.reshape(env_trading.reset(date=date), 200)
        score = 0

        rewards = []
        while(True):
            action = agentDDPG.actor.act([state], step = e)
            action += next( noise )
            next_state, reward, done, _ = env_trading.step( action )
            next_state = np.reshape(next_state, 200)
            score += reward
            rewards.append( reward )

            agentDDPG.store_step(state, action, reward, next_state, done)

            if done:
                agentDDPG.train()
                scores.append(score)
                # print("Episode: {}, Total reward: {}".format(e, score))
                break
            state = next_state

        # Testing session
        state = np.reshape(env_trading.reset( date = date_test ), 200)
        score_test = 0
        actions = []
        while(True):
            action = agentDDPG.actor.act([state])
            next_state, reward, done, _ = env_trading.step( action )
            actions.append( action )
            next_state = np.reshape(next_state, 200)
            score_test += reward
            if done:
                agentDDPG.actor.update_averages( rewards, [score_test] )
                agentDDPG.actor.record_summary( e )
                scores_test.append(score_test)
                portfolios.append( env_trading.portfolio_value )
                if e % 100 == 0:
                    sample_actions.append( actions )
                print("Episode: {}, Training reward: {}, Testing reward: {}".format(e, score, score_test))
                break
            state = next_state

    plt.figure()
    plt.plot( scores_test, label = 'Testing' )
    plt.plot( scores, label = 'Training' )
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot( np.array( sample_actions ).T )
    plt.show()

    plt.figure()
    plt.plot( np.array( portfolios ).T )
    plt.show()

if __name__ == "__main__":

    main()