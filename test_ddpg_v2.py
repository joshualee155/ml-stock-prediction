import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime

import gym
import trading_env

from agent_v2.ddpg_agent import DDPGAgent
import sys
from utils import max_reward

def main():

    env_trading = gym.make('test_trading-v2')
    NUM_EP = 400

    agentDDPG = DDPGAgent(env_trading, 
                        buffer_size=1000000,
                        tau = 0.01, 
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
    date_test = datetime.datetime(2017, 9, 15, 0, 0)

    m = max_reward( env_trading, date )
    m_test = max_reward( env_trading, date_test )

    print("Max Reward for env {} : {:.2f}".format(date.date(), m))
    print("Max Reward for env {} : {:.2f}".format(date_test.date(), m_test))

    noise = UONoise()
    scores = []
    scores_test = []
    sample_actions = [] # Keep track of actions every 100 episode
    portfolios = []
    actor_grads = []
    critic_grads = []

    while (len(agentDDPG.memory) < 100000):
        state = env_trading.reset(date = date)
        state = np.reshape(state,200)
        while (True):
            # action = agentDDPG.actor.act(state)
            # action = np.clip( action + next(noise), -1, 1 )
            action = env_trading.action_space.sample()[0]
            next_state, reward, done, _ = env_trading.step(action)
            state = state.reshape(200)
            next_state = next_state.reshape(200)
            agentDDPG.store_step(state, action, reward, next_state, done)
            state = next_state
            print("\rPopulating memory buffer: {:5d}/100000".format(len(agentDDPG.memory)), end="")
            sys.stdout.flush()
            if done:
                break

    print("\n")

    for e in range(NUM_EP):
        state = np.reshape(env_trading.reset(date=date), 200)
        score = 0

        rewards = []
        actions = []
        while(True):
            action = agentDDPG.actor.act(state)
            action += next( noise )
            action = np.clip(action, -1, 1)
            actions.append(action)
            next_state, reward, done, _ = env_trading.step( action )
            next_state = np.reshape(next_state, 200)
            score += reward
            rewards.append( reward )

            agentDDPG.store_step(state, action, reward, next_state, done)

            if done:
                actor_grad, critic_grad = agentDDPG.train()
                actor_grads.append(actor_grad)
                critic_grads.append(critic_grad)
                scores.append(score)
                # print("Episode: {}, Total reward: {}".format(e, score))
                break
            state = next_state

        # Testing session
        state = np.reshape(env_trading.reset( date = date_test ), 200)
        score_test = 0
        actions_test = []
        while(True):
            action = agentDDPG.actor.act(state)
            next_state, reward, done, _ = env_trading.step( action )
            actions_test.append( action )
            next_state = np.reshape(next_state, 200)
            score_test += reward
            if done:
                # agentDDPG.actor.update_averages( rewards, [score_test] )
                # agentDDPG.actor.record_summary( e )
                scores_test.append(score_test)
                portfolios.append( env_trading.portfolio_value )
                if e % 100 == 0:
                    sample_actions.append( actions_test )
                print("\rEpisode: {}, Training reward: {:.2f}, Testing reward: {:.2f}, Actor grad: {:.4f}, Critic grad: {:.4f}, Actions: {:.4f}+/-{:.4f}, Test Actions: {:.4f}+/-{:.4f}".format(e, score, score_test, actor_grad, critic_grad, np.mean(actions), np.std(actions), np.mean(actions_test), np.std(actions_test)), end="")
                sys.stdout.flush()
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