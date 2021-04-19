import numpy as np

def max_reward(env_trading, date):
    """
    Get maximum reward by forward looking at returns next step
    """
    state = env_trading.reset(date)
    total_reward = 0.0
    while (True):
        action = env_trading.best_action()
        # action = env_trading.action_space.sample()
        
        next_state, reward, done, _ = env_trading.step(action)
        state = np.reshape(state,200)
        next_state = next_state.reshape(200)
        state = next_state
        total_reward += reward
        if done:
            break
    return total_reward