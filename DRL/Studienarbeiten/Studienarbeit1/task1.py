import numpy as np
import optuna

from gridworld import *
from plot import *

# Hyperparameters, found with optuna
EPS = 0.5715130612955508
ALPHA = 4.608278597795137e-06
GAMMA = 0.8588797410337669
EPS_DECAY = 0.8276369203682284
INIT_REWARD = 2

EPISODES = 5000
MAX_EPISODE_LENGTH = 100

def qlearning(env):
    q_table = np.zeros((env.num_states(), env.num_actions()))
    q_table.fill(INIT_REWARD)
    # run a certain number of episodes
    for episode in range(EPISODES):
        state = env.reset()
        episode_length = 1
        action = select_action(episode_length, state, q_table)

        done = False

        # run episode until a goal state or the maximum number of steps has been reached
        while not done and episode_length < MAX_EPISODE_LENGTH:
            next_state, reward, done = env.step(action)
            next_action = select_action(episode_length, next_state, q_table)

            # Q-Learning update rule
            delta = reward + GAMMA * np.max(q_table[next_state, next_action]) * (done < 0.5) - q_table[state, action]
            q_table[state, action] += ALPHA * delta

            state = next_state
            action = next_action
            episode_length += 1

    return q_table


def select_action(episode_length, state, q_table):
    # do random action
    if np.random.random() < EPS/(EPS_DECAY*episode_length**2):
        return np.random.randint(0, len(q_table[0]))
    # or do best action
    else:
        return np.argmax(q_table[state])
    


if __name__ == "__main__":
    # create environment
    env = Random(size=12, water=0.3, mountain=0.0)
    # create nonsense V-values and nonsense policy
    #sarsa_table = sarsa(env)

    q_learning_table = qlearning(env)


    # either plot V-values and Q-values without the policy...
    # plot_v_table(env, v_table)
    # plot_q_table(env, q_table)
    # ...or with the policy
    #plot_q_table(env, sarsa_table)
    plot_q_table(env, q_learning_table)
