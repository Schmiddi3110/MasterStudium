"""
Small demo to illustrate how the plot function and the gridworld environment work
"""
import numpy as np
import optuna

from gridworld import *
from plot import *

# SARSA parameters
EPS = 0.1
ALPHA = 0.1
GAMMA = 0.9
EPISODES = 100000
MAX_EPISODE_LENGTH = 200


def sarsa(env):
    q_table = np.zeros((env.num_states(), env.num_actions()))

    # run a certain number of episodes
    for episode in range(EPISODES):
        state = env.reset()
        action = select_action(state, q_table)

        done = False
        episode_length = 0

        # run episode until a goal state or the maximum number of steps has been reached
        while not done and episode_length < MAX_EPISODE_LENGTH:
            next_state, reward, done = env.step(action)
            next_action = select_action(next_state, q_table)

            # SARSA update rule
            delta = reward + GAMMA * q_table[next_state, next_action] * (done < 0.5) - q_table[state, action]
            q_table[state, action] += ALPHA * delta

            state = next_state
            action = next_action
            episode_length += 1

    return q_table


def qlearning(env):
    q_table = np.zeros((env.num_states(), env.num_actions()))

    # run a certain number of episodes
    for episode in range(EPISODES):
        state = env.reset()
        action = select_action(state, q_table)

        done = False
        episode_length = 0

        # run episode until a goal state or the maximum number of steps has been reached
        while not done and episode_length < MAX_EPISODE_LENGTH:
            next_state, reward, done = env.step(action)
            next_action = select_action(next_state, q_table)

            # Q-Learning update rule
            delta = reward + GAMMA * np.max(q_table[next_state, next_action]) * (done < 0.5) - q_table[state, action]
            q_table[state, action] += ALPHA * delta

            state = next_state
            action = next_action
            episode_length += 1

    return q_table


def select_action(state, q_table):
    # do random action
    if np.random.random() < EPS:
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
