"""
Small demo to illustrate how the plot function and the gridworld environment work
"""
import numpy as np

from gridworld import *
from plot import *

GAMMA = 0.7
THETA = 0.001


def calc_v_values(env, state=None):
    v = np.zeros(env.num_states())
    if state is None:
        for state in range(len(v)):
            v[state] = calc_v_values(env, state)[state]

    else:
        for action in range(env.num_actions()):
            next_state, reward, done = env.step_dp(state, action)
            v[state] = reward + GAMMA * v[next_state] * (done < 0.5)
    return v


def calc_q_values(env, v_table, state=None):
    if state is None:
        q = np.zeros((env.num_states(), env.num_actions()))
        for state in range(len(v_table)):
            q[state, :] = calc_q_values(env, v_table, state)

    else:
        q = np.zeros(env.num_actions())
        for action in range(env.num_actions()):
            next_state, reward, done = env.step_dp(state, action)
            q[action] = reward + GAMMA * v_table[next_state] * (done < 0.5)

    return q


def policy_improvement(env, policy_old, v_table):
    policy = np.zeros((env.num_states(), env.num_actions()))

    # greedy policy
    for state in range(env.num_states()):
        q = calc_q_values(env, v_table, state)
        policy[state, np.argmax(q)] = 1

    # check if policy has been changed since last iteration
    policy_stable = np.array_equal(policy, policy_old)

    return policy, policy_stable


def policy_evaluation(env, policy):
    v_table = np.zeros(env.num_states())

    # calculate optimal value function
    while True:
        delta = 0
        for state in range(len(v_table)):
            # calculate q values for all action for the given state
            q = calc_q_values(env, v_table, state)
            pol = policy[state]
            # update delta
            delta = max(delta, np.abs(pol @ q - v_table[state]))
            # update v-value with best Q-value
            v_table[state] = pol @ q

        if delta < THETA:
            break

    return v_table


def policy_iteration(env):
    # use uniform policy as initialization
    policy = np.ones((env.num_states(), env.num_actions())) / env.num_actions()

    # calculate optimal value function
    while True:
        v_table = policy_evaluation(env, policy)
        policy, policy_stable = policy_improvement(env, policy, v_table)

        if policy_stable:
            break

    return v_table, policy


def value_iteration(env):
    v_table = np.zeros(env.num_states())

    # calculate optimal value funciton
    while True:
        delta = 0
        for state in range(len(v_table)):
            # calculate q values for all action for the given state
            q = calc_q_values((env, v_table, state))
            best_q = np.max(q)
            # update delta
            delta = max(delta, np.abs(best_q - v_table[state]))
            # update v-value with best q-vaule
            v_table[state] = best_q

        if delta < THETA:
            break

    # calculate correspnding policy
    policy = np.zeros(env.num_states(), env.num_actions())
    for state in range(env.num_states()):
        q = calc_q_values(env, v_table, state)
        # set probability of best action to 1
        policy[state, np.argmax(q)] = 1

    return v_table, policy


if __name__ == "__main__":
    # create environment
    env = Random(size=12, water=0.3, mountain=0.0)
    # create nonsense V-values and nonsense policy
    # v_table = np.random.random((env.num_states()))
    #v_table = calc_v_values(env)
    # q_table = np.random.random((env.num_states(), env.num_actions()))
    #q_table = calc_q_values(env, v_table)
    # policy = np.random.random((env.num_states(), env.num_actions()))
    v_table, policy = policy_iteration(env)

    # either plot V-values and Q-values without the policy...
    # plot_v_table(env, v_table)
    # plot_q_table(env, q_table)
    # ...or with the policy
    plot_v_table(env, v_table, policy)
    #plot_q_table(env, q_table, policy)
