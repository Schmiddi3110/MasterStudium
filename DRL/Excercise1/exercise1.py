"""
Small demo to illustrate how the plot function and the gridworld environment work
"""

from gridworld import *
from plot import *

if __name__ == "__main__":
    # create environment
    env = ExerciseWorld()
    # create nonsense V-values and nonsense policy
    v_table = np.random.random((env.num_states()))
    q_table = np.random.random((env.num_states(), env.num_actions()))
    policy = np.random.random((env.num_states(), env.num_actions()))

    # either plot V-values and Q-values without the policy...
    # plot_v_table(env, v_table)
    # plot_q_table(env, q_table)
    # ...or with the policy
    plot_v_table(env, v_table, policy)
    plot_q_table(env, q_table, policy)
