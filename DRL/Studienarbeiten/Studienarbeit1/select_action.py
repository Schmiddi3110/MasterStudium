import numpy as np


def decay_epsilon_greedy(eps, episode, state, q_table, decay):
    """
    This function determines which action to take in a given state. Decaying-Epsilon-Greedy is used to return a random action or the best possible action.
    Parameters:
        eps: Parameter for Decaying-Epsilon-Greedy Algorithm
        decay: Parameter for Decaying-Epsilon-Greedy Algorithm
        episode: current episode. Used for Decaying-Epsilon-Greedy Algorithm
        state: Current state of the Episode
        q_table: Current Q-table

    Returns:
        Action to take in a given state
    """
    # do random action
    if np.random.random() < eps / (decay * (episode + 1)):
        return np.random.randint(0, len(q_table[0]))
    # or do best action
    else:
        return np.argmax(q_table[state])


def epsilon_greedy(eps, state, q_table):
    """
    This function determines which action to take in a given state. Decaying-Epsilon-Greedy is used to return a random action or the best possible action.
    Parameters:
        eps: Parameter for Decaying-Epsilon-Greedy Algorithm
        decay: Parameter for Decaying-Epsilon-Greedy Algorithm
        episode: current episode. Used for Decaying-Epsilon-Greedy Algorithm
        state: Current state of the Episode
        q_table: Current Q-table

    Returns:
        Action to take in a given state
    """
    # do random action
    if np.random.random() < eps:
        return np.random.randint(0, len(q_table[0]))
    # or do best action
    else:
        return np.argmax(q_table[state])





