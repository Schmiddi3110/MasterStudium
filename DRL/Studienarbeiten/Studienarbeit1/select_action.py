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
    if np.random.random() < (eps / (decay * (episode + 1))):
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

def ucb(t, ucb_table, eps, q_table):

    if np.random.random() < eps:
        return np.random.randint(0, len(q_table[0]))

    else:
        ucb_values = []
        for action in ucb_table:
            value = action[0]/t + np.sqrt(2 * np.log10(t)/action[1])
            ucb_values.append(value)

        return np.argmax(ucb_values)





