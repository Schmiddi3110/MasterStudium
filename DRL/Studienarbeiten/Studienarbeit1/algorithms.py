import numpy as np


def select_action(eps, decay, episode, state, q_table):
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


def qlearning(env, alpha, gamma, episodes, max_episode_length, init_reward, eps, decay):
    """
    This function performs the Q-Learning algorithm on a given GridWorld environment
    Parameters:
        env: Environment to perform Q-Learning on.
        alpha: Learning rate
        gamma: Discount factor
        episodes: Number of episodes to perform
        max_episode_length: Maximum length of an episode
        init_reward: Initial reward for every possible action
        eps: Chance to perform a random action
        decay: Parameter for decaying epsilon-greedy

    Return:
         Dictionary of shape: environment_N: { episode_M: { [steps_taken_in_episode, reward_from_episode] } }
    """

    q_table = np.zeros((env.num_states(), env.num_actions()))
    q_table.fill(init_reward)
    learning_data = {}
    cum_reward = 0
    # run a certain number of episodes
    for episode in range(episodes):
        state = env.reset()
        action = select_action(eps, decay, episode, state, q_table)

        done = False
        episode_length = 0

        # run episode until a goal state or the maximum number of steps has been reached
        while not done and episode_length < max_episode_length:
            next_state, reward, done = env.step(action)
            next_action = select_action(eps, decay, episode, next_state, q_table)

            # Q-Learning update rule
            delta = reward + gamma * np.max(q_table[next_state, next_action]) * (done < 0.5) - q_table[state, action]
            q_table[state, action] += alpha * delta
            cum_reward += alpha * delta

            state = next_state
            action = next_action
            episode_length += 1

        learning_data[episode] = [episode_length, cum_reward]

    return learning_data


def sarsa(env, alpha, gamma, episodes, max_episode_length, init_reward, eps, decay):
    """
    This function performs the SARSA algorithm on a given GridWorld environment
    Parameters:
        env: Environment to perform Q-Learning on.
        alpha: Learning rate
        gamma: Discount factor
        episodes: Number of episodes to perform
        max_episode_length: Maximum length of an episode
        init_reward: Initial reward for every possible action
        eps: Chance to perform a random action
        decay: Parameter for decaying epsilon-greedy

    Return:
         Dictionary of shape: environment_N: { episode_M: { [steps_taken_in_episode, reward_from_episode] } }
    """
    q_table = np.zeros((env.num_states(), env.num_actions()))
    q_table.fill(init_reward)
    learning_data = {}
    cum_reward = 0
    # run a certain number of episodes
    for episode in range(episodes):
        state = env.reset()
        action = select_action(eps, decay, episode, state, q_table)

        done = False
        episode_length = 0

        # run episode until a goal state or the maximum number of steps has been reached
        while not done and episode_length < max_episode_length:
            next_state, reward, done = env.step(action)
            next_action = select_action(eps, decay, episode, next_state, q_table)

            # Q-Learning update rule
            delta = reward + gamma * q_table[next_state, next_action] * (done < 0.5) - q_table[state, action]
            q_table[state, action] += alpha * delta
            cum_reward += alpha * delta

            state = next_state
            action = next_action
            episode_length += 1

        learning_data[episode] = [episode_length, cum_reward]

    return learning_data
