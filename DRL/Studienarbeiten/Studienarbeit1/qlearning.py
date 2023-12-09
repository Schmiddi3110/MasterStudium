import numpy as np
from select_action import *


class qlearning():
    def __init__(self,  alpha, gamma, episodes, max_episode_length, init_reward, eps, decay=None):
        self.alpha = alpha
        self.gamma = gamma
        self.episodes = episodes
        self.max_episode_length = max_episode_length
        self.init_reward = init_reward
        self.eps = eps
        self.decay = decay

    def run_decay_epsilon_greedy(self, env):
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
        q_table.fill(self.init_reward)
        learning_data = {}
        cum_reward = 0
        # run a certain number of episodes
        for episode in range(self.episodes):
            state = env.reset()

            action = decay_epsilon_greedy(self.eps, episode, state, q_table, self.decay)

            done = False
            episode_length = 0

            # run episode until a goal state or the maximum number of steps has been reached
            while not done and episode_length < self.max_episode_length:
                next_state, reward, done = env.step(action)
                next_action = decay_epsilon_greedy(self.eps, episode, next_state, q_table, self.decay)

                # Q-Learning update rule
                delta = reward + self.gamma * np.max(q_table[next_state, next_action]) * (done < 0.5) - q_table[
                    state, action]
                q_table[state, action] += self.alpha * delta
                cum_reward += self.alpha * delta

                state = next_state
                action = next_action
                episode_length += 1

            learning_data[episode] = [episode_length, cum_reward]

        return learning_data


    def run_epsilon_greedy(self, env):
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
        q_table.fill(self.init_reward)



        learning_data = {}
        cum_reward = 0
        # run a certain number of episodes
        for episode in range(self.episodes):

            state = env.reset()
            action = epsilon_greedy(self.eps, state, q_table)

            done = False
            episode_length = 0

            # run episode until a goal state or the maximum number of steps has been reached
            while not done and episode_length < self.max_episode_length:
                next_state, reward, done = env.step(action)

                next_action = epsilon_greedy(self.eps, state, q_table)

                # Q-Learning update rule
                delta = reward + self.gamma * np.max(q_table[next_state, next_action]) * (done < 0.5) - q_table[state, action]
                q_table[state, action] += self.alpha * delta
                cum_reward += self.alpha * delta

                state = next_state
                action = next_action
                episode_length += 1

            learning_data[episode] = [episode_length, cum_reward]

        return learning_data


    def run_ucb(self, env):
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
        q_table.fill(self.init_reward)

        ucb_table = np.zeros((env.num_states(), env.num_actions(), 2))  # [cum_reward, count_action]
        ucb_table.fill(1)

        learning_data = {}
        cum_reward = 0
        # run a certain number of episodes
        for episode in range(self.episodes):
            state = env.reset()
            episode_length = 0

            action = ucb(episode_length, ucb_table[state], self.eps, q_table)

            done = False

            # run episode until a goal state or the maximum number of steps has been reached
            while not done and episode_length < self.max_episode_length:
                next_state, reward, done = env.step(action)
                ucb_table[state, action] = [ucb_table[state, action][0] + reward, ucb_table[state, action][1] + 1]
                next_action = ucb(episode_length, ucb_table[state], self.eps, q_table)

                # Q-Learning update rule
                delta = reward + self.gamma * np.max(q_table[next_state, next_action]) * (done < 0.5) - q_table[state, action]
                q_table[state, action] += self.alpha * delta
                cum_reward += self.alpha * delta

                state = next_state
                action = next_action
                episode_length += 1

            learning_data[episode] = [episode_length, cum_reward]

        return learning_data
