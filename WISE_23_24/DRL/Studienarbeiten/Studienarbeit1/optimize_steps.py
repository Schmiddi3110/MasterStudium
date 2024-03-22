import optuna
import numpy as np
from select_action import *
from gridworld import *

class optimize_steps():
    def __init__(self, env):
        self.env = env

    def study_qlearning_decay_epsilon(self, study_name, storage, direction):
        study = optuna.create_study(study_name=study_name, storage=storage,load_if_exists=True, direction=direction)
        study.optimize(lambda trial: self.__qlearning_decay_epsilon(trial), n_trials=1000)

    def study_qlearning_epsilon_greedy(self, study_name, storage, direction):
        study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True,
                                    direction=direction)
        study.optimize(lambda trial: self.__qlearning_epsilon_greedy(trial), n_trials=1000)

    def study_qlearning_ucb(self, study_name, storage, direction):
        study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True,
                                    direction=direction)
        study.optimize(lambda trial: self.__qlearning_ucb(trial), n_trials=1000)

    def study_sarsa_decay_epsilon(self, study_name, storage, direction):
        study = optuna.create_study(study_name=study_name, storage=storage,load_if_exists=True, direction=direction)
        study.optimize(lambda trial: self.__sarsa_decay_epsilon(trial), n_trials=1000)

    def study_sarsa_epsilon_greedy(self, study_name, storage, direction):
        study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True,
                                    direction=direction)
        study.optimize(lambda trial: self.__sarsa_epsilon_greedy(trial), n_trials=1000)

    def study_sarsa_ucb(self, study_name, storage, direction):
        study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True,
                                    direction=direction)
        study.optimize(lambda trial: self.__sarsa_ucb(trial), n_trials=1000)

    def __qlearning_decay_epsilon(self, trial):
        q_table = np.zeros((self.env.num_states(),self.env.num_actions()))
        EPISODES = trial.suggest_int('EPISODES', 50,250)
        INIT_VALUE = trial.suggest_int('INIT_VALUE',0,10)
        MAX_EPISODE_LENGTH = trial.suggest_int('MAX_EPISODE_LENGTH',30,50000)
        GAMMA = trial.suggest_float('GAMMA', 0.5, 1)
        ALPHA = trial.suggest_float('ALPHA', 0.2, 1)
        EPS = trial.suggest_float('EPS',0, 1)
        DECAY = trial.suggest_float('DECAY', 0, 1)
        q_table.fill(INIT_VALUE)

        learning_data = {}
        cum_reward = 0
        # run a certain number of episodes
        for episode in range(EPISODES):
            state = self.env.reset()
            action = decay_epsilon_greedy(EPS, episode, state, q_table, DECAY)

            done = False
            episode_length = 0

            # run episode until a goal state or the maximum number of steps has been reached
            while not done and episode_length < MAX_EPISODE_LENGTH:
                next_state, reward, done = self.env.step(action)
                next_action = decay_epsilon_greedy(EPS, episode, next_state, q_table, DECAY)

                # Q-Learning update rule
                delta = reward + GAMMA * np.max(q_table[next_state, next_action]) * (done < 0.5) - q_table[state, action]
                q_table[state, action] += ALPHA * delta
                cum_reward += ALPHA * delta

                state = next_state
                action = next_action
                episode_length += 1

            learning_data[episode] = [episode_length, cum_reward]

        return cum_reward/(MAX_EPISODE_LENGTH)

    def __qlearning_epsilon_greedy(self, trial):
        q_table = np.zeros((self.env.num_states(),self.env.num_actions()))
        EPISODES = trial.suggest_int('EPISODES', 50,250)
        INIT_VALUE = trial.suggest_int('INIT_VALUE',3,15)
        MAX_EPISODE_LENGTH = trial.suggest_int('MAX_EPISODE_LENGTH',30,50000)
        GAMMA = trial.suggest_float('GAMMA', 0.5, 1)
        ALPHA = trial.suggest_float('ALPHA', 0.2, 1)
        EPS = trial.suggest_float('EPS',0, 1)
        q_table.fill(INIT_VALUE)

        learning_data = {}
        cum_reward = 0
        # run a certain number of episodes
        for episode in range(EPISODES):
            state = self.env.reset()
            action = epsilon_greedy(EPS, state, q_table)

            done = False
            episode_length = 0

            # run episode until a goal state or the maximum number of steps has been reached
            while not done and episode_length < MAX_EPISODE_LENGTH:
                next_state, reward, done = self.env.step(action)
                next_action = epsilon_greedy(EPS, state, q_table)

                # Q-Learning update rule
                delta = reward + GAMMA * np.max(q_table[next_state, next_action]) * (done < 0.5) - q_table[state, action]
                q_table[state, action] += ALPHA * delta
                cum_reward += ALPHA * delta

                state = next_state
                action = next_action
                episode_length += 1

            learning_data[episode] = [episode_length, cum_reward]

        return cum_reward/(MAX_EPISODE_LENGTH)

    def __qlearning_ucb(self, trial):
        q_table = np.zeros((self.env.num_states(),self.env.num_actions()))
        EPISODES = trial.suggest_int('EPISODES', 50,250)
        INIT_VALUE = trial.suggest_int('INIT_VALUE',0,10)
        MAX_EPISODE_LENGTH = trial.suggest_int('MAX_EPISODE_LENGTH',30,50000)
        GAMMA = trial.suggest_float('GAMMA', 0.5, 1)
        ALPHA = trial.suggest_float('ALPHA', 0.2, 1)
        EPS = trial.suggest_float('EPS',0, 1)
        q_table.fill(INIT_VALUE)

        q_table = np.zeros((self.env.num_states(), self.env.num_actions()))
        q_table.fill(INIT_VALUE)

        ucb_table = np.zeros((self.env.num_states(), self.env.num_actions(), 2))  # [cum_reward, count_action]
        ucb_table.fill(1)

        learning_data = {}
        cum_reward = 0
        # run a certain number of episodes
        for episode in range(EPISODES):
            state = self.env.reset()
            episode_length = 0

            action = ucb(episode_length, ucb_table[state], EPS, q_table)

            done = False

            # run episode until a goal state or the maximum number of steps has been reached
            while not done and episode_length < MAX_EPISODE_LENGTH:
                next_state, reward, done = self.env.step(action)
                ucb_table[state, action] = [ucb_table[state, action][0] + reward, ucb_table[state, action][1] + 1]
                next_action = ucb(episode_length, ucb_table[state], EPS, q_table)

                # Q-Learning update rule
                delta = reward + GAMMA * np.max(q_table[next_state, next_action]) * (done < 0.5) - q_table[
                    state, action]
                q_table[state, action] += ALPHA * delta
                cum_reward += ALPHA * delta

                state = next_state
                action = next_action
                episode_length += 1

            learning_data[episode] = [episode_length, cum_reward]

        return cum_reward/(MAX_EPISODE_LENGTH)



    def __sarsa_decay_epsilon(self, trial):
        q_table = np.zeros((self.env.num_states(),self.env.num_actions()))
        EPISODES = trial.suggest_int('EPISODES', 50,250)
        INIT_VALUE = trial.suggest_int('INIT_VALUE',0,10)
        MAX_EPISODE_LENGTH = trial.suggest_int('MAX_EPISODE_LENGTH', 30,50000)
        GAMMA = trial.suggest_float('GAMMA', 0.5, 1)
        ALPHA = trial.suggest_float('ALPHA', 0.2, 1)
        EPS = trial.suggest_float('EPS',0, 1)
        DECAY = trial.suggest_float('DECAY', 0, 1)
        q_table.fill(INIT_VALUE)

        learning_data = {}
        cum_reward = 0
        # run a certain number of episodes
        for episode in range(EPISODES):
            state = self.env.reset()
            action = decay_epsilon_greedy(EPS, episode, state, q_table, DECAY)

            done = False
            episode_length = 0

            # run episode until a goal state or the maximum number of steps has been reached
            while not done and episode_length < MAX_EPISODE_LENGTH:
                next_state, reward, done = self.env.step(action)
                next_action = decay_epsilon_greedy(EPS, episode, next_state, q_table, DECAY)

                # Q-Learning update rule
                delta = reward + GAMMA * q_table[next_state, next_action] * (done < 0.5) - q_table[state, action]
                q_table[state, action] += ALPHA * delta
                cum_reward += ALPHA * delta

                state = next_state
                action = next_action
                episode_length += 1

            learning_data[episode] = [episode_length, cum_reward]

        return cum_reward/(MAX_EPISODE_LENGTH)

    def __sarsa_epsilon_greedy(self, trial):
        q_table = np.zeros((self.env.num_states(),self.env.num_actions()))
        EPISODES = trial.suggest_int('EPISODES', 50,250)
        INIT_VALUE = trial.suggest_int('INIT_VALUE',3,15)
        MAX_EPISODE_LENGTH = trial.suggest_int('MAX_EPISODE_LENGTH',30,50000)
        GAMMA = trial.suggest_float('GAMMA', 0.5, 1)
        ALPHA = trial.suggest_float('ALPHA', 0.2, 1)
        EPS = trial.suggest_float('EPS',0, 1)
        q_table.fill(INIT_VALUE)

        learning_data = {}
        cum_reward = 0
        # run a certain number of episodes
        for episode in range(EPISODES):
            state = self.env.reset()
            action = epsilon_greedy(EPS, state, q_table)

            done = False
            episode_length = 0

            # run episode until a goal state or the maximum number of steps has been reached
            while not done and episode_length < MAX_EPISODE_LENGTH:
                next_state, reward, done = self.env.step(action)
                next_action = epsilon_greedy(EPS, state, q_table)

                # Q-Learning update rule
                delta = reward + GAMMA * q_table[next_state, next_action] * (done < 0.5) - q_table[state, action]
                q_table[state, action] += ALPHA * delta
                cum_reward += ALPHA * delta

                state = next_state
                action = next_action
                episode_length += 1

            learning_data[episode] = [episode_length, cum_reward]

        return cum_reward/(MAX_EPISODE_LENGTH)

    def __sarsa_ucb(self, trial):
        q_table = np.zeros((self.env.num_states(),self.env.num_actions()))
        EPISODES = trial.suggest_int('EPISODES', 50,250)
        INIT_VALUE = trial.suggest_int('INIT_VALUE',0,10)
        MAX_EPISODE_LENGTH = trial.suggest_int('MAX_EPISODE_LENGTH',30,50000)
        GAMMA = trial.suggest_float('GAMMA', 0.5, 1)
        ALPHA = trial.suggest_float('ALPHA', 0.2, 1)
        EPS = trial.suggest_float('EPS',0, 1)
        q_table.fill(INIT_VALUE)

        q_table = np.zeros((self.env.num_states(), self.env.num_actions()))
        q_table.fill(INIT_VALUE)

        ucb_table = np.zeros((self.env.num_states(), self.env.num_actions(), 2))  # [cum_reward, count_action]
        ucb_table.fill(1)

        learning_data = {}
        cum_reward = 0
        # run a certain number of episodes
        for episode in range(EPISODES):
            state = self.env.reset()
            episode_length = 0

            action = ucb(episode_length, ucb_table[state], EPS, q_table)

            done = False

            # run episode until a goal state or the maximum number of steps has been reached
            while not done and episode_length < MAX_EPISODE_LENGTH:
                next_state, reward, done = self.env.step(action)
                ucb_table[state, action] = [ucb_table[state, action][0] + reward, ucb_table[state, action][1] + 1]
                next_action = ucb(episode_length, ucb_table[state], EPS, q_table)

                # Q-Learning update rule
                delta = reward + GAMMA * q_table[next_state, next_action] * (done < 0.5) - q_table[
                    state, action]
                q_table[state, action] += ALPHA * delta
                cum_reward += ALPHA * delta

                state = next_state
                action = next_action
                episode_length += 1

            learning_data[episode] = [episode_length, cum_reward]

        return cum_reward/(MAX_EPISODE_LENGTH)


