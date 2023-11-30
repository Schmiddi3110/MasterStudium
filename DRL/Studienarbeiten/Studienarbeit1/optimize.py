import optuna
import numpy as np
import algorithms
from gridworld import *
from numba import jit, cuda

def load_study(study_name, storage):
    loaded_study = optuna.load_study(study_name=study_name, storage=storage)
    best_params = loaded_study.best_trial.params
    return best_params


class optimize():
    def __init__(self, env):
        self.env = env

    def run_study(self, study_name, storage, direction):
        study = optuna.create_study(study_name=study_name, storage=storage,load_if_exists=True, direction=direction)
        study.optimize(lambda trial: self.qlearning_steps(trial), n_trials=3000)

    def qlearning_steps(self, trial):
        q_table = np.zeros((self.env.num_states(),self.env.num_actions()))
        EPISODES = trial.suggest_int('EPISODES', 11620,11621)
        INIT_VALUE = trial.suggest_int('INIT_VALUE',5,6)
        MAX_EPISODE_LENGTH = trial.suggest_int('MAX_EPISODE_LENGTH',1544,1545)
        GAMMA = trial.suggest_float('GAMMA', 0.9955785729498876,0.9955785729498877)
        ALPHA = trial.suggest_float('ALPHA', 0.6660010689218203,0.6660010689218204)
        EPS = trial.suggest_float('EPS',0.5579320801207901,0.5579320801207902)
        DECAY = trial.suggest_float('DECAY', 0.1763870404498077, 0.1763870404498078)
        q_table.fill(INIT_VALUE)

        learning_data = {}
        cum_reward = 0
        # run a certain number of episodes
        for episode in range(EPISODES):
            state = self.env.reset()
            action = algorithms.select_action(EPS, DECAY, episode, state, q_table)

            done = False
            episode_length = 0

            # run episode until a goal state or the maximum number of steps has been reached
            while not done and episode_length < MAX_EPISODE_LENGTH:
                next_state, reward, done = self.env.step(action)
                next_action = algorithms.select_action(EPS, DECAY, episode, next_state, q_table)

                # Q-Learning update rule
                delta = reward + GAMMA * np.max(q_table[next_state, next_action]) * (done < 0.5) - q_table[state, action]
                q_table[state, action] += ALPHA * delta
                cum_reward += ALPHA * delta

                state = next_state
                action = next_action
                episode_length += 1

            learning_data[episode] = [episode_length, cum_reward]

        return cum_reward/episode_length