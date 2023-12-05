import matplotlib.pyplot as plt

"""
Small demo to illustrate how the plot function and the gridworld environment work
"""
import numpy as np
import optuna
from gridworld import *
from plot import *
import optimize
from sarsa import *
from qlearning import *

COUNT_ENVIRONMENT = 10


# study = optimize(env)
# study.run_study_qlearning_steps("test", "sqlite:///test.db", "maximize")

def run_A1():
    qlearning_data = {}
    sarsa_data = {}
    envs_A1 = []

    q = qlearning(Q_PARAMS_A1['ALPHA'],
                  Q_PARAMS_A1['GAMMA'],
                  Q_PARAMS_A1['EPISODES'],
                  Q_PARAMS_A1['MAX_EPISODE_LENGTH'],
                  Q_PARAMS_A1['INIT_REWARD'],
                  Q_PARAMS_A1['EPS'],
                  Q_PARAMS_A1['DECAY'])

    s = sarsa(SARSA_PARAMS_A1['ALPHA'],
              SARSA_PARAMS_A1['GAMMA'],
              SARSA_PARAMS_A1['EPISODES'],
              SARSA_PARAMS_A1['MAX_EPISODE_LENGTH'],
              SARSA_PARAMS_A1['INIT_REWARD'],
              SARSA_PARAMS_A1['EPS'],
              SARSA_PARAMS_A1['DECAY'])

    for i in range(COUNT_ENVIRONMENT):
        env = Random(size=12, water=0.3, mountain=0)
        envs_A1.append(env)
        print(f"running environment: {i}")

        qlearning_data[i] = q.run(env)
        sarsa_data[i] = s.run(env)


    return qlearning_data, sarsa_data



def run_A2():
    qlearning_data_A2 = {}
    sarsa_data_A2 = {}
    envs2 = []

    q2 = qlearning(Q_PARAMS_A2['ALPHA'],
                   Q_PARAMS_A2['GAMMA'],
                   Q_PARAMS_A2['EPISODES'],
                   Q_PARAMS_A2['MAX_EPISODE_LENGTH'],
                   Q_PARAMS_A2['INIT_VALUE'],
                   Q_PARAMS_A2['EPS'],
                   Q_PARAMS_A2['DECAY'])

    s2 = sarsa(SARSA_PARAMS_A2['ALPHA'],
               SARSA_PARAMS_A2['GAMMA'],
               SARSA_PARAMS_A2['EPISODES'],
               SARSA_PARAMS_A2['MAX_EPISODE_LENGTH'],
               SARSA_PARAMS_A2['INIT_REWARD'],
               SARSA_PARAMS_A2['EPS'],
               SARSA_PARAMS_A2['DECAY'])

    for i in range(COUNT_ENVIRONMENT):
        env = Random(size=12, water=0, mountain=0.3)
        envs2.append(env)
        print(f"running env: {i}")

        qlearning_data_A2[i] = q2.run(env)
        sarsa_data_A2[i] = s2.run(env)

    return qlearning_data_A2, sarsa_data_A2


if __name__ == "__main__":
    SARSA_PARAMS_A1 = optimize.load_study('DRL_Studienarbeit1_A1_sarsa', 'sqlite:///DRL_Studienarbeit1_A1_sarsa.db')
    SARSA_PARAMS_A2 = optimize.load_study('DRL_Studienarbeit1_A2_sarsa', 'sqlite:///DRL_Studienarbeit1_A2_sarsa.db')

    Q_PARAMS_A1 = optimize.load_study('DRL_Studienarbeit1_A1', 'sqlite:///DRL_Studienarbeit1_A1.db')
    Q_PARAMS_A2 = optimize.load_study('test', 'sqlite:///DRL_Studienarbeit1_A2.db')

    q_data_A1, sarsa_data_A1 = run_A1()

    plot_episodes(q_data_A1, Q_PARAMS_A1['EPISODES'], sarsa_data_A1, SARSA_PARAMS_A1['EPISODES'])

    q_data_A2, sarsa_data_A2 = run_A2()
    plot_steps(q_data_A2, Q_PARAMS_A2['EPISODES'], sarsa_data_A2, SARSA_PARAMS_A2['EPISODES'])
