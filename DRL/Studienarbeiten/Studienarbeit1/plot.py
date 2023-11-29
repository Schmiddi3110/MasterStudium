from config import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pathEffects


def plot_v_table(env, v_table, policy=None):
    """
    Plot V-function for a given gridworld environment

    :param env: GridworldEnv environment
    :param v_table: 1D Numpy array with shape (number of states). State order must be consistent with
    observations given by the GridworldEnv._state_to_obs function
    :param policy: (optional) 2D Numpy array with shape (number of states, number of actions). State order must be consistent with
    observations given by the GridworldEnv._state_to_obs function. Action order must be consistent with
    G_UP / G_RIGHT / G_DOWN / G_LEFT as described in the config.py file
    """
    plot_table(env=env, table=v_table, policy=policy)


def plot_q_table(env, q_table, policy=None):
    """
    Plot Q-function for a given gridworld environment

    :param env: GridWorld environment
    :param v_table: 2D Numpy array with shape (number of states, number of actions). State order must be consistent with
    observations given by the GridworldEnv._state_to_obs function. Action order must be consistent with
    G_UP / G_RIGHT / G_DOWN / G_LEFT as described in the config.py file
    :param policy: (optional) 2D Numpy array with shape (number of states, number of actions). State order must be consistent with
    observations given by the GridworldEnv._state_to_obs function. Action order must be consistent with
    G_UP / G_RIGHT / G_DOWN / G_LEFT as described in the config.py file
    """
    plot_table(env=env, table=q_table, policy=policy)


def plot_table(env, table, policy):
    m, n = env.shape()

    fig, ax = plt.subplots(figsize=(n, m))
    normalized_table = normalize_table(env, table)

    for x in range(m):
        for y in range(n):
            if env.grid[x][y].upper() in ['#', 'O', 'G']:
                draw_state_type(x=y, y=-x, type=env.grid[x][y])
            else:
                idx = env._state_to_obs((x, y))

                if table.ndim == 2:
                    # plot Q values
                    for action in range(env.num_actions()):
                        draw_q_polygon(x=y, y=-x, action=action, values=normalized_table[idx])
                        draw_q_value(x=y, y=-x, action=action, values=table[idx])
                else:
                    # plot V values
                    draw_v_polygon(x=y, y=-x, value=normalized_table[idx])
                    draw_v_value(x=y, y=-x, value=table[idx])

                # plot resulting policy
                if policy is not None:
                    for action in range(env.num_actions()):
                        draw_policy(x=y, y=-x, action=action, values=policy[idx])

                if env.grid[x][y].upper() in ['A', 'S']:
                    draw_state_type(x=y, y=-x, type=env.grid[x][y])

    ax.axis('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def normalize_table(env, table):
    # normalize V-/Q-values to [0,1] range for plotting
    # consider also cases where the minimum/maximum values are
    # smaller/larger than the ones of the goal / water states
    if np.min(table) < min(env.o_reward, env.g_reward) * 1.01:
        a = np.min(table)
    else:
        a = np.min(table[table > min(env.o_reward, env.g_reward) * 0.99])

    if np.max(table) > max(env.o_reward, env.g_reward) * 1.01:
        A = np.max(table)
    else:
        A = np.max(table[table < max(env.o_reward, env.g_reward) * 0.99])

    return (table - a) / (A - a + 1e-9)


def map_value_to_color(value):
    # maps a value [0,1] to a suitable color (0-red, 0.5-yellow, 1-green)

    if value <= 0:
        return np.array([1, 0, 0])
    elif value <= 0.5:
        return np.array([1, 0, 0]) + 2 * value * np.array([0, 1, 0])
    elif value <= 1:
        return np.array([1, 1, 0]) + 2 * (value - 0.5) * np.array([-1, 0, 0])
    else:
        return np.array([0, 1, 0])


def draw_policy(x, y, action, values):
    if values[action] < 1e-3:
        return

    LENGTH = 0.25
    if action == G_UP:
        dx = 0
        dy = LENGTH
    elif action == G_RIGHT:  # right
        dx = LENGTH
        dy = 0
    elif action == G_DOWN:  # down
        dx = 0
        dy = -LENGTH
    elif action == G_LEFT:  # left
        dx = -LENGTH
        dy = 0

    value = values[action]
    plt.arrow(x, y, dx, dy, head_length=LENGTH, width=value / 20)


def draw_q_polygon(x, y, action, values):
    if action == G_UP:
        xs = [x - 0.5, x + 0.5, x]
        ys = [y + 0.5, y + 0.5, y]
    elif action == G_RIGHT:  # right
        xs = [x + 0.5, x + 0.5, x]
        ys = [y - 0.5, y + 0.5, y]
    elif action == G_DOWN:  # down
        xs = [x - 0.5, x + 0.5, x]
        ys = [y - 0.5, y - 0.5, y]
    elif action == G_LEFT:  # left
        xs = [x - 0.5, x - 0.5, x]
        ys = [y - 0.5, y + 0.5, y]

    color = map_value_to_color(values[action])
    plt.fill(xs, ys, facecolor=color)


def draw_q_value(x, y, action, values):
    OFFSET = 0.25
    dx, dy = 0, 0
    if action == G_UP:  # up
        dy = OFFSET
    elif action == G_RIGHT:  # right
        dx = OFFSET
    elif action == G_DOWN:  # down
        dy = -OFFSET
    elif action == G_LEFT:  # left
        dx = -OFFSET

    value = values[action]

    color = "w"
    if value == np.max(values):
        color = "deepskyblue"
    text = plt.text(x + dx, y + dy, value.round(decimals=2), ha="center", va="center", color="k")
    text.set_path_effects([pathEffects.Stroke(linewidth=8, foreground=color), pathEffects.Normal()])


def draw_v_polygon(x, y, value):
    color = map_value_to_color(value)
    rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=color)
    plt.gca().add_patch(rect)


def draw_v_value(x, y, value):
    text = plt.text(x, y, value.round(decimals=2), ha="center", va="center", color="k")
    text.set_path_effects([pathEffects.Stroke(linewidth=8, foreground="w"), pathEffects.Normal()])


def draw_state_type(x, y, type):
    if type == 'S' or type == 'A':
        facecolor = 'none'
        offset = -0.15
    else:
        facecolor = 'w'
        offset = 0

    rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=facecolor)
    plt.gca().add_patch(rect)
    plt.text(x + offset, y + offset, type, ha="center", va="center", color='k', fontsize=24, fontweight='bold')


def plot_A1(qlearning_data, Q_EPISODES, sarsa_data, SARSA_EPISODES):
    qlearning_x_avg = np.zeros(Q_EPISODES)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    plt.subplots_adjust(wspace=20, right=0.7)
    for i in range(qlearning_data.keys().__len__()):
        qlearning_x = [value[1] for value in list(qlearning_data[i].values())]

        qlearning_x_avg = [sum(x) for x in zip(qlearning_x, qlearning_x_avg)]

        ax1.plot(range(Q_EPISODES), qlearning_x, label='Env ' + str(i), alpha=0.3)

    qlearning_x_avg = [a / 10 for a in qlearning_x_avg]
    ax1.plot(range(Q_EPISODES), qlearning_x_avg, '--', label="Average")

    ax1.set_xlabel('Episode', fontdict={'size': 15})
    ax1.set_ylabel('Cumulative reward', fontdict={'size': 15})
    ax1.set_title('Q-Learning', fontdict={'size': 15})
    ax1.set_ylim(
        min(val for inner_dict in qlearning_data.values() for inner_list in inner_dict.values() for val in inner_list),
        max(val for inner_dict in qlearning_data.values() for inner_list in inner_dict.values() for val in inner_list))

    ax1.set_xlim(0, 200)

    # SARSA
    sarsa_x_avg = np.zeros(SARSA_EPISODES)
    for i in range(sarsa_data.keys().__len__()):
        sarsa_x = [value[1] for value in list(sarsa_data[i].values())]

        sarsa_x_avg = [sum(x) for x in zip(sarsa_x, sarsa_x_avg)]

        ax2.plot(range(SARSA_EPISODES), sarsa_x, label='Env ' + str(i), alpha=0.3)

    sarsa_x_avg = [a / 10 for a in sarsa_x_avg]
    ax2.plot(range(SARSA_EPISODES), sarsa_x_avg, '--', label="Average")

    ax2.set_xlabel('Episode', fontdict={'size': 15})
    ax2.set_title('SARSA', fontdict={'size': 15})
    ax2.set_ylim(
        min(val for inner_dict in sarsa_data.values() for inner_list in inner_dict.values() for val in inner_list),
        max(val for inner_dict in sarsa_data.values() for inner_list in inner_dict.values() for val in inner_list))
    ax2.set_xlim(0, 169)
    fig.suptitle("Cumulative reward over episodes", fontsize=25)

    lines_labels = ax1.get_legend_handles_labels()
    lines, labels = [sum(lol, []) for lol in zip(lines_labels)]
    fig.legend(lines, labels, bbox_to_anchor=(1, 0.84), loc='upper left', ncol=1)

    fig.tight_layout()
    plt.show()


def plot_A2(qlearning_data, Q_EPISODES, sarsa_data, SARSA_EPISODES):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    plt.subplots_adjust(wspace=20, right=0.7)
    qlearning_x_avg = np.zeros((Q_EPISODES))
    qlearning_y_avg = np.zeros((Q_EPISODES))
    for i in range(qlearning_data.keys().__len__()):
        qlearning_x = np.cumsum([value[0] for value in list(qlearning_data[i].values())])
        qlearning_y = np.cumsum([value[1] for value in list(qlearning_data[i].values())])

        qlearning_x_avg = [sum(x) for x in zip(qlearning_x, qlearning_x_avg)]

        qlearning_y_avg = [sum(y) for y in zip(qlearning_y, qlearning_y_avg)]

        ax1.plot(qlearning_x, qlearning_y, label=i, alpha=0.2)

    qlearning_x_avg = [a / 10 for a in qlearning_x_avg]
    qlearning_y_avg = [a / 10 for a in qlearning_y_avg]

    ax1.plot(qlearning_x_avg, qlearning_y_avg, '--', label="avg")

    # Adding labels and title
    ax1.set_xlabel('number of steps', fontdict={'size': 15})
    ax1.set_ylabel('Cumulative reward', fontdict={'size': 15})
    ax1.set_title('Q-Learning', fontdict={'size': 15})
    ax1.set_ylim(
        min(val for inner_dict in qlearning_data.values() for inner_list in inner_dict.values() for val in inner_list),
        max(val for inner_dict in qlearning_data.values() for inner_list in inner_dict.values() for val in inner_list))
    ax1.set_xlim(0, 9000)

    # SARSA
    sarsa_x_avg = np.zeros((SARSA_EPISODES))
    sarsa_y_avg = np.zeros((SARSA_EPISODES))
    for i in range(sarsa_data.keys().__len__()):
        sarsa_x = np.cumsum([value[0] for value in list(sarsa_data[i].values())])
        sarsa_y = np.cumsum([value[1] for value in list(sarsa_data[i].values())])

        sarsa_x_avg = [sum(x) for x in zip(sarsa_x, sarsa_x_avg)]

        sarsa_y_avg = [sum(y) for y in zip(sarsa_y, sarsa_y_avg)]

        plt.plot(sarsa_x, sarsa_y, label=i, alpha=0.2)

    sarsa_x_avg = [a / 10 for a in sarsa_x_avg]
    sarsa_y_avg = [a / 10 for a in sarsa_y_avg]

    plt.plot(sarsa_x_avg, sarsa_y_avg, '--', label="avg")

    # Adding labels and title
    ax2.set_xlabel('number of steps', fontdict={'size': 15})
    ax2.set_title('SARSA', fontdict={'size': 15})
    ax2.set_ylim(
        min(val for inner_dict in sarsa_data.values() for inner_list in inner_dict.values() for val in inner_list),
        max(val for inner_dict in sarsa_data.values() for inner_list in inner_dict.values() for val in inner_list))
    ax2.set_xlim(0, 9000)

    fig.suptitle("total number of steps over Cumulative reward", fontsize=25)

    lines_labels = ax1.get_legend_handles_labels()
    lines, labels = [sum(lol, []) for lol in zip(lines_labels)]
    fig.legend(lines, labels, bbox_to_anchor=(1, 0.84), loc='upper left', ncol=1)

    fig.tight_layout()
    plt.show()