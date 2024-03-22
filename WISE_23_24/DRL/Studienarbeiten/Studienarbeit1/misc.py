from tabulate import tabulate
import pandas as pd
import optuna

def eval(data):
    """
    This function returns a dictionary of shape {environment: [episode_goal_first_found, count_goal_found, average_steps_in_episode]}
    Parameters:
        data (dictionary): Containing Q-Learning data or SARSA-data
    """
    goal_dict = {}
    for i in data:
        first_goal = 0
        count_goal = 0
        steps = 0
        for keys in data[i]:
            steps += data[i][keys][0]
            if data[i][keys][2] and first_goal == 0:
                first_goal = keys

            if data[i][keys][2]:
                count_goal += 1

        goal_dict[i] = [first_goal, count_goal, steps / len(data[i].keys())]

    goal_dict = average_dict_elementwise(goal_dict)
    return goal_dict


def average_dict_elementwise(input_dict):
    """
    This function retrun the average of the provided dictionary elementwise
    """
    if not input_dict:
        return {}

    list_length = len(next(iter(input_dict.values())))

    average_dict = {}

    for i in range(list_length):
        average_dict[i] = sum(input_dict[key][i] for key in input_dict) / len(input_dict)

    return average_dict

def print_stats(stats, caption):
    print(caption)
    print(f"Found the target for the first on average in Episode {stats[0]}")
    print(f"Found the target on average {stats[1]} times")
    print(f"Average Episode length: {stats[2]}")



def print_parameters(Q_Params, SARSA_Params):
    df = pd.DataFrame(
        {"Q-Learning Parameters": Q_Params, "SARSA Parameters": SARSA_Params})
    print(tabulate(df, headers='keys', tablefmt='psql'))


def load_best_paramerters(study_name, storage):
    """
    This function retrun a dictionary of parameters of trial in the given study
    Parameters:
        study_name: Name of the study to load
        storage: Location of the study

    Return:
         Dictionary of best parameters
    """
    loaded_study = optuna.load_study(study_name=study_name, storage=storage)
    best_params = loaded_study.best_trial.params
    return best_params