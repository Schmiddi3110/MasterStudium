import optuna


def load_study(study_name, storage):
    loaded_study = optuna.load_study(study_name=study_name, storage=storage)
    best_params = loaded_study.best_trial.params
    return best_params


def run_study(study_name, storage, function, direction):
    study = optuna.create_study(study_name=study_name, storage='storage',load_if_exists=True, direction=direction)
    study.optimize(function, n_trials=3000)