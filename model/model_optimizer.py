import optuna
from torch import nn
from optuna.samplers import TPESampler

from .model import Model
from .model_architecure import MLP
from .utils import create_study_path, global_seed
from .model_config import ModelConfig
from .model_execution_strategy import ModelExecutionStrategy


def activation_functions(activation_name):
    functions = {
        "ReLU": nn.ReLU(),
        "Tanh": nn.Tanh(),
        "Sigmoid": nn.Sigmoid(),
        "Leaky": nn.LeakyReLU(),
    }
    return functions[activation_name]


def print_best_trial(trial: optuna.trial.FrozenTrial) -> None:
    print("Best trial:")
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


class ModelOptimization(ModelExecutionStrategy):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def execute(self):
        study = self.create_optuna_study()
        print_best_trial(study.best_trial)
        self.save_study_as_csv(study)

    def create_optuna_study(self) -> optuna.Study:
        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
        n_trials = int(self.config.N_TRIALS)
        study.optimize(self.trial, n_trials)
        return study

    def save_study_as_csv(self, study: optuna.Study) -> None:
        active_vars = self.config.ACTIVE_RESPONSE_VARS
        study_dir_path = self.config.STUDY_DIR
        study_path = create_study_path(active_vars, study_dir_path)
        df = study.trials_dataframe()
        df.to_csv(study_path, index=False)

    def trial(self, trial: optuna.trial.Trial) -> float:
        global_seed()
        max_layers = self.config.NUM_LAYERS
        fl_range = (1, 24)
        train_size_range = (0.1, 0.8)
        weight_decay_range = (1e-5, 1e-1)
        learning_rate_range = (1e-5, 1e-1)
        batch_size_options = (10, 100)
        num_epochs_options = (50, 500)
        activation_functions_options: list[str] = ["ReLU", "Tanh", "Sigmoid", "Leaky"]

        params = {
            "n_layers": trial.suggest_int("n_layers", 1, max_layers),
            "batch_size": trial.suggest_int("batch_size", *batch_size_options),
            "num_epochs": trial.suggest_int("num_epochs", *num_epochs_options),
            "train_size": trial.suggest_float("train_size", *train_size_range, log=True),
            "weight_decay": trial.suggest_float("weight_decay", *weight_decay_range, log=True),
            "learning_rate": trial.suggest_float("learning_rate", *learning_rate_range, log=True),
        }

        layers = [9]
        activations = []

        for i in range(params["n_layers"]):
            layer = trial.suggest_int(f"l_{i + 1}", *fl_range)
            layers.append(layer)
            activation = trial.suggest_categorical(f"a_{i + 1}", activation_functions_options)
            activations.append(activation_functions(activation))

        layers.append(len(self.config.ACTIVE_RESPONSE_VARS))

        # Instance base model and pass everything
        core = MLP(layers, activations)

        model = Model(self.config, core, params)

        mse, mre, r2 = model.run()
        trial.set_user_attr("mse", mse)
        trial.set_user_attr("mre", mre)
        trial.set_user_attr("r2", r2)

        print("Params: ", params)
        print("Layers: ", layers)
        print("Activations: ", activations)
        print(f"MSE: {mse}")
        print(f"MRE: {mre}")
        print(f"R2: {r2}")

        return mse
