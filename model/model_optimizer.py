import optuna
import numpy as np
from torch import nn
from optuna.samplers import TPESampler

from .model import NeuralNetworkModel
from .model_architecture import Net
from .model_config import ModelConfig
from .utils import create_study_path, global_seed
from .model_execution_strategy import ModelExecutionStrategy


def get_activation_function(name):
    return {
        "ReLU": nn.ReLU(),
        "Tanh": nn.Tanh(),
        "Sigmoid": nn.Sigmoid(),
        "Leaky": nn.LeakyReLU(),
    }[name]


def get_transformation_function(name):
    return {
        "log": np.log,
        "sqrt": np.sqrt,
        "none": lambda x: x,
    }[name]


def display_best_trial_info(trial):
    print(f"Best trial:\n  Value: {trial.value}\n  Params: ")
    for param_name, param_value in trial.params.items():
        print(f"    {param_name}: {param_value}")


class HyperparameterOptimizer(ModelExecutionStrategy):
    def __init__(self, config: ModelConfig):
        self.config = config

    def execute(self):
        study = self.create_study()
        display_best_trial_info(study.best_trial)
        self.export_study_results(study)

    def create_study(self):
        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
        study.optimize(self.optimize_model, int(self.config.N_TRIALS))
        return study

    def export_study_results(self, study):
        results_df = study.trials_dataframe().drop(
            columns=["datetime_start", "datetime_complete", "duration"]
        )
        results_path = create_study_path(self.config.ACTIVE_RESPONSE_VARS, self.config.STUDY_DIR)
        results_df.to_csv(results_path, index=False)

    def optimize_model(self, trial: optuna.Trial):
        global_seed()

        # fmt: off
        hyperparams = {
            "epochs": trial.suggest_int("epochs", 50, 500),
            "batch_size": trial.suggest_int("batch_size", 10, 100),
            "train_size": trial.suggest_float("train_size", 0.1, 0.8, log=True),
            "num_layers": trial.suggest_int("num_layers", 1, self.config.NUM_LAYERS),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        }
        # fmt: on

        layer_sizes = (
            [len(self.config.INPUT_VARS)]
            + [trial.suggest_int(f"layer_{i + 1}", 1, 24) for i in range(hyperparams["num_layers"])]
            + [len(self.config.ACTIVE_RESPONSE_VARS)]
        )

        # fmt: off
        activation_funcs = [
            get_activation_function(
                trial.suggest_categorical(f"activation_{i + 1}", ["ReLU", "Tanh", "Sigmoid", "Leaky"])
                # trial.suggest_categorical(f"activation_{i + 1}", ["ReLU"])
                # trial.suggest_categorical(f"activation_{i + 1}", ["Tanh"])
                # trial.suggest_categorical(f"activation_{i + 1}", ["Sigmoid"])
                # trial.suggest_categorical(f"activation_{i + 1}", ["Leaky"])
            )
            for i in range(hyperparams["num_layers"])
        ]
        # fmt: on

        neural_network = Net(layer_sizes, activation_funcs)
        model_instance = NeuralNetworkModel(self.config, neural_network, hyperparams)

        mse, mape, r2 = model_instance.run()
        trial.set_user_attr("mse", mse.item())
        trial.set_user_attr("mape", mape.item())
        trial.set_user_attr("r2", r2.item())

        return mse
