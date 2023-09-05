import torch
import optuna
import datetime
import pandas as pd

from torch import nn
from optuna.trial import Trial
from .mlp_model import MLP
from .base_model import BaseModel
from .model_config import ModelConfig
from .model_execution_strategy import ModelExecutionStrategy


class ModelOptimization(ModelExecutionStrategy):
    def __init__(self, config: ModelConfig, n_trials: int = 2000) -> None:
        self.config = config
        self.n_trials = n_trials
        self.device = self.config.DEVICE
        self.data_path = self.config.DATA_PATH
        self.study_csv_path = self.config.STUDY_CSV_PATH
        self.synthetic_data_path = self.config.SYNTHETIC_DATA_PATH

    def execute(self):
        study = self.create_optuna_study()
        self.print_best_trial(study.best_trial)
        self.save_study_as_csv(study)

    def create_optuna_study(self) -> optuna.Study:
        study = optuna.create_study()
        study.optimize(self.trial, n_trials=self.n_trials)
        return study

    def print_best_trial(self, trial: optuna.trial.FrozenTrial) -> None:
        print("Best trial:")
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    def save_study_as_csv(self, study: optuna.Study) -> None:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        active_vars = ''.join(self.config.ACTIVE_RESPONSE_VARS)

        save_path = f"results_{timestamp}_{active_vars}.csv"
        study_path = self.config.create_path(save_path)
        df = study.trials_dataframe()
        df.to_csv(study_path, index=False)

    def activation_functions(self, activation_name):
        functions = {
            'ReLU': nn.ReLU(),
            'Tanh': nn.Tanh(),
            'Sigmoid': nn.Sigmoid(),
            'Leaky': nn.LeakyReLU()
        }
        return functions[activation_name]

    def trial(self, trial: Trial) -> float:
        max_layers = self.config.NUM_LAYERS
        fl_range = (1, 24)
        train_size_range = (0.1, 0.8)
        weight_decay_range = (1e-5, 1e-1)
        learning_rate_range = (1e-5, 1e-1)
        batch_size_options = (10, 100)
        num_epochs_options = (50, 500)
        activation_functions_options: list[str] = [
            'ReLU', 'Tanh', 'Sigmoid', 'Leaky']

        config = {
            'params_n_layers': trial.suggest_int('n_layers', 1, max_layers),
            'params_batch_size': trial.suggest_int('batch_size', *batch_size_options),
            'params_num_epochs': trial.suggest_int('num_epochs', *num_epochs_options),
            'params_train_size': trial.suggest_float('train_size', *train_size_range, log=True),
            'params_weight_decay': trial.suggest_float('weight_decay', *weight_decay_range, log=True),
            'params_learning_rate': trial.suggest_float('learning_rate', *learning_rate_range, log=True),
        }
        trial.study.set_user_attr('config', config)

        layers = [11]
        activations = []

        for i in range(config['params_n_layers']):
            layer = trial.suggest_int(f'fl{i + 1}', *fl_range)
            layers.append(layer)

            activation = trial.suggest_categorical(
                f'activation{i + 1}', activation_functions_options)
            activations.append(self.activation_functions(activation))

        layers.append(len(self.config.ACTIVE_RESPONSE_VARS))

        # Instance base model and pass everything
        model = MLP(layers, activations)

        paths = [self.config.SYNTHETIC_DATA_PATH, self.config.DATA_PATH]
        base_model = BaseModel(paths=paths, model=model,
                               config=self.config,
                               device=self.device,
                               batch_size=config["params_batch_size"],
                               num_epochs=config["params_num_epochs"],
                               train_size=config["params_train_size"],
                               learning_rate=config["params_learning_rate"],
                               weight_decay=config["params_weight_decay"])

        mse, mre = base_model.train(self.config.DEBUG)
        trial.set_user_attr("mre", mre)
        return mse
