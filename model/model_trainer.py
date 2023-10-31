import torch.nn as nn
from .model import Model
from .model_architecure import MLP
from .model_config import ModelConfig
from .model_execution_strategy import ModelExecutionStrategy


class ModelTraining(ModelExecutionStrategy):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def activation_functions(self, activation_name):
        functions = {
            "ReLU": nn.ReLU(),
            "Tanh": nn.Tanh(),
            "Sigmoid": nn.Sigmoid(),
            "Leaky": nn.LeakyReLU(),
        }
        return functions[activation_name]

    def params(self):
        return {
            "n_layers": 1,
            "batch_size": 20,
            "num_epochs": 150,
            "train_size": 0.8,
            "weight_decay": 0.001,
            "learning_rate": 0.001,
        }

    def execute(self):
        # The first layer is 9 because the input has 9 variables
        # The last layer is 3 because the output has 3 variables
        layers = [9, *[24], 3]

        activations = [self.activation_functions("Leaky"), self.activation_functions("Tanh")]
        params = self.params()

        net = MLP(layers, activations)
        model = Model(self.config, net, params)
        mse, mre, r2 = model.run()

        print("Params: ", params)
        print("Layers: ", layers)
        print("Activations: ", activations)

        print(f"MSE: {mse}")
        print(f"MRE: {mre}")
        print(f"R2: {r2}")
