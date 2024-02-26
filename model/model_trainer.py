import torch.nn as nn
from .model import NeuralNetworkModel
from .model_architecture import Net
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
            "num_layers": 2,
            "batch_size": 70,
            "epochs": 500,
            "train_size": 0.8,
            "weight_decay": 0.0001,
            "learning_rate": 0.1,
        }

    def execute(self):
        layers = [len(self.config.INPUT_VARS), *[6, 17], len(self.config.ACTIVE_RESPONSE_VARS)]

        activations = [self.activation_functions("Sigmoid"), self.activation_functions("Leaky")]
        params = self.params()

        net = Net(layers, activations)
        model = NeuralNetworkModel(self.config, net, params)
        mape, r2 = model.run()

        print("Params: ", params)
        print("Layers: ", layers)
        print("Activations: ", activations)

        print(f"MRE: {mape}")
        print(f"R2: {r2}")
