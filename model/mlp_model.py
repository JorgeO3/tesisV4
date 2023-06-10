import torch.nn as nn


class MLP(nn.Module):
    '''
    Multilayer Perceptron for regression.
    '''

    def __init__(self, layers, activation_functions):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(activation_functions):
                self.layers.append(activation_functions[i])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
