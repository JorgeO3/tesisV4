import os
import sys
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torchmetrics.functional.regression.r2 import r2_score
from torchmetrics.functional.regression.mse import mean_squared_error
from torchmetrics.functional.regression.mae import mean_absolute_error
from torchmetrics.functional.regression.mape import mean_absolute_percentage_error

SEED = 42
SAVE = False
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
pd.set_option("display.max_rows", None)

RESPONSE_VARIABLES = ["TS", "WVP", "%E"]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
current_dir = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.join(current_dir, "../trained_models/wvp")


class ModelDataset(Dataset):
    """
    Prepare the dataset for regression
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class MLP(nn.Module):
    """
    Multilayer Perceptron for regression.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(9, 24),
            nn.Tanh(),
            nn.Linear(24, 22),
            nn.LeakyReLU(),
            nn.Linear(22, 9),
            nn.Tanh(),
            nn.Linear(9, 1),
        )

    def forward(self, x):
        """
        Forward pass
        """
        return self.layers(x)


def compute_mre(y_pred, y_true):
    """
    Compute the mean relative error
    """
    return (np.abs(y_true - y_pred)) / y_true


def main(BATCH_SIZE, NUM_EPOCHS, TRAIN_SIZE, WEIGHT_DECAY, LEARNING_RATE):
    TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA_PATH")
    TEST_DATA_PATH = os.environ.get("TEST_DATA_PATH")
    SCALER_PATH = os.environ.get("SCALER_PATH")


if __name__ == "__main__":
    BATCH_SIZE = 64
    NUM_EPOCHS = 1000
    TRAIN_SIZE = 0.7
    WEIGHT_DECAY = 0.01
    LEARNING_RATE = 0.001

    main(BATCH_SIZE, NUM_EPOCHS, TRAIN_SIZE, WEIGHT_DECAY, LEARNING_RATE)
