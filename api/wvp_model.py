import os
import torch
import csv
import random
import joblib
import sys
import numpy as np
import torch.nn as nn
import pandas as pd


# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)

MODEL_TYPE = "wvp"
SCALER_X_FILE = os.environ.get("SCALER_X_FILE")
SCALER_Y_FILE = os.environ.get("SCALER_Y_FILE")
MODEL_FILE = os.environ.get("MODEL_FILE")
MODELS_DIR = os.environ.get("MODELS_DIR")

MODEL_PATH = os.path.join(MODELS_DIR, MODEL_TYPE, MODEL_FILE)
SCALER_X_PATH = os.path.join(MODELS_DIR, MODEL_TYPE, SCALER_X_FILE)
SCALER_Y_PATH = os.path.join(MODELS_DIR, MODEL_TYPE, SCALER_Y_FILE)

input_vars = ["Chi", "Gel", "Gly", "Pec", "Sta", "Oil", "T(°C)", "%RH", "t(h)"]
response_vars = ["PVA"]


class MLP(nn.Module):
    """
    Multilayer Perceptron for regression.
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(9, 22),
            nn.ReLU(),
            nn.Linear(22, 1),
        )

    def forward(self, x):
        """
        Forward pass
        """
        return self.network(x)


class WVPModel:
    def __init__(self):
        self.model = MLP()
        self.model.load_state_dict(torch.load(MODEL_PATH))

    def save_scaler_params(self, scaler_X, scaler_y):
        scaler_x_mean, scaler_x_std = scaler_X.mean_, scaler_X.scale_
        scaler_y_mean, scaler_y_std = scaler_y.mean_, scaler_y.scale_

        with open(f"mean_std_{MODEL_TYPE}.csv", "a") as f:
            for i in range(len(scaler_x_mean)):
                writer = csv.writer(f)
                writer.writerow(
                    [
                        input_vars[i],
                        scaler_x_mean[i],
                        scaler_x_std[i],
                    ]
                )
            writer.writerow([response_vars[0], scaler_y_mean[0], scaler_y_std[0]])

    def generate_tensor(self, data):
        return torch.tensor(data, dtype=torch.float32)

    def unlog_transform(self, preds):
        return np.expm1(preds)

    def normalize_inputs(self, inputs):
        scaler_X = joblib.load(SCALER_X_PATH)
        return scaler_X.transform(inputs), scaler_X

    def unnormalize_predictions(self, preds):
        scaler_y = joblib.load(SCALER_Y_PATH)
        return scaler_y.inverse_transform(preds), scaler_y

    def inference(self, X):
        input = np.array(X)
        input, scaler_x = self.normalize_inputs(input)
        input = self.generate_tensor(input)

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(input)
            y_pred, scaler_y = self.unnormalize_predictions(y_pred)
            self.save_scaler_params(scaler_x, scaler_y)
            y_pred = self.unlog_transform(y_pred)
            return y_pred.tolist()
