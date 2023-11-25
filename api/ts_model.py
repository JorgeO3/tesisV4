import os
import torch
import joblib
import numpy as np
import torch.nn as nn

MODEL_TYPE = "ts"
SCALER_X_FILE = os.environ.get("SCALER_X_FILE")
SCALER_Y_FILE = os.environ.get("SCALER_Y_FILE")
MODEL_FILE = os.environ.get("MODEL_FILE")
MODELS_DIR = os.environ.get("MODELS_DIR")

MODEL_PATH = os.path.join(MODELS_DIR, MODEL_TYPE, MODEL_FILE)
SCALER_X_PATH = os.path.join(MODELS_DIR, MODEL_TYPE, SCALER_X_FILE)
SCALER_Y_PATH = os.path.join(MODELS_DIR, MODEL_TYPE, SCALER_Y_FILE)

input_vars = ["Chi", "Gel", "Gly", "Pec", "Sta", "Oil", "T(°C)", "%RH", "t(h)"]
response_vars = ["TS"]


class MLP(nn.Module):
    """
    Multilayer Perceptron for regression.
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(9, 13),
            nn.LeakyReLU(),
            nn.Linear(13, 1),
        )

    def forward(self, x):
        """
        Forward pass
        """
        return self.network(x)


class TSModel:
    def __init__(self):
        self.model = MLP()
        self.model.load_state_dict(torch.load(MODEL_PATH))

    def generate_tensor(self, data):
        return torch.tensor(data, dtype=torch.float32)

    def unlog_transform(self, preds):
        return np.expm1(preds)

    def normalize_inputs(self, inputs):
        scaler_X = joblib.load(SCALER_X_PATH)
        return scaler_X.transform(inputs)

    def unnormalize_predictions(self, preds):
        scaler_y = joblib.load(SCALER_Y_PATH)
        return scaler_y.inverse_transform(preds)

    def inference(self, X):
        input = np.array(X)
        input = self.normalize_inputs(input)
        input = self.generate_tensor(input)

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(input)
            y_pred = self.unnormalize_predictions(y_pred)
            y_pred = self.unlog_transform(y_pred)
            return y_pred.tolist()
