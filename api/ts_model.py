import os
import torch
import joblib
import numpy as np
import torch.nn as nn

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_FOLDER = os.path.join(CURRENT_DIR, "../trained_models/ts")


class MLP(nn.Module):
    '''
    Multilayer Perceptron for regression.
    '''

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(11, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 9),
            nn.ReLU(),
            nn.Linear(9, 9),
            nn.Tanh(),
            nn.Linear(9, 1),
        )

    def forward(self, x):
        '''
        Forward pass
        '''
        return self.layers(x)


class TSModel:
    def __init__(self):
        model_path = os.path.join(MODEL_FOLDER, "mlp-model.pth")
        self.model = MLP()
        self.model.load_state_dict(torch.load(model_path))

    def generate_tensor(self, data):
        return torch.tensor(data, dtype=torch.float32)

    def normalize_inputs(self, X):
        scaler_X_path = os.path.join(MODEL_FOLDER, "scaler_X.save")
        scaler_X = joblib.load(scaler_X_path)
        return scaler_X.transform(X)

    def unnormalize_predictions(self, preds):
        scaler_y_path = os.path.join(MODEL_FOLDER, "scaler_y.save")
        scaler_y = joblib.load(scaler_y_path)
        preds = preds.numpy().reshape(-1, 1)
        return scaler_y.inverse_transform(preds)

    def inference(self, X):
        input = np.array(X)
        normalized_input = self.normalize_inputs(input)
        input_tensor = self.generate_tensor(normalized_input)

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(input_tensor)

        y_pred = self.unnormalize_predictions(y_pred)
        y_pred = y_pred.reshape(-1, 1)
        combined = np.hstack((X, y_pred))

        return combined
