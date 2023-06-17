import os
import torch
import random
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from torchmetrics.functional.regression.mse import mean_squared_error
from torchmetrics.functional.regression.mae import mean_absolute_error
from torchmetrics.functional.regression.mape import mean_absolute_percentage_error

from simple_model import MLP

SEED = 42
SAVE = False
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

current_dir = os.path.dirname(os.path.abspath(__file__))
X_scaler_path = os.path.join(current_dir, "scaler_X.save")
y_scaler_path = os.path.join(current_dir, "scaler_y.save")
model_path = os.path.join(current_dir, "mlp-model.pth")

X_scaler: StandardScaler = joblib.load(X_scaler_path)
y_scaler: StandardScaler = joblib.load(y_scaler_path)
mlp = torch.load(model_path, map_location=torch.device('cpu'))

input_vars = ["%Chi", "%Gel", "%Gly", "%Pec", "%Sta", "%Oil",
           "%W", "%AA", "T(°C)", "%RH", "t(h)"]

response_vars = ["TS", "WVP", "%E"]

vars = input_vars + response_vars

row = [[1.0, 0.0, 0.2, 0.0, 0.0, 0.0, 97.8, 1.0,
       60.0, 52.51, 12.0, 22.2, 69.78, 41.98]]

df = pd.DataFrame(row, columns=vars)

raw_X = df.drop(df.columns[[11, 12, 13]], axis=1).values
raw_y = df[response_vars].values

X = X_scaler.transform(raw_X)
y = y_scaler.transform(raw_y)

test = X_scaler.inverse_transform(X)
print (test)

inputs = torch.tensor(X, dtype=torch.float32)
targets = torch.tensor(y, dtype=torch.float32)

mlp.eval()
with torch.no_grad():
    preds = mlp(inputs)

    mae = mean_absolute_error(preds, targets)
    mape = mean_absolute_percentage_error(preds, targets)
    rmse = mean_squared_error(preds, targets, False)


    # Desescala las predicciones, los objetivos y las entradas
    inputs = X_scaler.inverse_transform(inputs.numpy().reshape(1, -1))
    targets = y_scaler.inverse_transform(targets.numpy().reshape(1, -1))
    preds = y_scaler.inverse_transform(preds.numpy().reshape(1, -1))

    # Convierte los arrays de numpy en dataframes de pandas
    inputs_df = pd.DataFrame(inputs, columns=input_vars)
    targets_df = pd.DataFrame(targets, columns=response_vars)
    preds_df = pd.DataFrame(preds, columns=response_vars)

    print(f"MAE: {mae:.4f} - MAPE: {mape:.4f} - RMSE: {rmse:.4f}")
    print("\nInputs:\n", inputs_df.to_string(index=False))
    print("\nTargets:\n", targets_df.to_string(index=False))
    print("\nPredictions:\n", preds_df.to_string(index=False))
    print('-' * 60 + '\n')

