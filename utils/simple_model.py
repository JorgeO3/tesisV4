import os
import sys
import copy

import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
# Save models
pd.set_option('display.max_rows', None)
RESPONSE_VARIABLES = ["TS", "WVP", "%E"]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModelDataset(Dataset):
    '''
    Prepare the dataset for regression
    '''

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.double)
        self.y = torch.tensor(y, dtype=torch.double)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def normalizer(X_train, y_train, X_test, y_test, X_val, y_val):
    # Normalization of X data
    scaler_X = StandardScaler()
    scaler_X.fit(X_train)
    joblib.dump(scaler_X, 'scaler_X.save') 

    X_train = scaler_X.transform(X_train)
    X_test = scaler_X.transform(X_test)
    X_val = scaler_X.transform(X_val)

    # Normalization of y data
    scaler_y = StandardScaler()
    scaler_y.fit(y_train)
    joblib.dump(scaler_y, 'scaler_y.save') 

    y_train = scaler_y.transform(y_train)
    y_test = scaler_y.transform(y_test)
    y_val = scaler_y.transform(y_val)

    return X_train, y_train, X_test, y_test, X_val, y_val, scaler_X, scaler_y

class MLP(nn.Module):
    '''
    Multilayer Perceptron for regression.
    '''

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(11, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.Tanh(),
            nn.Linear(6, 3),
        )

    def forward(self, x):
        '''
        Forward pass
        '''
        return self.layers(x)


if __name__ == '__main__':
    # ================ Params for training =================
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    TRAIN_SIZE = 0.8
    WEIGHT_DECAY = 0.000528953198990119
    LEARNING_RATE = 0.058093027922883
    # ========================= // =========================

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../data/", "train_data.csv")
    synthetic_data_path = os.path.join(current_dir, "../data/", "synthetic_data_cleaned.csv")
    val_data_path = os.path.join(current_dir, "../data/", "test_data.csv")

    data = pd.read_csv(data_path).sample(frac=1).reset_index(drop=True)
    synthetic_data = pd.read_csv(synthetic_data_path).values
    val_data = pd.read_csv(val_data_path)

    df = pd.DataFrame(np.append(data.values, synthetic_data, axis=0), columns=data.columns)
    df = df.sample(frac=1).reset_index(drop=True)

    y = df[RESPONSE_VARIABLES].values
    X = df.drop(df.columns[[11, 12, 13]], axis=1).values

    y_val = val_data[RESPONSE_VARIABLES].values
    X_val = val_data.drop(df.columns[[11, 12, 13]], axis=1).values

    # Splitting the dataset
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, shuffle=True)

    # Normalize the data
    X_train, y_train, X_test, y_test, X_val, y_val, scaler_X, scaler_y = normalizer(
        X_train_raw, y_train, X_test_raw, y_test, X_val, y_val)

    # Generate test tensors
    X_test = torch.tensor(X_test, dtype=torch.double).to(DEVICE)
    y_test = torch.tensor(y_test, dtype=torch.double).to(DEVICE)

    # Generate tensor dataset
    train_dataset = ModelDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    mlp = MLP().to(DEVICE).double()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Metrics for analysis
    best_mse = np.inf
    best_weights = None
    history = []

    for i in range(NUM_EPOCHS):
        mlp.train()

        for _, (inputs, targets) in enumerate(train_loader, 0):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            outputs = mlp(inputs)
            loss = loss_function(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mlp.eval()
        y_pred = mlp(X_test)

        mse = loss_function(y_pred, y_test)
        mse = float(mse)
        r2 = r2_score(y_pred, y_test)
        history.append(mse)

        print(f"=========== MSE - EPOCH: {i} ==========")
        print(f"MSE: {mse}, R2: {r2}")
        print("========================================\n")

        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(mlp.state_dict())
            torch.save(mlp.state_dict(), "model_weights.pth")

    mlp.load_state_dict(best_weights) # type: ignore
    print("MSE: %.2f" % best_mse)
    print("RMSE: %.2f" % np.sqrt(best_mse))

    plt.plot(history)
    plt.show()

    # Generate test tensors
    X_val = torch.tensor(X_val, dtype=torch.double)
    y_val = torch.tensor(y_val, dtype=torch.double)

    torch.save(mlp, "mlp-model.pth")

    # Make predictions with the model
    mlp.eval()
    with torch.no_grad():
        for i in range(len(X_val)):
            inputs, target = X_val[i].to(DEVICE), y_val[i].to(DEVICE)
            preds = mlp(inputs)
            
            r2_c = r2_score(preds, target)
            mae = mean_absolute_error(preds, target)
            mape = mean_absolute_percentage_error(preds, target)
            rmse = mean_squared_error(preds, target, False)

            # Desescala las predicciones, los objetivos y las entradas
            inputs = scaler_X.inverse_transform(inputs.cpu().numpy().reshape(1, -1))
            preds = scaler_y.inverse_transform(preds.cpu().numpy().reshape(1, -1))
            targets = scaler_y.inverse_transform(target.cpu().numpy().reshape(1, -1))
            
            # Convierte los arrays de numpy en dataframes de pandas
            inputs_df = pd.DataFrame(inputs, columns=["%Chi", "%Gel", "%Gly", "%Pec", "%Sta", "%Oil", "%W", "%AA", "T(°C)", "%RH", "t(h)"])  # Coloca los nombres de tus features
            targets_df = pd.DataFrame(targets, columns=["TS", "WVP", "%E"]) 
            preds_df = pd.DataFrame(preds, columns=["TS", "WVP", "%E"])
            
            # Reducir el número de decimales para una mejor visualización
            pd.options.display.float_format = "{:,.2f}".format
            
            preds = torch.tensor(preds, dtype=torch.double)
            targets = torch.tensor(targets, dtype=torch.double)

            print(f"MAE: {mae:.4f} - MAPE: {mape:.4f} - RMSE: {rmse:.4f}")
            print("\nInputs:\n", inputs_df.to_string(index=False))
            print("\nTargets:\n", targets_df.to_string(index=False))
            print("\nPredictions:\n", preds_df.to_string(index=False))
            print('-' * 60 + '\n')