import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torchmetrics.functional import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

# Constants
SEED = 42
DEBUG = os.environ.get("DEBUG") == "1"
EARLY_STOPPING = os.environ.get("STOPPING") == "1"
SAVE_MODEL = os.environ.get("SAVE_MODEL") == "1"
ACTIVE_RESPONSE_VARIABLES = ["WVP"]
RESPONSE_VARIABLES = ["TS", "WVP", "%E"]
SCALER_X_PATH = os.environ.get("SCALER_X")
SCALER_Y_PATH = os.environ.get("SCALER_Y")
VAL_DATA_PATH = os.environ.get("VAL_DATA_PATH")
TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA_PATH")

# Setting seeds for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModelDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 12),
            nn.ReLU(),
            nn.Linear(12, output_size),
        )

    def forward(self, x):
        return self.network(x)


def compute_mre(predictions, targets):
    return (torch.abs(targets - predictions)) / targets


def loss_fn(output, target):
    # MAPE loss
    return torch.mean(torch.abs((target - output) / target))


def train_model(model, train_loader, optimizer, loss_fn):
    model.train()
    for features, targets in train_loader:
        features, targets = features.to(DEVICE), targets.to(DEVICE)
        predictions = model(features)
        loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate_model(model, features, targets):
    model.eval()
    with torch.no_grad():
        predictions = model(features)
        mse = mean_squared_error(predictions, targets)
        mae = mean_absolute_error(predictions, targets)
        mape = mean_absolute_percentage_error(predictions, targets)
        r2 = r2_score(predictions, targets)
        return mse, mae, mape, r2


def create_tensor(x):
    return torch.tensor(x, dtype=torch.float32).to(DEVICE)


def make_predictions(model, features, targets, scaler_x, scaler_y):
    model.eval()
    mre_list = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i in range(len(features)):
            input_feature, target = features[i], targets[i]
            prediction = model(input_feature)

            # Almacenar todas las predicciones y objetivos para el cálculo del error relativo promedio
            all_preds.append(prediction)
            all_targets.append(target)

            mae = mean_absolute_error(prediction, target)
            mape = mean_absolute_percentage_error(prediction, target)
            rmse = mean_squared_error(prediction, target, False)
            mre = compute_mre(prediction, target).flatten()[0]

            # Transformar las predicciones, objetivos y características a la escala original
            input_feature = scaler_x.inverse_transform(input_feature.cpu().numpy().reshape(1, -1))
            prediction = scaler_y.inverse_transform(prediction.cpu().numpy().reshape(1, -1))
            target = scaler_y.inverse_transform(target.cpu().numpy().reshape(1, -1))

            prediction = np.expm1(prediction)
            target = np.expm1(target)

            # Convertir los arrays de numpy a dataframes de pandas y darles nombre a las columnas
            input_df = pd.DataFrame(
                input_feature,
                columns=["%Chi", "%Gel", "%Gly", "%Pec", "%Sta", "%Oil", "T(°C)", "%RH", "t(h)"],
            )
            target_df = pd.DataFrame(target, columns=["WVP"])
            pred_df = pd.DataFrame(prediction, columns=["WVP"])

            # Reducir el número de decimales para una mejor visualización
            pd.options.display.float_format = "{:,.2f}".format

            print(f"MAE: {mae:.4f} - MAPE: {mape:.4f} - RMSE: {rmse:.4f} - MRE: {mre}")
            print("\nInputs:\n", input_df.to_string(index=False))
            print("\nTargets:\n", target_df.to_string(index=False))
            print("\nPredictions:\n", pred_df.to_string(index=False))

            mre_list.append(mre)

    # Calcular el error relativo promedio para todas las predicciones
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    overall_mre = compute_mre(all_preds, all_targets).mean().item()

    print(f"Error Relativo Promedio General: {overall_mre:.4f}")

    return mre_list


def main(batch_size, num_epochs, train_size, weight_decay, learning_rate):
    # Load data
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    val_data = pd.read_csv(VAL_DATA_PATH)

    # Train-test split
    train_data, test_data = train_test_split(train_data, train_size=train_size, random_state=SEED)

    # Feature-target split
    x_train, y_train = (
        train_data.drop(RESPONSE_VARIABLES, axis=1),
        train_data[ACTIVE_RESPONSE_VARIABLES],
    )

    x_test, y_test = (
        test_data.drop(RESPONSE_VARIABLES, axis=1),
        test_data[ACTIVE_RESPONSE_VARIABLES],
    )
    x_val, y_val = val_data.drop(RESPONSE_VARIABLES, axis=1), val_data[ACTIVE_RESPONSE_VARIABLES]

    # Logarithmic transformation
    y_train = np.log1p(y_train)
    y_test = np.log1p(y_test)
    y_val = np.log1p(y_val)

    # Scaling
    scaler_x = StandardScaler().fit(x_train)
    scaler_y = StandardScaler().fit(y_train)

    x_train = scaler_x.transform(x_train)
    y_train = scaler_y.transform(y_train)
    x_test = scaler_x.transform(x_test)
    y_test = scaler_y.transform(y_test)
    x_val = scaler_x.transform(x_val)
    y_val = scaler_y.transform(y_val)

    # Save scalers
    if SAVE_MODEL:
        joblib.dump(scaler_x, SCALER_X_PATH)
        joblib.dump(scaler_y, SCALER_Y_PATH)

    # Prepare datasets and dataloaders
    train_dataset = ModelDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    x_test, y_test = create_tensor(x_test), create_tensor(y_test)
    x_val, y_val = create_tensor(x_val), create_tensor(y_val)

    # Initialize model
    input_size = x_train.shape[1]
    output_size = y_train.shape[1]
    model = MLP(input_size, output_size).to(DEVICE)

    # Loss function and optimizer
    # loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    best_mse = float("inf")
    history = []
    patience = 10
    no_improve = 0

    for epoch in range(num_epochs):
        train_model(model, train_loader, optimizer, loss_fn)
        mse, mae, mape, r2 = evaluate_model(model, x_test, y_test)
        history.append(mse.item())

        if DEBUG:
            print(f"Epoch {epoch}: MSE={mse:.4}, MAE={mae:.4}, MAPE={mape:.4}, R2={r2:.4}")

        if mse < best_mse:
            best_mse = mse
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience and EARLY_STOPPING:
            print("Early stopping!")
            break

    print(f"Best MSE: {best_mse}")
    plt.plot(history)
    plt.show()

    # Make predictions
    _mre_list = make_predictions(model, x_val, y_val, scaler_x, scaler_y)

    # Save model
    if SAVE_MODEL:
        torch.save(model.state_dict(), "mlp_model.pt")


# Example parameters
batch_size = 32
num_epochs = 200
train_size = 0.8
weight_decay = 0.001
learning_rate = 0.001

if __name__ == "__main__":
    main(batch_size, num_epochs, train_size, weight_decay, learning_rate)
