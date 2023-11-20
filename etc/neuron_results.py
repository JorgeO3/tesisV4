import os
import csv
import torch
import numpy as np
import pandas as pd

from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torchmetrics.functional.regression.r2 import r2_score

# Constants
SEED = 42
ACTIVE_RESPONSE_VARIABLES = ["%E"]
RESPONSE_VARIABLES = ["TS", "WVP", "%E"]

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

VAL_DATA_PATH = os.environ.get("VAL_DATA_PATH")
TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA_PATH")


# fmt: off
def activation_function(x):
    return {
        "tanh": nn.Tanh(), 
        "relu": nn.ReLU(), 
        "sigmoid": nn.Sigmoid(), 
        "leaky": nn.LeakyReLU(),
    }[x]
# fmt: on


def global_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ModelDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class MLP(nn.Module):
    def __init__(self, input_size, output_size, n_neurons, afunction):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, n_neurons),
            activation_function(afunction),
            nn.Linear(n_neurons, output_size),
        )

    def forward(self, x):
        return self.network(x)


def create_tensor(x):
    return torch.tensor(x, dtype=torch.float32).to(DEVICE)


def train_by_neurons(
    n_neurons, a_function, batch_size, num_epochs, train_size, weight_decay, learning_rate
):
    # Load data
    train_data = pd.read_csv(TRAIN_DATA_PATH)

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

    # Logarithmic transformation
    y_train = np.log1p(y_train)
    y_test = np.log1p(y_test)

    # Scaling
    scaler_x = StandardScaler().fit(x_train)
    scaler_y = StandardScaler().fit(y_train)

    x_train = scaler_x.transform(x_train)
    y_train = scaler_y.transform(y_train)
    x_test = scaler_x.transform(x_test)
    y_test = scaler_y.transform(y_test)

    # Prepare data for training and testing
    train_dataset = ModelDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    x_test, y_test = create_tensor(x_test), create_tensor(y_test)

    # Initialize model
    input_size = x_train.shape[1]
    output_size = y_train.shape[1]
    model = MLP(input_size, output_size, n_neurons, a_function).to(DEVICE)

    # Loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    mse = float("inf")
    r2 = None

    for _ in range(num_epochs):
        model.train()
        for input, target in train_loader:
            input, target = input.to(DEVICE), target.to(DEVICE)

            preds = model(input)
            loss = loss_fn(preds, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(x_test)
            mse = loss_fn(preds, y_test).item()
            r2 = r2_score(preds, y_test)

    with open(f"{ACTIVE_RESPONSE_VARIABLES[0]}_results.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([ACTIVE_RESPONSE_VARIABLES[0], a_function, n_neurons, mse, r2])

    print(
        f"Var: {ACTIVE_RESPONSE_VARIABLES[0]}, func: {a_function}, Neurons: {n_neurons}, MSE: {mse:.4}, R2: {r2:.4}"
    )


def main():
    # Parameters
    # TS
    # batch_size = 68
    # num_epochs = 227
    # train_size = 0.6609677444187338
    # weight_decay = 0.0003377668780981758
    # learning_rate = 0.0033575076398968512
    # functions = ["tanh", "relu", "leaky", "sigmoid"]

    # WVP
    # batch_size = 62
    # num_epochs = 287
    # train_size = 0.6635009939260634
    # weight_decay = 9.844325856384082e-05
    # learning_rate = 0.008470692972985645

    # %E
    batch_size = 11
    num_epochs = 274
    train_size = 0.7681075112311813
    weight_decay = 2.5630843936694257e-05
    learning_rate = 0.004243753652173263

    functions = ["tanh", "relu", "leaky", "sigmoid"]

    for function in functions:
        for n_neurons in range(1, 25):
            # set global seed
            global_seed()
            train_by_neurons(
                n_neurons=n_neurons,
                a_function=function,
                batch_size=batch_size,
                num_epochs=num_epochs,
                train_size=train_size,
                weight_decay=weight_decay,
                learning_rate=learning_rate,
            )


main()
