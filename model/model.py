import joblib
import numpy as np
import torch as th
import pandas as pd
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torchmetrics.functional.regression.r2 import r2_score

from .model_config import ModelConfig
from .mlp_dataset import ModelDataset
from .error_functions import calculate_mape


class NeuralNetworkModel:
    def __init__(self, config: ModelConfig, network, hyperparameters) -> None:
        self.config = config
        self.network = network
        self.hyperparameters = hyperparameters

    def loss_function(self, predicted, actual):
        return th.mean(th.abs((actual - predicted) / actual))

    def preprocess_data(self, data):
        response_variables = self.config.IND_RESPONSE_VARS
        data[:, response_variables] = np.log1p(data[:, response_variables])
        return data

    def create_data_loader(self, data):
        dataset = ModelDataset(data[0], data[1])
        return DataLoader(dataset, batch_size=self.hyperparameters["batch_size"])

    def split_data(self, data) -> tuple[np.ndarray, np.ndarray]:
        input_variables = len(self.config.INPUT_VARS)
        response_variables = self.config.IND_RESPONSE_VARS
        return data[:, :input_variables], data[:, response_variables]

    def normalize_data(self, training_data, testing_data):
        training_x, training_y = training_data
        testing_x, testing_y = testing_data

        scaler_x = StandardScaler()
        scaler_x.fit(training_x)

        training_x = scaler_x.transform(training_x)
        testing_x = scaler_x.transform(testing_x)

        scaler_y = StandardScaler()
        scaler_y.fit(training_y)

        training_y = scaler_y.transform(training_y)
        testing_y = scaler_y.transform(testing_y)

        if self.config.SAVE_MODEL:
            joblib.dump(scaler_x, self.config.SCALER_X)
            joblib.dump(scaler_y, self.config.SCALER_Y)

        return (training_x, training_y), (testing_x, testing_y), (scaler_x, scaler_y)

    def run(self):
        mape_values = []
        early_stopping_patience = 10
        epochs_without_improvement = 0
        best_mse = float("inf")
        device = self.config.DEVICE

        self.network.to(device)
        loss_fn = nn.MSELoss()
        optimizer = Adam(
            self.network.parameters(),
            lr=self.hyperparameters["learning_rate"],
            weight_decay=self.hyperparameters["weight_decay"],
        )

        data = pd.read_csv(self.config.TRAIN_DATA_PATH)
        data = self.preprocess_data(data.values)

        train_data, test_data = train_test_split(
            data, train_size=self.hyperparameters["train_size"], random_state=42
        )

        train = self.split_data(train_data)
        test = self.split_data(test_data)

        train_data, test_data, scalers = self.normalize_data(train, test)
        train_loader = self.create_data_loader(train_data)

        x_test = th.tensor(test_data[0], dtype=th.float32).to(device)
        y_test = th.tensor(test_data[1], dtype=th.float32).to(device)

        th.manual_seed(self.config.SEED)
        for epoch in range(self.hyperparameters["epochs"]):
            self.network.train()

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = self.network(x_batch)
                loss = loss_fn(predictions, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.network.eval()
            with th.no_grad():
                y_predicted = self.network(x_test)

                mse = loss_fn(y_predicted, y_test)
                mape = calculate_mape(y_predicted, y_test, scalers[1])
                r2 = r2_score(y_predicted, y_test)
                mape_values.append(mape)

                if self.config.DEBUG:
                    print(f"=========== MSE - EPOCH: {epoch} ==========")
                    print(f"MSE: {mse}, R2: {r2}, MAPE: {mape}")
                    print("========================================\n")

                if mse < best_mse:
                    best_mse = mse
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= early_stopping_patience and self.config.STOPPING:
                    print("Early stopping!")
                    break

        average_mape = sum(mape_values) / len(mape_values)
        return best_mse, average_mape, r2
