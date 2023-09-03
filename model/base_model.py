import torch
import numpy as np
import joblib

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torchmetrics.functional.regression.r2 import r2_score

from .mlp_dataset import ModelDataset
from .model_config import ModelConfig


class BaseModel:
    def __init__(self, config: ModelConfig, paths, model, device, batch_size, num_epochs, train_size, learning_rate, weight_decay):
        self.config = config
        self.paths = paths
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.train_size = train_size
        self.loss_function = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay)
        self.save_scaler = self.config.SAVE_SCALER
        self.scaler_path = self.config.SCALER_PATH

    def generate_dataloader(self):
        data = self.config.concatenate_data(self.paths)

        raw_train_data, raw_test_data = train_test_split(
            data, train_size=self.train_size, shuffle=True)

        scaler = StandardScaler()
        scaler.fit(raw_train_data)

        train_data = scaler.transform(raw_train_data)
        test_data = scaler.transform(raw_test_data)

        if self.save_scaler:
            joblib.dump(scaler, self.scaler_path)

        train_x, train_y = self.config.data_processor(train_data)
        train_dataset = ModelDataset(train_x, train_y)

        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True), test_data

    def train(self, debug: bool):
        self.model.to(self.device)
        best_mse = np.inf
        history = []

        train_loader, test_data = self.generate_dataloader()
        X_test, y_test = self.config.data_processor(test_data)

        # Generate test tensors
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        for i in range(self.num_epochs):
            self.model.train()

            for _, (inputs, targets) in enumerate(train_loader, 0):
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(X_test)

                mse = self.loss_function(y_pred, y_test)
                mse = float(mse)

                history.append(mse)

                if debug:
                    r2 = r2_score(y_pred, self.y_test)
                    print(f"=========== MSE - EPOCH: {i} ==========")
                    print(f"MSE: {mse}, R2: {r2}")
                    print("========================================\n")

                if mse < best_mse:
                    best_mse = mse

        return best_mse
