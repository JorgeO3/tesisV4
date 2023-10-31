import joblib
import torch as th
import pandas as pd
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torchmetrics.functional.regression.r2 import r2_score

from .utils import seed_worker
from .model_config import ModelConfig
from .mlp_dataset import ModelDataset


class Model:
    def __init__(self, config: ModelConfig, core, params) -> None:
        self.config = config
        self.core = core
        self.params = params

    def compute_mre(self, preds, target, epsilon=1e-10):
        mre = th.mean(th.abs(preds - target) / (th.abs(target) + epsilon)) * 100
        return mre.item()

    def gen_data(self, data):
        x, y = self.split_data(data)
        dataset = ModelDataset(x, y)
        return DataLoader(
            dataset,
            batch_size=self.params["batch_size"],
            generator=self.config.GENERATOR,
            worker_init_fn=seed_worker,
        )

    def split_data(self, data):
        length = len(self.config.INPUT_VARS)
        return data[:, :length], data[:, length:]

    def scale_data(self, train, test):
        scaler = StandardScaler()
        scaler.fit(train)

        train = scaler.transform(train)
        test = scaler.transform(test)

        if self.config.SAVE_MODEL:
            joblib.dump(scaler, self.config.SCALER_PATH)

        return train, test

    def run(self):
        mre_list = []
        patience = 10
        no_improve = 0
        r2 = None
        best_mse = float("inf")
        device = self.config.DEVICE

        self.core.to(device)
        loss_fn = nn.MSELoss()
        optimizer = Adam(
            self.core.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
        )

        data = pd.read_csv(self.config.TRAIN_DATA_PATH)
        train, test = train_test_split(data, train_size=self.params["train_size"], random_state=42)
        train, test = self.scale_data(train, test)

        train_loader = self.gen_data(train)
        x_test, y_test = self.split_data(test)

        # Generate the tensors for the test data
        x_test = th.tensor(x_test, dtype=th.float32).to(device)
        y_test = th.tensor(y_test, dtype=th.float32).to(device)

        for i in range(self.params["num_epochs"]):
            self.core.train()

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                preds = self.core(x)
                loss = loss_fn(preds, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.core.eval()
            with th.no_grad():
                y_pred = self.core(x_test)

                mse = loss_fn(y_pred, y_test)
                mre = self.compute_mre(y_pred, y_test)
                r2 = r2_score(y_pred, y_test)
                mre_list.append(mre)

                if self.config.DEBUG:
                    print(f"=========== MSE - EPOCH: {i} ==========")
                    print(f"MSE: {mse}, R2: {r2}")
                    print("========================================\n")

                if mse < best_mse:
                    best_mse = mse
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience and self.config.STOPPING:
                    print("Early stopping!")
                    break

        avg_mre = sum(mre_list) / len(mre_list)
        return best_mse, avg_mre, r2
