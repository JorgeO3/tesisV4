import torch
from torch.utils.data import Dataset


class ModelDataset(Dataset):
    """
    Prepare the dataset for regression
    """

    def __init__(self, x, y):
        self.X = torch.Tensor(x, dtype=torch.float32)
        self.y = torch.Tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
