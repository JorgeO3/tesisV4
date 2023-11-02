import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from torch.nn.functional import mse_loss

# df = pd.read_csv("./data/gretel_70/test_data.csv")

# data = df.values
# data[:, [9, 10, 11]] = np.log1p(data[:, [9, 10, 11]])


# transform = FunctionTransformer(func=np.log1p)
# transformed_data = transform.fit_transform(df.values[:, [9, 10, 11]])

# data2 = train_test_split(data, test_size=0.2, random_state=42)

# print(transformed_data)
# print(transformed_data.shape)

# print(data)
# print(data.shape)
# print(data2[0])
# print(data2[1])


def loss_fn(output, target):
    # MAPE loss
    return torch.mean(torch.abs((target - output) / target))


dato1 = torch.tensor([1, 2, 3], dtype=torch.float32)
dato2 = torch.tensor([4, 5, 6], dtype=torch.float32)

print(loss_fn(dato1, dato2).item())
