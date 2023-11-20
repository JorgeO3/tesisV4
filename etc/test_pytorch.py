import torch
from torch import nn

loss = nn.MSELoss()
input = torch.tensor(1, dtype=torch.float32)
target = torch.tensor([2], dtype=torch.float32)
output = loss(input, target)

print(output)
