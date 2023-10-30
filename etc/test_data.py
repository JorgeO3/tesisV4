import numpy as np
import pandas as pd

df = pd.read_csv("data/gretel_70/test_data.csv").values
x = df[:, :9]
y = df[:, 9:]
print(x)
print(y)

print(np.arange(1e-5, 1e-1, 1e-5))
