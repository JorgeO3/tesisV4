import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
# data_path = os.path.join(
#     current_dir, "../data/gretel_77_s1", "synthetic_gretel.csv")
data_path = os.path.join(current_dir, "../data/data.csv")
df = pd.read_csv(data_path)

# Suponiendo que df es tu DataFrame
rows_with_nan = df[df.isna().any(axis=1)]

print("Índices de las filas con NaN:")
print(rows_with_nan.index.tolist())

print("\nDetalles de las filas con NaN:")
print(rows_with_nan)
