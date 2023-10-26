import os
import pandas as pd

data_path = os.environ.get("FILE_PATH")

df = pd.read_csv(data_path)

# Suponiendo que df es tu DataFrame
rows_with_nan = df[df.isna().any(axis=1)]
rows_with_duplicated = df[df.duplicated()]
rows_with_negative = df[(df < 0).any(axis=1)]

print("\nÍndices de las filas con NaN:")
print(rows_with_nan.index.tolist())

print("\nNumber of rows with NaN: ")
print(len(rows_with_nan.index.tolist()))

print("\nDetalles de las filas con NaN:")
print(rows_with_nan)

print("\nÍndices de las filas duplicadas:")
print(rows_with_duplicated.index.tolist())

print("\nNumber of rows with duplicated values: ")
print(len(rows_with_duplicated.index.tolist()))

print("\nDetalles de las filas negativas:")
print(rows_with_negative)
print("\n")
