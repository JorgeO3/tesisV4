import os
import pandas as pd

current_dir = os.getcwd()
data_path = os.environ.get("DATA_PATH")
cleaned_data_file = os.environ.get("CLEANED_FILE")
data_path = os.path.join(current_dir, data_path, cleaned_data_file)

df = pd.read_csv(data_path)

# Suponiendo que df es tu DataFrame
rows_with_nan = df[df.isna().any(axis=1)]
rows_with_duplicated = df[df.duplicated()]
rows_with_negative = df[(df < 0).any(axis=1)]

print("Índices de las filas con NaN:")
print(rows_with_nan.index.tolist())

print("\nDetalles de las filas con NaN:")
print(rows_with_nan)

print("\nÍndices de las filas duplicadas:")
print(rows_with_duplicated.index.tolist())

print("\nDetalles de las filas negativas:")
print(rows_with_negative)
