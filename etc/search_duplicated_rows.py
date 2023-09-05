import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
# data_path = os.path.join(
#     current_dir, "../data/gretel_83_s1", "synthetic_gretel.csv")

df = pd.read_csv(os.path.join(current_dir, "../data/data.csv"))
# df = pd.read_csv(data_path)

# Identificar filas duplicadas
duplicated_rows = df[df.duplicated()]
print(f"Índices de las filas duplicadas:{duplicated_rows}")
# clean_data = df.drop_duplicates().reset_index(drop=True)
# print(clean_data)
