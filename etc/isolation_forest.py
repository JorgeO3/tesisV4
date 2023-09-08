import os
import pandas as pd
from sklearn.ensemble import IsolationForest

folder = "gretel_74_v2_s1"
current_dir = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(current_dir, "../data/data.csv")
clean_data_path = os.path.join(current_dir, "../data/clean_data.csv")
synthetic_data_path = os.path.join(
    current_dir, f"../data/{folder}", f"{folder}.csv.gz")
clean_synthetic_path = os.path.join(
    current_dir, f"../data/{folder}", "synthetic_gretel.csv")

# Función para limpiar datos


def clean_data(data, model=None):
    data = data.dropna()
    data = data.drop_duplicates()

    # Si no se proporciona un modelo, entrena uno
    if model is None:
        model = IsolationForest(contamination=0.20)
        outliers = model.fit_predict(data)
    else:
        outliers = model.predict(data)

    clean_data = data[outliers == 1]
    clean_data = clean_data.round(4)

    return clean_data, model


# Carga de datos originales
df = pd.read_csv(data_path)

# Limpieza de datos originales
df_clean, iso_forest_model = clean_data(df)
df_clean.to_csv(clean_data_path, index=False)

# Carga de datos sintéticos
df_synthetic = pd.read_csv(synthetic_data_path)

# Limpieza de datos sintéticos usando el modelo entrenado en datos originales
df_clean_synthetic, _ = clean_data(df_synthetic, iso_forest_model)
df_clean_synthetic.to_csv(clean_synthetic_path, index=False)
