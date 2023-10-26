import os
import pandas as pd
from sklearn.ensemble import IsolationForest

# Chi: 3, Gel: 3, Gly, 3, Pec: 3, Sta: 3, Oil: 3, T(ºC): 0; %RH: 0, t(h): 0, TS: 2, WVP: 2, E: 2
dv = [3, 3, 3, 3, 3, 3, 0, 0, 0, 2, 2, 2]

# Set paths
raw_data_path = os.environ.get("RAW_DATA")
raw_synthetic_data_path = os.environ.get("RAW_SYNTHETIC_DATA")
cleaned_data_file_path = os.environ.get("CLEANED_FILE")


def clean_data(data: pd.DataFrame, model=None):
    data = data.dropna().reset_index(drop=True)
    data = data.drop_duplicates().reset_index(drop=True)

    # Si no se proporciona un modelo, entrena uno
    if model is None:
        model = IsolationForest(contamination=0.20)
        outliers = model.fit_predict(data)
    else:
        outliers = model.predict(data)

    clean_data = data[outliers == 1]
    clean_data = clean_data.round(4)

    return clean_data, model


# Read raw data and clean it
df = pd.read_csv(raw_data_path)
_, model = clean_data(df)

# Read raw synthetic data, clean it (imputing missing values) and save it
df_synthetic = pd.read_csv(raw_synthetic_data_path)
df_clean_synthetic, _ = clean_data(df_synthetic, model)
df_clean_synthetic.to_csv(cleaned_data_file_path, index=False)
