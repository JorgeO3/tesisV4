import os
import numpy as np
import pandas as pd
from scipy.stats import chi2


def load_data(path, compression=None):
    return pd.read_csv(path, compression=compression)


def clean_data(dataframe: pd.DataFrame):
    dataframe = dataframe.drop_duplicates().reset_index(drop=True)
    dataframe = dataframe.dropna().reset_index(drop=True)
    return dataframe


def compute_mahalanobis_distances(data, reference_data):
    covariance = np.cov(reference_data, rowvar=False)
    covariance_pm1 = np.linalg.matrix_power(covariance, -1)
    center_point = np.mean(reference_data, axis=0)
    distances = [(val - center_point).T.dot(covariance_pm1).dot(val -
                                                                center_point) for val in data]
    return np.array(distances)


def remove_outliers(data, reference_data, dataframe):
    distances = compute_mahalanobis_distances(data, reference_data)
    cutoff = chi2.ppf(0.95, reference_data.shape[1])
    outliers_indexes = np.where(distances > cutoff)
    data_without_outliers = np.delete(data, outliers_indexes, axis=0)
    return pd.DataFrame(data_without_outliers, columns=dataframe.columns)


folder = "gretel_77_s1"
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "../data", "data.csv")

df = load_data(data_path)
df = clean_data(df)

# Remove outliers from original data based on its own Mahalanobis distance
df = remove_outliers(df.to_numpy(), df.to_numpy(), df)

synthetic_data_path = os.path.join(
    current_dir, f"../data/{folder}", f"{folder}.csv.gz")
synthetic_df = load_data(synthetic_data_path, compression="gzip")
synthetic_df = clean_data(synthetic_df)

# Remove outliers from synthetic data based on cleaned original data's Mahalanobis distance
clean_synthetic_df = remove_outliers(
    synthetic_df.values, df.to_numpy(), synthetic_df)

print(f"Original shape: {clean_synthetic_df.shape}")
# Save the clean data
clean_synthetic_df.to_csv(os.path.join(
    current_dir, f"../data/{folder}", "synthetic_gretel.csv"), index=False)
