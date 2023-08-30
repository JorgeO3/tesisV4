import os

import numpy as np
import pandas as pd
from scipy.stats import chi2

# Path of the data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "../data", "data.csv")
synthetic_data_path = os.path.join(
    current_dir, "../data", "data.csv.gz")

synthetic_data = pd.read_csv(synthetic_data_path, compression="gzip")
# synthetic_data = synthetic_data.drop(synthetic_data.columns[0], axis=1)

data = pd.read_csv(data_path)
df = data.to_numpy()

synthetic_df = synthetic_data.values

# Covarance matrix
covariance = np.cov(df, rowvar=False)

# Covariance matrix power of -1
covariance_pm1 = np.linalg.matrix_power(covariance, -1)

# Center point
center_point = np.mean(df, axis=0)

# Distances between center and point
distances = []

for _, val in enumerate(synthetic_df):
    p1 = val
    p2 = center_point
    distance = (p1 - p2).T.dot(covariance_pm1).dot(p1 - p2)
    distances.append(distance)

distances = np.array(distances)

# Cutoff (threshold) value from Chi-Sqaure Distribution for detecting outliers
cutoff = chi2.ppf(0.95, df.shape[1])

# Index of outliners
outliers_indexes = np.where(distances > cutoff)

# Delete outliers from the data
data_without_outliers = np.delete(synthetic_df, outliers_indexes, axis=0)

# Transform into df
clean_data = pd.DataFrame(data_without_outliers, columns=data.columns)

# Save the clean data
clean_data.to_csv(os.path.join(current_dir, "../data",
                  "synthetic_data.csv"), index=False)
