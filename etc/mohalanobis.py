import os

import numpy as np
import pandas as pd
from scipy.stats import chi2

# Path of the data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "../data", "data.csv")

df = pd.read_csv(data_path)
data = df.to_numpy()


def mohalanobis(data):
    # Covarance matrix
    covariance = np.cov(data, rowvar=False)

    # Covariance matrix power of -1
    covariance_pm1 = np.linalg.matrix_power(covariance, -1)

    # Center point
    center_point = np.mean(data, axis=0)

    # Distances between center and point
    distances = []

    for _, val in enumerate(data):
        p1 = val
        p2 = center_point
        distance = (p1 - p2).T.dot(covariance_pm1).dot(p1 - p2)
        distances.append(distance)

    distances = np.array(distances)

    # Cutoff (threshold) value from Chi-Sqaure Distribution for detecting outliers
    cutoff = chi2.ppf(0.95, data.shape[1])

    # Index of outliners
    outliers_indexes = np.where(distances > cutoff)

    # Delete outliers from the data
    data_without_outliers = np.delete(data, outliers_indexes, axis=0)

    # Transform into data
    clean_data = pd.DataFrame(data_without_outliers, columns=df.columns)

    return clean_data


firt = mohalanobis(data)
second = mohalanobis(firt.to_numpy())

# Save the clean data
second.to_csv(os.path.join(current_dir, "../data",
                           "gretel_data.csv"), index=False)
