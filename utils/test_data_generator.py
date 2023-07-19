import os

import numpy as np
import pandas as pd
from scipy.stats import chi2

current_dir = os.path.dirname(os.path.abspath(__file__))
mohalanobis_data_path = os.path.join(current_dir, "../data", "mohalanobis.csv")

mohalanobis_data = pd.read_csv(mohalanobis_data_path)
mohalanobis_data = mohalanobis_data.sample(frac=1).reset_index(drop=True)

test_data = mohalanobis_data[80:]
mohalanobis_data = mohalanobis_data[:80]

test_data.to_csv(os.path.join(current_dir, "../data", "test_data.csv"))
mohalanobis_data.to_csv(os.path.join(current_dir, "../data", "train_data.csv"))