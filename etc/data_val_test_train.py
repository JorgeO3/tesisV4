import os

import pandas as pd

folder = "gretel_74_v2_s1"

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "../data", "clean_data.csv")

data = pd.read_csv(data_path).reset_index(drop=True)
train_data = data.sample(frac=0.92, random_state=0)
test_data = data.drop(train_data.index)

train_data.to_csv(os.path.join(current_dir, f"../data/{folder}",
                  "train_data.csv"), index=False)
test_data.to_csv(os.path.join(current_dir, f"../data/{folder}",
                 "test_data.csv"), index=False)
