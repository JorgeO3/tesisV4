import os
import pandas as pd

# This is the data that we want to split
data_path = os.environ.get("DATA_PATH")
synthetic_data_path = os.environ.get("SYNTHETIC_DATA_PATH")

# This is the name of the files that we want to save
train_data_path = os.environ.get("TRAIN_FILE_PATH")
test_data_path = os.environ.get("TEST_FILE_PATH")

# Split data
data = pd.read_csv(data_path)
synthetic = pd.read_csv(synthetic_data_path)

train_data = data.sample(frac=0.875, random_state=0)
test_data = data.drop(train_data.index).reset_index(drop=True)
train_data = pd.concat([train_data, synthetic]).sample(frac=1).reset_index(drop=True)

# Save data
train_data.to_csv(train_data_path, index=False)
test_data.to_csv(test_data_path, index=False)
