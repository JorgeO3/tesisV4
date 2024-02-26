import pandas as pd

df = pd.read_csv("./data/cleaned_data.csv")

train_data = df.sample(frac=0.875, random_state=0)
test_data = df.drop(train_data.index).reset_index(drop=True)

train_data.to_csv("./etc/train_data.csv", index=False)
test_data.to_csv("./etc/test_data.csv", index=False)
