import os
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Chi: 3, Gel: 3, Gly, 3, Pec: 3, Sta: 3, Oil: 3, T(ºC): 0; %RH: 0, t(h): 0, TS: 2, WVP: 2, E: 2
dv = [3, 3, 3, 3, 3, 3, 0, 0, 0, 2, 2, 2]

# Set paths
raw_data_path = os.environ.get("RAW_DATA_FILE")
cleaned_data_path = os.environ.get("CLEANED_FILE")

# Read raw data
df = pd.read_csv(raw_data_path)


def round_colums(data, dv=dv):
    columns = data.columns
    for i, col in enumerate(columns):
        formato = "{:." + str(dv[i]) + "f}"
        data[col] = data[col].round(dv[i])
        data[col] = data[col].apply(lambda x: formato.format(x))
    return data


# Drop duplicated rows
df = df.drop_duplicates()

# Round columns
df = round_colums(df)

# Impute missing values
imp = IterativeImputer(
    estimator=RandomForestRegressor(random_state=42), max_iter=10, random_state=0
)
data = imp.fit_transform(df)

# Round columns and save clean data
df = pd.DataFrame(data, columns=df.columns)
df = round_colums(df)
df.to_csv(cleaned_data_path, index=False)
