import os
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest

# Chi: 3, Gel: 3, Gly, 3, Pec: 3, Sta: 3, Oil: 3, T(ºC): 0; %RH: 0, t(h): 0, TS: 2, WVP: 2, E: 2
dv = [3, 3, 3, 3, 3, 3, 0, 0, 0, 2, 2, 2]

# Set paths
raw_data_file = os.environ.get("RAW_DATA_FILE")
cleaned_data_file = os.environ.get("CLEANED_FILE")


def clean_data(dataframe: pd.DataFrame):
    dataframe = dataframe.drop_duplicates().reset_index(drop=True)
    dataframe = dataframe.dropna().reset_index(drop=True)
    return dataframe


df = pd.read_csv(raw_data_file)
df = clean_data(df)
