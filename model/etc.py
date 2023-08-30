import numpy as np
import pandas as pd

from functools import reduce
from .model_config import ModelConfig


def data_processor(data: np.ndarray):
    resp_vars = ModelConfig.RESPONSE_VARS
    input_vars = ModelConfig.INPUT_VARS
    df = pd.DataFrame(data, columns=[*input_vars, *resp_vars])

    shuffled_data = df.sample(frac=1).reset_index(drop=True)

    # shuffled_data = shuffled_data[0:300]

    # Define the input and response variables
    X = shuffled_data.drop(shuffled_data.columns[[11, 12, 13]], axis=1).values
    y = shuffled_data["WVP"].values.reshape(-1, 1)

    return X, y


def concatenate_data(paths):
    data_frames = [pd.read_csv(file_path) for file_path in paths]
    concatenated_data = reduce(lambda df1, df2: pd.concat(
        [df1, df2]), data_frames, pd.DataFrame())
    return concatenated_data.values
