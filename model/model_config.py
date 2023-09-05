import os
import random

import torch
import numpy as np
import pandas as pd
from functools import reduce


class ModelConfig:
    """
    Configuration class for the model.

    This class provides configurations and file paths used in the model.
    It contains static variables for the random seed, execution device,
    response variables, input variables, scaler file paths, study results,
    test data, data used for the model, manual model, and synthetic data.

    Attributes:
        SEED (int): Seed for random number generation.
        CURRENT_DIR (str): Current directory of the file.
        DEVICE (torch.device): Execution device (GPU or CPU).
        RESPONSE_VARS (List[str]): Response variables of the model.
        INPUT_VARS (List[str]): Input variables of the model.
        SCALER_Y_PATH (str): File path for the scaler of the response variable.
        SCALER_X_PATH (str): File path for the scalers of the input variables.
        STUDY_CSV_PATH (str): File path for saving the study results as CSV.
        TEST_DATA_PATH (str): File path for the test data for the model.
        DATA_PATH (str): File path for the data used for the model.
        MANUAL_MODEL_PATH (str): File path for saving the manual model.
        SYNTHETIC_DATA_PATH (str): File path for the synthetic data.

    Methods:
        initialize(): Initializes the random number generation seed and sets common options.
    """
    SEED = 42
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEVICE = torch.device("cpu")
    RESPONSE_VARS = ["TS", "WVP", "%E"]
    INPUT_VARS = ["%Chi", "%Gel", "%Gly", "%Pec", "%Sta",
                  "%Oil", "%W", "%AA", "T(°C)", "%RH", "t(h)"]
    ACTIVE_RESPONSE_VARS = []
    NUM_LAYERS = None

    def __init__(self, folder: str):
        self.FOLDER = folder
        self.SCALER_PATH = os.path.join(
            self.CURRENT_DIR, "../data", "scaler.pkl")
        self.STUDY_CSV_PATH = os.path.join(
            self.CURRENT_DIR, f"../results/{self.FOLDER}", "study.csv")
        self.TEST_DATA_PATH = os.path.join(
            self.CURRENT_DIR, f"../data/{self.FOLDER}", "test_data.csv")
        self.DATA_PATH = os.path.join(
            self.CURRENT_DIR, f"../data/{self.FOLDER}", "train_data.csv")
        self.MANUAL_MODEL_PATH = os.path.join(
            self.CURRENT_DIR, "../results", "manual_model.pt")
        self.SYNTHETIC_DATA_PATH = os.path.join(
            self.CURRENT_DIR, f"../data/{self.FOLDER}", "synthetic_gretel.csv")
        self.SAVE_SCALER = False
        self.DEBUG = False

    @staticmethod
    def initialize():
        """
        Initializes the random number generation seed and sets common options.

        This static method is called before running the model to set the random
        number generation seed and configure common options like the number of
        CPU threads used and the pandas display setting.
        """
        random.seed(ModelConfig.SEED)
        np.random.seed(ModelConfig.SEED)
        torch.manual_seed(ModelConfig.SEED)
        pd.set_option("display.max_rows", None)

    def set_active_resp_vars(self, active_resp_vars: list[str] = ["TS", "WVP", "%E"]):
        """
        Sets the active response variables of the model.
        """
        self.ACTIVE_RESPONSE_VARS = active_resp_vars

    def enable_gpu(self, gpu: bool = False):
        """
        Enables GPU execution.
        """
        if gpu:
            self.DEVICE = torch.device("cuda:0")

    def set_num_threads(self, num_threads: int):
        """
        Sets the number of CPU threads used.
        """
        torch.set_num_threads(num_threads)

    def set_num_layers(self, num_layers: int):
        """
        Sets the number of layers of the model.
        """
        self.NUM_LAYERS = num_layers

    def create_path(self, file_name: str) -> str:
        """
        Creates a file path for the given file name.

        Args:
            file_name (str): Name of the file.

        Returns:
            str: File path for the given file name.
        """
        return os.path.join(self.CURRENT_DIR, f"../results/{self.FOLDER}", file_name)

    def data_processor(self, data: np.ndarray):
        resp_vars = self.RESPONSE_VARS
        input_vars = self.INPUT_VARS
        active = self.ACTIVE_RESPONSE_VARS
        df = pd.DataFrame(data, columns=[*input_vars, *resp_vars])

        shuffled_data = df.sample(frac=1).reset_index(drop=True)

        # Define the input and response variables
        X = shuffled_data.drop(columns=resp_vars).values
        y = shuffled_data[active].values.reshape(-1, len(active))

        return X, y

    def concatenate_data(self, paths):
        data_frames = [pd.read_csv(file_path) for file_path in paths]
        concatenated_data = reduce(lambda df1, df2: pd.concat(
            [df1, df2]), data_frames, pd.DataFrame())
        return concatenated_data.values
