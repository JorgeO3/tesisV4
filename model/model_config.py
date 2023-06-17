import os
import random

import torch
import numpy as np
import pandas as pd


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
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    RESPONSE_VARS = ["TS", "WVP", "%E"]
    INPUT_VARS = ["%Chi", "%Gel", "%Gly", "%Pec", "%Sta", "%Oil", "%W", "%AA", "T(°C)", "%RH", "t(h)"]

    SCALER_Y_PATH = os.path.join(CURRENT_DIR, "data", "scaler_y.pkl")
    SCALER_X_PATH = os.path.join(CURRENT_DIR, "data", "scaler_x.pkl")
    STUDY_CSV_PATH = os.path.join(CURRENT_DIR, "results", "study.csv")
    TEST_DATA_PATH = os.path.join(CURRENT_DIR, "data", "test_model.csv")
    DATA_PATH = os.path.join(CURRENT_DIR, "data", "mohalanobis_data.csv")
    MANUAL_MODEL_PATH = os.path.join(CURRENT_DIR, "results", "manual_model.pt")
    SYNTHETIC_DATA_PATH = os.path.join(CURRENT_DIR, "data", "mohalanobis3.csv")

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
        torch.set_num_threads(2)
        pd.set_option("display.max_rows", None)
