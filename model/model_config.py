import os
import random

import torch
import numpy as np
import pandas as pd


class ModelConfig:
    SEED = 42
    RESPONSE_VARS = ["TS", "WVP", "%E"]
    INPUT_VARS = [
        "%Chi",
        "%Gel",
        "%Gly",
        "%Pec",
        "%Sta",
        "%Oil",
        "T(°C)",
        "%RH",
        "t(h)",
    ]

    N_TRIALS = os.environ.get("N_TRIALS")
    DEBUG = True if os.environ.get("DEBUG") == "1" else False
    STOPPING = True if os.environ.get("STOPPING") == "1" else False
    SAVE_MODEL = True if os.environ.get("SAVE_MODEL") == "1" else False

    STUDY_DIR = os.environ.get("STUDY_DIR")
    SCALER_PATH = os.environ.get("SCALER_PATH")
    COMMANDS_FILE = os.environ.get("COMMANDS_FILE")
    TEST_DATA_PATH = os.environ.get("TEST_DATA_PATH")
    TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA_PATH")

    def __init__(self) -> None:
        self.NUM_LAYERS = None
        self.DEVICE = None
        self.ACTIVE_RESPONSE_VARS = None
        self.NUM_RESPONSE_VARS = None
        self.IND_RESPONSE_VARS = None

        #TODO: Fix the problem of seeds, it is not working for reproducibility
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed(self.SEED)
        torch.cuda.manual_seed_all(self.SEED)
        pd.set_option("display.max_rows", None)

        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def set_active_resp_vars(self, active_resp_vars: list[str]):
        """
        Sets the active response variables of the model.
        """
        self.ACTIVE_RESPONSE_VARS = active_resp_vars
        self.IND_RESPONSE_VARS = []
        #TODO: Fix this shit
        if "TS" in active_resp_vars:
            self.IND_RESPONSE_VARS.append(9)
        if "WVP" in active_resp_vars:
            self.IND_RESPONSE_VARS.append(10)
        if "E" in active_resp_vars:
            self.IND_RESPONSE_VARS.append(11)
        

    def enable_gpu(self, gpu: bool = False):
        """
        Enables GPU execution.
        """
        if gpu:
            self.DEVICE = torch.device("cuda:0")

    def set_num_threads(self, num_threads):
        """
        Sets the number of CPU threads used.
        """
        torch.set_num_threads(num_threads)

    def set_num_layers(self, num_layers: int):
        """
        Sets the number of layers of the model.
        """
        self.NUM_LAYERS = num_layers
