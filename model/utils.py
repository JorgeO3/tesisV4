import os
import torch
import random
import numpy as np
import pandas as pd
from datetime import datetime
from .model_config import ModelConfig


def create_study_path(vars, study_dir_path):
    active_vars = "".join(vars)
    timestamp = datetime.now().strftime("%m-%d_%H:%M:%S")
    study_file = f"{timestamp}_{active_vars}.csv"
    return os.path.join(study_dir_path, study_file)


def global_seed():
    SEED = ModelConfig.SEED
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    pd.set_option("display.max_rows", None)

    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
