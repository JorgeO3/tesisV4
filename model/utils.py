import os
import random
import numpy as np
import torch as th
from datetime import datetime


def create_study_path(vars, study_dir_path):
    active_vars = "".join(vars)
    timestamp = datetime.now().strftime("%m-%d_%H:%M:%S")
    study_file = f"{timestamp}_{active_vars}.csv"
    return os.path.join(study_dir_path, study_file)


def seed_worker(worker_id):
    worker_seed = th.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
