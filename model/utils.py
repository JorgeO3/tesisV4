import os
from datetime import datetime


def create_study_path(vars, study_dir_path):
    active_vars = "".join(vars)
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    study_file = f"{timestamp}_{active_vars}.csv"
    return os.path.join(study_dir_path, study_file)
