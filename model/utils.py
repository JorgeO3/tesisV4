import os
import pandas as pd
from functools import reduce
from datetime import datetime


def create_study_path(vars, study_dir_path):
	active_vars = "".join(vars)
	timestamp = datetime.now().strftime("%m-%d_%H:%M:%S")
	study_file = f"{timestamp}_{active_vars}.csv"
	return os.path.join(study_dir_path, study_file)


def merge_data(*paths):
	data_frames = [pd.read_csv(file_path) for file_path in paths]
	data = reduce(lambda df1, df2: pd.concat([df1, df2]), data_frames, pd.DataFrame())
	data = data.sample(frac=1).reset_index(drop=True)
	return data.values
