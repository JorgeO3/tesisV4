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

	DEBUG = True if os.environ.get("DEBUG") >= "1" else False
	SAVE_SCALER = True if os.environ.get("SAVE_SCALER") >= "1" else False
	N_TRIALS = os.environ.get("N_TRIALS")
	STOPPING = os.environ.get("STOPPING")

	DATA_PATH = os.environ.get("DATA_PATH")
	STUDY_DIR = os.environ.get("STUDY_DIR")
	SCALER_PATH = os.environ.get("SCALER_PATH")
	COMMANDS_FILE = os.environ.get("COMMANDS_FILE")
	TEST_DATA_PATH = os.environ.get("TEST_DATA_PATH")
	SYNTHETIC_DATA_PATH = os.environ.get("SYNTHETIC_DATA_PATH")

	def __init__(self) -> None:
		random.seed(self.SEED)
		np.random.seed(self.SEED)
		torch.manual_seed(self.SEED)
		pd.set_option("display.max_rows", None)

	def set_active_resp_vars(self, active_resp_vars: list[str]):
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