from .model_config import ModelConfig


class Model:
    def __init__(self, config: ModelConfig, core, params) -> None:
        self.config = config
        self.core = core
        self.params = params

    def run(self) -> float:
        pass
