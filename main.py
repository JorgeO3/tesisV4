from abc import ABC, abstractmethod

from .model.model_config import ModelConfig
from .model.model_trainer import ModelTraining
from .model.model_validator import ModelValidation
from .model.model_predictor import ModelPrediction
from .model.model_optimizer import ModelOptimization


class ModelExecutionStrategy(ABC):
    @abstractmethod
    def execute(self):
        pass


class ModelRunner:
    def __init__(self, execution_strategy: ModelExecutionStrategy):
        self.execution_strategy = execution_strategy

    def run(self):
        self.execution_strategy.execute()


class ModelStrategyFactory:
    @staticmethod
    def get_model(mode, model_config):
        models = {
            "training": ModelTraining(model_config),
            "validation": ModelValidation(model_config),
            "optimization": ModelOptimization(model_config),
            "prediction": ModelPrediction(model_config),
        }
        return models.get(mode)


def main():
    ModelConfig.initialize()
    model_config = ModelConfig()

    mode = "training"

    execution_strategy = ModelStrategyFactory.get_model(mode, model_config)
    if execution_strategy is not None:
        model_runner = ModelRunner(execution_strategy)
        model_runner.run()
    else:
        print(f"Mode {mode} is not valid.")


if __name__ == "__main__":
    print("Hello world")
