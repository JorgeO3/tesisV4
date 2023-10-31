from model.cli import Cli
from model.model_config import ModelConfig
from model.model_trainer import ModelTraining
from model.model_predictor import ModelPrediction
from model.model_optimizer import ModelOptimization
from model.model_execution_strategy import ModelExecutionStrategy


class ModelRunner:
    def __init__(self, execution_strategy: ModelExecutionStrategy):
        self.execution_strategy = execution_strategy

    def run(self):
        self.execution_strategy.execute()


class ModelStrategyFactory:
    @staticmethod
    def get_model(mode, config: ModelConfig):
        models = {
            "training": ModelTraining(config),
            "optimization": ModelOptimization(config),
            "prediction": ModelPrediction(config),
        }
        return models.get(mode)


def main():
    model_config = ModelConfig()
    mode, resp_vars, gpu, threads, layers = Cli().parse_args()

    model_config.set_active_resp_vars(resp_vars)
    model_config.enable_gpu(gpu)
    model_config.set_num_threads(threads)
    model_config.set_num_layers(layers)

    execution_strategy = ModelStrategyFactory.get_model(mode, model_config)
    if execution_strategy is not None:
        model_runner = ModelRunner(execution_strategy)
        model_runner.run()
    else:
        print(f"Mode {mode} is not valid.")


if __name__ == "__main__":
    main()
