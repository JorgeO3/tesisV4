import sys
import argparse
from abc import ABC, abstractmethod

from model.model_config import ModelConfig
from model.model_trainer import ModelTraining
from model.model_predictor import ModelPrediction
from model.model_optimizer import ModelOptimization


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
            "optimization": ModelOptimization(model_config),
            "prediction": ModelPrediction(model_config),
        }
        return models.get(mode)


def cli():
    parser = argparse.ArgumentParser(description="PyTorch Model Execution")
    parser.add_argument("mode", choices=[
        "training", "optimization", "prediction"], help="Model mode")
    parser.add_argument("--ts", action="store_true",
                        help="Enable ts response variable")
    parser.add_argument("--wvp", action="store_true",
                        help="Enable wvp response variable")
    parser.add_argument("--e", action="store_true",
                        help="Enable e response variable")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU execution")
    parser.add_argument("--threads", type=int, help="Enable number of threads")
    parser.add_argument("--layers", type=int,
                        help="Set number of model layers")
    parser.add_argument("--folder", type=str, help="Specify data input")
    return parser.parse_args()


def main():
    resp_vars = []
    args = cli()
    gpu = False
    threads = 1
    layers = 5
    folder = ""

    if args.ts:
        resp_vars.append("TS")
    if args.wvp:
        resp_vars.append("WVP")
    if args.e:
        resp_vars.append("%E")
    if args.gpu:
        gpu = True
    if args.threads is not None:
        threads = args.threads
    if args.layers is not None:
        layers = args.layers
    if args.folder:
        folder = args.folder

    if len(resp_vars) == 0:
        print("Please include at least one response variable.")
        return

    if not folder:
        print("Folder is mandatory. Please provide a folder path.")
        return

    ModelConfig.initialize()
    model_config = ModelConfig(folder)
    model_config.set_active_resp_vars(resp_vars)
    model_config.enable_gpu(gpu)
    model_config.set_num_threads(threads)
    model_config.set_num_layers(layers)

    mode = args.mode

    execution_strategy = ModelStrategyFactory.get_model(mode, model_config)
    if execution_strategy is not None:
        model_runner = ModelRunner(execution_strategy)
        model_runner.run()
    else:
        print(f"Mode {mode} is not valid.")


if __name__ == "__main__":
    main()
