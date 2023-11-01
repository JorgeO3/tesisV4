import yaml
import argparse
from .model_config import ModelConfig


def process_args(args: argparse.Namespace):
    resp_args = ["ts", "wvp", "e"]
    resp_vars = [arg.upper() for arg in resp_args if getattr(args, arg)]

    gpu = getattr(args, "gpu", False)
    threads = getattr(args, "threads", 1)
    layers = getattr(args, "layers", 5)

    if not resp_vars:
        print("Please include at least one response variable.")
        exit(1)

    return (
        args.mode,
        resp_vars,
        gpu,
        threads,
        layers,
    )


class Cli:
    def __init__(self) -> None:
        self.commands_path = ModelConfig.COMMANDS_FILE

    def parse_args(self):
        args = self.generate_args()
        return process_args(args)

    def generate_args(self):
        type_dict = {
            "int": int,
            "str": str,
        }

        with open(self.commands_path, "r") as file:
            commands = yaml.load(file, Loader=yaml.FullLoader)["commands"]

        parser = argparse.ArgumentParser(description="PyTorch Model Execution")

        # Generate arguments from yaml file
        for cmd in commands:
            arg_dic = {
                k: (type_dict[v] if k == "type" else v)
                for k, v in cmd.items()
                if k != "command" and v is not None
            }
            parser.add_argument(cmd["command"], **arg_dic)

        return parser.parse_args()
