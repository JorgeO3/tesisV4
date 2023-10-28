import yaml
import argparse
from .model_config import ModelConfig


class Cli:
    def __init__(self) -> None:
        self.commands_path = ModelConfig.COMANDS_FILE

    def parse_args(self):
        args = self.generate_args()
        return self.process_args(args)

    def generate_args(self):
        # Diccionario de tipos predefinidos
        type_dict = {
            "int": int,
            "str": str,
        }
        # Usar with para la lectura de archivos
        with open(self.commands_path, "r") as file:
            commands = yaml.load(file, Loader=yaml.FullLoader)["commands"]

        # Start parser
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

    def process_args(self, args: argparse.Namespace):
        # Lista de argumentos de los cuales recoger las respuestas
        resp_args = ["ts", "wvp", "e"]
        # Construye resp_vars basado en los argumentos proporcionados
        resp_vars = [arg.upper() for arg in resp_args if getattr(args, arg)]

        # Atributos predeterminados
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
