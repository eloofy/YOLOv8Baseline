import os
from configs.config import home_path
import yaml

from model import CustomYOLO


class CustomYOLOv8:
    def __init__(self, model_cfg_file: str):
        self.yaml_inputs = self.load_inputs(model_cfg_file)
        self.model = self._load_model()

    @staticmethod
    def load_inputs(model_cfg_file: str):
        with open(os.path.join(home_path, model_cfg_file), "r") as file:
            return yaml.safe_load(file)

    def _load_model(self):
        return CustomYOLO(self.yaml_inputs["training_params"]["model"])

    def train(self):
        self.model.train(**self.yaml_inputs)


def main():
    model = CustomYOLOv8("configs/model_cfg.yaml")
    model.train()


if __name__ == "__main__":
    main()
