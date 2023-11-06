import os
import yaml
from ultralytics import YOLO
import mlflow
from ultralytics import settings
from src.train.MLflowTracking.cust_mlflow import MLflowTracking

from configs.config import home_path
settings.update({"mlflow": False})


class YOLOTrainer:
    def __init__(self, tracking_uri: str, experiment_name: str, cfg_model: str):
        """
        Initialize the YOLOTrainer.
        Args:
            tracking_uri: The MLflow tracking URI.
            experiment_name: Name of the MLflow experiment.
        """
        self.model = YOLO(
            self._load_model_config(cfg_model)["training_params"]["model"]
        )
        # self.model.callbacks = {...} Add custom func
        self.mlflow_tracking = MLflowTracking(tracking_uri, experiment_name)

    # @staticmethod
    # def _load_dataset_description(count_datasets):

    @staticmethod
    def _load_model_config(model_cfg_file: str):
        """
        Load the YOLO model configuration from a YAML file.

        Args:
            model_cfg_file: Path to the YOLO model configuration file.

        Returns:
            dict: Loaded model configuration as a dictionary.
        """
        with open(os.path.join(home_path, model_cfg_file), "r") as file:
            return yaml.safe_load(file)

    def _train_yolo_model(self, model_config: dict):
        """
        Train a YOLO model based on the provided configuration.

        Args:
            model_config: YOLO model configuration as a dictionary.
        """
        with mlflow.start_run(
            run_name="YOLOv8_ver",
            description=self.mlflow_tracking.load_description(
                file_path=os.path.join(home_path, "src/train/MainTrain/data.yaml")
            )
        ):
            self.model.train(**model_config["training_params"])
            self.mlflow_tracking.set_all_params(self.model, model_config)


    def run_training(self, model_cfg_file: str):
        """
        Run the YOLO model training process.

        Args:
            model_cfg_file: Path to the YOLO model configuration file.
        """
        model_config = self._load_model_config(model_cfg_file)
        self._train_yolo_model(model_config)


def main():
    model_cfg_path = "configs/model_cfg.yaml"

    trainer = YOLOTrainer("http://neuron:5000",
                          "YOLOv8BaseLineTrain",
                          model_cfg_path)

    trainer.run_training(model_cfg_path)


if __name__ == "__main__":
    main()
