import os
import yaml
from ultralytics import YOLO
import mlflow

from configs.config import home_path


class YOLOTrainer:
    def __init__(self, tracking_uri: str, experiment_name: str):
        """
        Initialize the YOLOTrainer.
        Args:
            tracking_uri: The MLflow tracking URI.
            experiment_name: Name of the MLflow experiment.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

    @staticmethod
    def _setup_mlflow_tracking(self):
        """
        Set up MLflow tracking using the provided tracking URI and experiment name.
        If the experiment doesn't exist, it creates one.
        """
        mlflow.set_tracking_uri(self.tracking_uri)

        if mlflow.get_experiment_by_name(self.experiment_name) is None:
            mlflow.create_experiment(self.experiment_name)

        mlflow.set_experiment(self.experiment_name)

    @staticmethod
    def _load_model_config(model_cfg_file: str):
        """
        Load the YOLO model configuration from a YAML file.

        Args:
            model_cfg_file (str): Path to the YOLO model configuration file.

        Returns:
            dict: Loaded model configuration as a dictionary.
        """
        with open(os.path.join(home_path, model_cfg_file), "r") as file:
            return yaml.safe_load(file)

    @staticmethod
    def _train_yolo_model(model_config: dict):
        """
        Train a YOLO model based on the provided configuration.

        Args:
            model_config (dict): YOLO model configuration as a dictionary.
        """
        with mlflow.start_run(run_name="YOLOv8_ver"):
            model = YOLO(model_config['training_params']['model'])
            model.train(**model_config['training_params'])

    def run_training(self, model_cfg_file: str):
        """
        Run the YOLO model training process.

        Args:
            model_cfg_file (str): Path to the YOLO model configuration file.
        """
        self._setup_mlflow_tracking(self)
        model_config = self._load_model_config(model_cfg_file)
        self._train_yolo_model(model_config)


def main():
    try:
        model_cfg_path = 'configs/model_cfg.yaml'

        trainer = YOLOTrainer(
            os.environ['MLFLOW_TRACKING_URI'],
            os.environ['MLFLOW_EXPERIMENT_NAME']
        )

        trainer.run_training(model_cfg_path)
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()