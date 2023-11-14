import os
import yaml
from ultralytics import YOLO
import mlflow
from ultralytics import settings
from src.train.MLflowTracking.cust_mlflow import MLflowTracking
from configs.config import home_path
from src.train.Callbacks.callbacks import on_fit_epoch_end


MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "YOLOv8BaseLineTrain"
MODEL_CONFIG_FILE = "configs/model_cfg.yaml"
RESULTS_DIR = os.path.join(home_path, "src/train/MainTrain/results/")

settings.update({"mlflow": False, "runs_dir": RESULTS_DIR})


class YOLOTrainer:
    def __init__(self, tracking_uri: str, experiment_name: str, cfg_model: str, callbacks):
        """
        Initialize the YOLOTrainer.

        Args:
            tracking_uri: The MLflow tracking URI.
            experiment_name: Name of the MLflow experiment.
        """
        self.cfg_file = self._load_model_config(cfg_model)
        self.model = YOLO(self.cfg_file["training_params"]["model"])
        self.callbacks = callbacks
        self.mlflow_tracking = MLflowTracking(tracking_uri, experiment_name)

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
            run_name="YOLOv8_ver_SN",
            description=self.mlflow_tracking.load_dataset_description(
                file_path=os.path.join(home_path, "src/train/MainTrain/train/TrainDataConfigs/data_SN4.yaml")
            )
        ):
            self.model.train(**model_config["training_params"])
            self.mlflow_tracking.set_all_params(self.model, model_config)

    def _set_callbacks(self):
        for callback_key in self.callbacks:
            self.model.add_callback(callback_key, self.callbacks[callback_key])

    def run_training(self):
        """
        Run the YOLO model training process.

        """
        self._set_callbacks()
        self._train_yolo_model(self.cfg_file)


def main():
    dict_callbacks = {
        "on_fit_epoch_end": on_fit_epoch_end
    }
    trainer = YOLOTrainer(MLFLOW_TRACKING_URI, EXPERIMENT_NAME, MODEL_CONFIG_FILE, dict_callbacks)

    trainer.run_training()


if __name__ == "__main__":
    main()