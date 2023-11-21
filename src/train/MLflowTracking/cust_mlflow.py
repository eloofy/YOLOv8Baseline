import os
import mlflow
import yaml
from configs.config import home_path
from typing import Any, Dict, Union
from ultralytics import YOLO


def clean_metric_names(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean metric names by removing parentheses.

    Args:
        metrics: Dictionary of metrics with possibly formatted names.

    Returns:
        Dict: Dictionary of metrics with cleaned names.
    """
    return {
        key.replace("(", "").replace(")", ""): value for key, value in metrics.items()
    }


class MLflowTracking:
    def __init__(self, tracking_uri: str, experiment_name: str):
        """
        Initialize an MLflowTracking instance.

        Args:
            tracking_uri: The MLflow tracking URI.
            experiment_name: Name of the MLflow experiment.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.setup_mlflow_tracking()

    def setup_mlflow_tracking(self) -> None:
        """
        Set up MLflow tracking using the provided tracking URI and experiment name.
        If the experiment doesn't exist, it creates one.
        """
        mlflow.set_tracking_uri(self.tracking_uri)

        if mlflow.get_experiment_by_name(self.experiment_name) is None:
            mlflow.create_experiment(self.experiment_name)

        mlflow.set_experiment(self.experiment_name)

    @staticmethod
    def log_training_params(model_cfg: Dict[str, Union[str, Dict[str, Any]]]) -> None:
        """
        Log training parameters to MLflow.

        Args:
            model_cfg: YOLO model configuration.
        """
        mlflow.log_params(model_cfg["training_params"])

    @staticmethod
    def log_training_metrics(model: YOLO) -> None:
        """
        Log training metrics to MLflow.

        Args:
            model: YOLO model.
        """
        metrics = {
            "best_" + name: value.item()
            for name, value in clean_metric_names(model.trainer.metrics.copy()).items()
        }

        mlflow.log_metrics({**metrics})

    @staticmethod
    def log_trained_model(model: YOLO) -> None:
        """
        Log the trained model to MLflow.

        Args:
            model (Any): YOLO model.
        """
        mlflow.pyfunc.log_model(
            artifact_path="model",
            artifacts={"model_path": str(model.trainer.save_dir)},
            python_model=mlflow.pyfunc.PythonModel(),
        )

    @staticmethod
    def load_dataset_description(file_path: str) -> str:
        """
        Load and return a description of the dataset from a file.

        Args:
            file_path (str): Path to the dataset description file.

        Returns:
            str: Description of the dataset.
        """
        with open(file_path, "r") as data_file:
            dataset_name = yaml.safe_load(data_file)["path"].split(os.path.sep)[-3]

        return f"Datasets:\n{dataset_name}"

    @staticmethod
    def log_custom_artifact(artifact_path, save_path):
        """
        Log some artifact.

        Args:
            artifact_path: Path artifact.
            save_path: Path to log.
        """

        mlflow.log_artifact(artifact_path, save_path)

    def set_all_params(
        self, model: YOLO, model_cfg: Dict[str, Union[str, Dict[str, Any]]]
    ) -> None:
        """
        Log all relevant parameters and artifacts to MLflow.

        Args:
            model: YOLO model.
            model_cfg (Dict[str, Union[str, Dict[str, Any]]): YOLO model configuration.
        """
        self.log_training_params(model_cfg)
        self.log_training_metrics(model)
        self.log_trained_model(model)
        mlflow.log_artifact(os.path.join(home_path, "configs/model_cfg.yaml"), "config")
