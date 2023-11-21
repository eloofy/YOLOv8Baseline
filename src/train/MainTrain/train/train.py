import os
import re

import pandas as pd
import yaml
from ultralytics import YOLO
import mlflow
from ultralytics import settings
from src.train.MLflowTracking.cust_mlflow import MLflowTracking
from configs.config import (
    home_path,
    RESULTS_DIR,
    MODEL_CONFIG_FILE,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
)
from src.train.Callbacks.callbacks import on_fit_epoch_end

settings.update({"mlflow": False, "runs_dir": RESULTS_DIR})


class YOLOTrainer:
    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        cfg_model: str,
        callbacks,
        sn4_nadir: bool = False,
    ):
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
        self.sn4_nadir = sn4_nadir

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

    def _val_metrics_nadir(self, path_save_res_nadirs: str):
        """
        Val SN4 model for each nadir.

        Args:
            path_save_res_nadirs: Path to the save nadirs result.
        """
        save_nadir_results_path = os.path.join(path_save_res_nadirs, "results.csv")
        list_data = sorted(
            os.listdir(
                os.path.join(
                    home_path,
                    "src/train/MainTrain/train/TrainDataConfigs/DataConfigsSN4Nadirs",
                )
            )
        )

        data_results = pd.DataFrame(
            columns=["nadir"] + list(self.model.trainer.metrics.keys())
        )
        for nadir_cfg in list_data[:2]:
            results = self.model.val(
                data=os.path.join(
                    home_path,
                    "src/train/MainTrain/train/TrainDataConfigs/DataConfigsSN4Nadirs",
                    nadir_cfg,
                ),
                split="test",
            )
            results_metrics = {
                name: round(results.results_dict[name], 4)
                for name in results.results_dict
            }
            results_metrics["nadir"] = re.findall("\d+", nadir_cfg).pop()
            data_results = pd.concat(
                [data_results, pd.DataFrame([results_metrics])], ignore_index=True
            )

        data_results.to_csv(save_nadir_results_path)
        self.mlflow_tracking.log_custom_artifact(
            save_nadir_results_path, "ResultsNadir"
        )

    def _train_yolo_model(self, model_config: dict):
        """
        Train a YOLO model based on the provided configuration.

        Args:
            model_config: YOLO model configuration as a dictionary.
        """
        run_name_mlflow = "YOLOv8_ver1_SN4"
        with mlflow.start_run(
            run_name=run_name_mlflow,
            description=self.mlflow_tracking.load_dataset_description(
                file_path=os.path.join(
                    home_path,
                    "src/train/MainTrain/train/TrainDataConfigs/data_SN4.yaml",
                )
            ),
        ):
            self.model.train(**model_config["training_params"])
            self.mlflow_tracking.set_all_params(self.model, model_config)

            if self.sn4_nadir:
                self._val_metrics_nadir(
                    os.path.join(
                        home_path,
                        "src/train/MainTrain/results/segment",
                        run_name_mlflow,
                    )
                )

    def _set_callbacks(self):
        """
        Set callbacks.
        """
        for callback_key in self.callbacks:
            self.model.add_callback(callback_key, self.callbacks[callback_key])

    def run_training(self):
        """
        Run the YOLO model training process.

        """
        self._set_callbacks()
        self._train_yolo_model(self.cfg_file)


def main():
    dict_callbacks = {"on_fit_epoch_end": on_fit_epoch_end}
    trainer = YOLOTrainer(
        MLFLOW_TRACKING_URI,
        EXPERIMENT_NAME,
        MODEL_CONFIG_FILE,
        dict_callbacks,
        sn4_nadir=True,
    )
    trainer.run_training()


if __name__ == "__main__":
    main()
