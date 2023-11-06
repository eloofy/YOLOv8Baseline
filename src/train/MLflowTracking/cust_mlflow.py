from ultralytics import settings
import mlflow
import os
import yaml
from configs.config import home_path

from ultralytics.utils.callbacks.base import on_fit_epoch_end

settings.update({'mlflow': False})


def change_metr_repr(metrics):
    clear_metrics = {key.replace('(', '').replace(')', ''): value for key, value in metrics.items()}

    return clear_metrics


class MLflowTracking:
    def __init__(self, tracking_uri, experiment_name):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._setup_mlflow_tracking()

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
    def _log_params(model_cfg):
        mlflow.log_params(
            model_cfg["training_params"]
        )

    @staticmethod
    def _log_metrics(model):
        model.trainer.metrics = change_metr_repr(metrics=model.trainer.metrics.copy())
        loss_metrics = dict(item for item in zip(list(model.trainer.loss_names), model.trainer.loss_items.tolist()))

        mlflow.log_metrics(
            {**model.trainer.metrics, **loss_metrics}
        )

    @staticmethod
    def load_description(file_path):
        with open(file_path, 'r') as data_file:
            print(data_file)
            dataset_name = yaml.safe_load(data_file)["path"].split(os.path.sep)[-4]

        return (f'Datasets:\n'
                f'{dataset_name}')

    def set_all_params(self, model, model_cfg):
        self._log_params(model_cfg)
        self._log_metrics(model)
        mlflow.pyfunc.log_model(artifact_path="model",
                                artifacts={'model_path': str(model.trainer.save_dir)},
                                python_model=mlflow.pyfunc.PythonModel())
