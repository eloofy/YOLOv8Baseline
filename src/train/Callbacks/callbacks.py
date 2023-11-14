import mlflow
from src.train.MLflowTracking.cust_mlflow import clean_metric_names


def on_fit_epoch_end(predictor):
    train_metrics_losses = clean_metric_names(predictor.metrics)
    val_losses = clean_metric_names(
        dict(
            zip(list(predictor.loss_names), predictor.validator.loss.tolist())
        )
    )
    mlflow.log_metrics(
        {**train_metrics_losses, **val_losses},
        predictor.epoch
    )
