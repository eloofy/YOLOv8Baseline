from ultralytics.models.yolo.segment.train import SegmentationTrainer
import mlflow
from src.train.MLflowTracking.cust_mlflow import clean_metric_names


def on_fit_epoch_end(predictor: SegmentationTrainer):
    """
    Callback function to log training and validation metrics to MLflow at the end of each epoch.

    Parameters:
    - predictor (Any): The predictor object containing training and validation metrics.

    Returns:
    - None
    """
    if predictor.epoch == 0:
        pass
    
    train_metrics_losses = clean_metric_names(predictor.metrics)
    val_losses = clean_metric_names(
        dict(
            zip(list(predictor.loss_names), predictor.tloss.tolist())
        )
    )
    mlflow.log_metrics(
        {**train_metrics_losses, **val_losses},
        predictor.epoch
    )
