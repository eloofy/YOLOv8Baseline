from pathlib import Path
import os

home_path = Path(__file__).parent.parent

EXPERIMENT_NAME = "YOLOv8BaseLineTrain"
MODEL_CONFIG_FILE = "configs/model_cfg.yaml"
RESULTS_DIR = os.path.join(home_path, "src/train/MainTrain/results/")
MLFLOW_TRACKING_URI = "http://mlflow:5000"
