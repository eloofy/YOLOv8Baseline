import os

import torch
from PIL import Image
from ultralytics import YOLO
from configs.config import home_path

BEST_MODEL_PATH = os.path.join(home_path, "src/train/runs/segment/yolov8n_custom7/weights/best.pt")


def predict_yolov8(image: Image.Image, model_path: str) -> torch.Tensor:
    """
    Perform YOLOv8 object detection on the given image using the specified model.

    Args:
        image: The input image to perform object detection on.
        model_path: The file path to the YOLOv8 model weights.

    Returns:
        torch.Tensor: Detection mask.
    """
    model = YOLO(model_path)
    results = model(image)

    return results[0].masks.data

