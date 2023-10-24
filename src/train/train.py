from ultralytics import YOLO, settings
from configs.config import seed, train_parameters


def train_yolov8():
    """
        Train a YOLOv8 model using the specified training parameters.
    """
    model = YOLO(train_parameters['model_path'])

    model.train(
        data=train_parameters['data_config'],
        imgsz=train_parameters['img_size'],
        epochs=train_parameters['epochs'],
        patience=train_parameters['patience'],
        batch=train_parameters['batch_size'],
        device=train_parameters['device'],
        workers=train_parameters['workers'],
        optimizer=train_parameters['optimizer'],
        seed=seed,
        cos_lr=train_parameters['cos_lr'],
        lr0=train_parameters['learning_rate'],
        name=train_parameters['model_name'],
        show_labels=train_parameters['show_labels']
    )


if __name__ == "__main__":
    train_yolov8()
