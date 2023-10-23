import os
import glob
import shutil

from ultralytics.data.converter import convert_coco
from configs.config import home_path


def preparing_folders(data_dir, data_set):
    # Move image amd annotation files
    new_image_path = os.path.join(data_dir, data_set + "_yolov8_converted/images")
    new_labels_path = os.path.join(data_dir, data_set + "_yolov8_converted/labels")
    path_with_labels_annotation = os.path.join(new_labels_path, "_annotations.coco")

    files_to_move = glob.glob(
        os.path.join(data_dir, data_set, "*.jpg")
    )

    for file_path in files_to_move:
        shutil.move(file_path, new_image_path)

    for item in os.listdir(path_with_labels_annotation):
        source_item = os.path.join(path_with_labels_annotation, item)
        shutil.move(source_item, new_labels_path)

    # Remove the original test directories
    shutil.rmtree(os.path.join(data_dir, data_set))
    shutil.rmtree(path_with_labels_annotation)


def convert_coco2yolov8(data_set):
    """
    Convert COCO format to YOLOv8 format and move image files.
    """
    data_dir = os.path.join(home_path, "data/BuildingsInstanceSegmentationImageDataset/DataLoader/Data")
    convert_coco(
        labels_dir=os.path.join(data_dir, data_set),
        use_segments=True,
        save_dir=os.path.join(data_dir, data_set + "_yolov8_converted")
    )

    preparing_folders(data_dir, data_set)

