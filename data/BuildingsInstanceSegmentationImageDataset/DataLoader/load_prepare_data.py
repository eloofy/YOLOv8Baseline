import logging
import json
from roboflow import Roboflow
from data.convert.convert_coco_yolo_buildings_instance_segmentation import convert_coco2yolov8


class DatasetDownloader:
    def __init__(self, config_file="config.json"):
        self.config_file = config_file

    def download_dataset(self):
        """
        Downloads a dataset from RoboFlow to a local location.
        """
        try:
            config = self.load_config()
            api_key = config["api_key"]
            workspace_name = config["workspace_name"]
            project_name = config["project_name"]
            version_number = config["version_number"]
            export_format = config["export_format"]
            local_location = config["local_location"]

            rf = Roboflow(api_key=api_key)
            workspace = rf.workspace(workspace_name)
            project = workspace.project(project_name)
            version = project.version(version_number)

            version.download(export_format, location=local_location)

            logger.info(f"Dataset successfully downloaded to '{local_location}'")
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")

    def load_config(self):
        """
        Load configuration from a JSON file.
        """
        with open(self.config_file, "r") as config_file:
            config = json.load(config_file)
        return config

    def run(self, data_sets, with_converted):
        # Download the dataset
        self.download_dataset()

        # Convert the dataset to YOLOv8 format
        if with_converted:
            for data_set in data_sets:
                convert_coco2yolov8(data_set)


def main():
    dataset_downloader = DatasetDownloader()
    dataset_downloader.run(["train", "valid", "test"], True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    main()
