import logging
import json
from roboflow import Roboflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_dataset(
    api_key: str,
    workspace_name: str,
    project_name: str,
    version_number: int,
    export_format: str,
    local_location: str,
):
    """
    Downloads a dataset from RoboFlow to a local location.

    Args:
        api_key: Your RoboFlow API key.
        workspace_name: Name of the RoboFlow workspace.
        project_name: Name of the RoboFlow project.
        version_number: Version number of the project.
        export_format: Export format for the dataset.
        local_location: Local directory where the dataset will be saved.

    Returns:
        None
    """
    rf = Roboflow(api_key=api_key)

    try:
        workspace = rf.workspace(workspace_name)
        project = workspace.project(project_name)

        version = project.version(version_number)

        version.download(export_format, location=local_location)

        logger.info(f"Dataset successfully downloaded to '{local_location}'")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")


def main():
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    download_dataset(
        config["api_key"],
        config["workspace_name"],
        config["project_name"],
        config["version_number"],
        config["export_format"],
        config["local_location"],
    )


if __name__ == "__main__":
    main()
