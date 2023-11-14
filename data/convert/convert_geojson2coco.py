import os
import re
from tqdm import tqdm
from osgeo import gdal
import numpy as np
from configs.config import home_path
import glob
import pandas as pd


class SpaceNetDataConvert:
    """
    Class for converting SpaceNet data to a specific format.
    """

    def __init__(
        self,
        data_convert_path: str,
        dirs_images_path: str,
        data_dir_labels_csv: str,
        data_dir_labels_txt: str,
        data_shape: float,
    ):
        """
        Initialize the SpaceNetDataConvert object.

        Parameters:
        - data_convert_path: Destination directory for converted data.
        - dirs_images_path: Path to the raw image directories.
        - data_dir_labels_csv: Path to the directory containing CSV label files.
        - data_dir_labels_txt: Path to the directory where converted labels will be saved.
        - data_shape: Data shape parameter.
        """
        self.data_convert_path = os.path.join(
            home_path, data_convert_path
        )  # dest_dir_name
        self.data_image_dirs_path = os.path.join(home_path, dirs_images_path)
        self.data_summary_labels_csv = os.path.join(home_path, data_dir_labels_csv)
        self.data_labels_txt = os.path.join(home_path, data_dir_labels_txt)
        self.data_shape = data_shape

    def _tif_jpg_conv(self):
        """
        Convert TIFF images to JPEG format and save them to the specified destination directory.
        """

        if not os.path.exists(self.data_convert_path):
            os.mkdir(self.data_convert_path)

        img_dirs = [im[1] for im in os.walk(os.path.join(self.data_image_dirs_path))][0]
        jpg_path = self.data_convert_path

        for images_dir in img_dirs:
            img_files = glob.glob(
                os.path.join(self.data_image_dirs_path, images_dir, "Pan-Sharpen/*.tif")
            )

            for image in tqdm(img_files, desc=os.path.basename(images_dir)):
                dest = os.path.join(jpg_path, os.path.basename(image)[:-4] + ".png")

                g_img = gdal.Open(image)
                scale = []
                for i in range(3):
                    arr = g_img.GetRasterBand(i + 1).ReadAsArray()
                    scale.append([np.percentile(arr, 1), np.percentile(arr, 99)])

                gdal.Translate(
                    dest,
                    image,
                    options=gdal.TranslateOptions(
                        outputType=gdal.GDT_Byte, scaleParams=scale
                    ),
                )

    def _normalize_data_ann(self, data: list):
        """
        Normalize a list of data based on the specified data shape parameter.

        Parameters:
        - list_data: List of data to be normalized.

        Returns:
        - list: Normalized list of data.
        """

        return list(map(str, [round(el / self.data_shape, 6) for el in data]))

    def _get_polypixel_coordinates(self, data: pd.DataFrame):
        """
        Extract polygon pixel coordinates from a DataFrame.

        Parameters:
        - data: DataFrame containing polygon pixel coordinates.

        Returns:
        - list: List of normalized polygon pixel coordinates.
        """
        coordinates = []
        for coordinate in data["PolygonWKT_Pix"].values:
            clear_cor = self._normalize_data_ann(
                list(map(float, re.findall(r"\d+\.\d+", coordinate)))
            )
            if clear_cor:
                coordinates.append(clear_cor)

        return coordinates

    def _write_coordinates_to_txt(self, coordinates, image_id):
        """
        Write normalized coordinates to a text file.

        Parameters:
        - coordinates: List of normalized coordinates.
        - image_id: Image identifier.
        """
        if image_id.split("_")[0] == "Pan-Sharpen":
            image_id = "_".join(image_id.split("_")[1:])

        with open(
            os.path.join(self.data_labels_txt, f"Pan-Sharpen_{image_id}.txt"), "w"
        ) as file:
            for coordinate in coordinates:
                line = " ".join(map(str, coordinate)) + "\n"
                file.write("0 " + line)

    @staticmethod
    def _load_dataset(data_csv_path: str):
        """
        Load a dataset from a CSV file.

        Parameters:
        - data_csv_path: Path to the CSV file.

        Returns:
        - pd.DataFrame: Loaded DataFrame.
        """
        df = pd.read_csv(data_csv_path)

        if "Unnamed: 0" in df.columns:
            df.drop(["Unnamed: 0"], axis=1, inplace=True)

        return df

    def _csv_txt_yolo_convert(self):
        """
        Convert CSV label files to YOLO format and save the results as text files.
        """
        for data_csv in tqdm(
            glob.glob(os.path.join(self.data_summary_labels_csv, "*.csv"))
        ):
            data_csv = self._load_dataset(data_csv)
            unique_images_id = data_csv["ImageId"].unique()

            for image_id in unique_images_id:
                coordinates = self._get_polypixel_coordinates(
                    data_csv[data_csv["ImageId"] == image_id]
                )
                self._write_coordinates_to_txt(coordinates, image_id)

    def run_convert(self):
        """
        Run the data conversion process.
        """
        self._tif_jpg_conv()
        self._csv_txt_yolo_convert()


def main():
    converter = SpaceNetDataConvert(
        "data/SpaceNet4/DataPrepared/images",
        "data/SpaceNet4/DataRaw",
        "data/SpaceNet4/summaryData",
        "data/SpaceNet4/DataPrepared/labels",
        900.000,
    )
    converter.run_convert()


if __name__ == "__main__":
    main()
