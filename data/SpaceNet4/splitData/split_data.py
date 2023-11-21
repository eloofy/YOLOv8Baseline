from ultralytics.data.utils import autosplit
from configs.config import home_path
import os


def main():
    autosplit(os.path.join(home_path, "data/SpaceNet4/DataPrepared/images"), (0.8, 0.1, 0.1), False)


if __name__ == "__main__":
    main()
