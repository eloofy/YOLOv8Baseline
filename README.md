# YOLOv8Baseline
Project Organization 
------------

    YOLOv8Baseline/
      ├── configs/
      │   ├── config.py                      <- Config to genereate homedir and another features.
      │   └── model_cfg.yaml                 <- Config with training params.
      │
      ├── data/                              <- Data path.
      │   └── BuildInstSegmeImData/          <- Dataset load path.
      │       └── DataLoader/
      │           ├── load_prepare_data.py  <- Script to lad data from roboflow
      │           └── config.json            <- Config to connect and load data from roboflow.
      │
      │   └── convert/                       <- Path with scripts which convert data from COCO to YOLOv8 format.
      │       └── convert_coco_yolo.py       <- Script to convert Buildings Instance Segmentation Dataset from COCO to YOLO.
      │
      ├── src/
      │   ├── inference/                     <- Dataset load path.
      │       └── predict.py                 <- Predcit script.
      │   └── train/                         <- Train path.
      │       └── Maintrain/                 <- Main train path.
      │           ├── runs/                  <- All results of train with best weights path.
      │           ├── data.yaml              <- Data config for train.
      │           └── train.py               <- train script
      


--------
