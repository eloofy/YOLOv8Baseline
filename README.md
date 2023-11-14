# YOLOv8Baseline
## Data Information 


## Project Organization 
------------

    YOLOv8Baseline/
      ├── configs/
      │   ├── config.py                      <- Config to genereate homedir and another features.
      │   └── model_cfg.yaml                 <- Config with training params.
      │
      ├── data/                              <- Data path.
      │   └── BuildInstSegmeImData/          <- Dataset load path.
      │       └── DataLoader/
      │           ├── load_prepare_data.py   <- Script to lad data from roboflow
      │           └── config.json            <- Config to connect and load data from roboflow.
      │   ├── SpaceNet4/                     <- Data SN4.
      │
      │   └── convert/                       <- Path with scripts which convert data from COCO to YOLOv8 format.
      │       ├── convert_coco_yolo.py       <- Script to convert Buildings Instance Segmentation Dataset from COCO to YOLO.
      │       └── convert_geojson2coco.py    <- Script to convert SN4 Dataset to YOLO.
      │
      ├── src/
      │   ├── inference/                     <- Inference path.
      │       └── predict.py                 <- Inference script.
      │ 
      │   └── train/                         <- Train path.
      │       ├── Callbacks/                 <- Callbacks path.
      │           └── callbacks.py           <- Callbacks module.
      │
      │       ├── Maintrain/                 <- Main train path.
      │           ├── runs/                  <- All results of train with best weights path.
      │           ├── TrainDataConfigs/      <- Data configs for each dataset.
      │           └── train.py               <- Train script
      │
      │       └── MLflowtracking/                
      │           └── cust_mlflow.py         <- Custom mlflow module.  
      


--------