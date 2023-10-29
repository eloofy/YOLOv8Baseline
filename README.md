# YOLOv8Baseline
Project Organization
------------

    DeepGlobeClassificationPr/
      ├── configs/
      │   ├── config.py              <- Config to genereate homedir and classRGBvalues
      │   └── config_train_test.py   <- Config with train and test augmentation
      ├── data/
      │   └── data_download.py       <- Script to download data from Kaggle(need access token)
      ├── dataLoader/
      │   ├── data_loader.py         <- Script to preprocess and load data
      │   └── data_module.py         <- Script to generate train/val datatoader
      ├── mlartifacts/
      ├── mlruns/
      └── src/
      │   └── CustomUnetNN/
      │      ├── fine_tune_model.py     <- Script to fineTune model from best checkpoint
      │      ├── train.py               <- Script to train
      │      ├── unet_nn.py             <- Class with NNUnet
      │      └── utils.py               <- Class with logging and configure with pytorch_lightning
      │   └── Production/
      │      ├── predict.py             <- Inference script
      


--------
