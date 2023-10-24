from pathlib import Path

home_path = Path(__file__).parent.parent
seed = 12345

train_parameters = {
    'model_path': '../yolov8n-seg.pt',
    'data_config': 'data.yaml',
    'img_size': 512,
    'epochs': 10,
    'patience': 100,
    'batch_size': 8,
    'device': 0,
    'workers': 12,
    'optimizer': 'AdamW',
    'learning_rate': 0.001,
    'model_name': 'yolov8n_custom',
    'cos_lr': True,
    'show_labels': False
}
