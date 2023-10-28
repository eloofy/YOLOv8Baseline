from ultralytics import YOLO
from trainer import CustomTrainer


class CustomYOLO(YOLO):
    def __init__(self, model_name: str):
        super().__init__(model_name)

    def train(self, **kwargs):
        args = kwargs["training_params"]
        self.trainer = CustomTrainer(overrides=args)

        if not args.get('resume'):
            self.trainer.model = self.trainer.get_model(
                weights=self.model if self.ckpt else None, cfg=self.model.yaml
            )
            self.model = self.trainer.model

        self.trainer.train()



