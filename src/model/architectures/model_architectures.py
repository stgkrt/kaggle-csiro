import torch
from torch import nn

from src.configs import ModelConfig
from src.model.architectures.simple_model import SimpleModel


class ModelArchitectures(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super(ModelArchitectures, self).__init__()
        self.config = model_config
        self.model = self._get_model()

    def _get_model(self):
        if self.config.model_name == "simple_model":
            model = SimpleModel(
                backbone_name=self.config.backbone_name,
                pretrained=self.config.pretrained,
                in_channels=self.config.in_channels,
                n_classes=self.config.n_classes,
            )
        else:
            print(f"Model {self.model_name} not implemented.")
            raise NotImplementedError
        return model

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = self.model(x)
        return x


if __name__ == "__main__":
    batch_size = 32
    config = ModelConfig(
        model_name="simple_model",
        backbone_name="tf_efficientnet_b0",
        in_channels=3,
        n_classes=18,
        pretrained=True,
    )
    model = ModelArchitectures(config)

    sample_input = {"image": torch.randn(batch_size, 3, 224, 224)}
    output = model(sample_input)
    print(output["logits"].shape)  # Expected output shape:
