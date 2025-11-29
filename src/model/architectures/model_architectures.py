from typing import Union

import torch
from torch import nn

from src.configs import ModelConfig
from src.model.architectures.height_gshh_model import HeightGHSSModel
from src.model.architectures.simple_clover_diff import SimpleCloverDiffModel
from src.model.architectures.simple_model import SimpleModel
from src.model.architectures.simple_total import SimpleTotalModel

MODEL_TYPE = Union[
    SimpleModel, SimpleTotalModel, SimpleCloverDiffModel, HeightGHSSModel
]


def get_model_architecture(
    model_name,
    backbone_name,
    pretrained,
    in_channels,
    n_classes,
    emb_dim=128,
    aux_dim_reduction_factor=2,
) -> MODEL_TYPE:
    if model_name == "simple_model":
        model: MODEL_TYPE = SimpleModel(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_channels=in_channels,
            n_classes=n_classes,
        )
    elif model_name == "simple_total":
        model = SimpleTotalModel(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_channels=in_channels,
            n_classes=n_classes,
        )
    elif model_name == "simple_clover_diff":
        model = SimpleCloverDiffModel(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_channels=in_channels,
            n_classes=n_classes,
        )
    elif model_name == "height_gshh_model":
        model = HeightGHSSModel(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_channels=in_channels,
            n_classes=n_classes,
            emb_dim=emb_dim,
            aux_dim_reduction_factor=aux_dim_reduction_factor,
        )
    else:
        print(f"Model {model_name} not implemented.")
        raise NotImplementedError
    return model


class ModelArchitectures(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super(ModelArchitectures, self).__init__()
        self.config = model_config
        self.model = get_model_architecture(
            model_name=self.config.model_name,
            backbone_name=self.config.backbone_name,
            pretrained=self.config.pretrained,
            in_channels=self.config.in_channels,
            n_classes=self.config.n_classes,
            emb_dim=self.config.emb_dim,
            aux_dim_reduction_factor=self.config.aux_dim_reduction_factor,
        )

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
