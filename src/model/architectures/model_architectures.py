from typing import Union

import torch
from torch import nn

from src.configs import ModelConfig
from src.model.architectures.clover_diffdead import CloverDiffDeadModel
from src.model.architectures.clover_diffdead2 import CloverDiffDead2Model
from src.model.architectures.clover_diffdead3 import CloverDiffDead3Model
from src.model.architectures.clover_model import CloverModel
from src.model.architectures.clover_sum import CloverSumModel
from src.model.architectures.clover_sum_height import CloverSumHeightModel
from src.model.architectures.height_gshh_model import HeightGHSSModel
from src.model.architectures.height_model import HeightModel
from src.model.architectures.simple_clover_diff import SimpleCloverDiffModel
from src.model.architectures.simple_model import SimpleModel
from src.model.architectures.simple_total import SimpleTotalModel

MODEL_TYPE = Union[
    SimpleModel,
    SimpleTotalModel,
    SimpleCloverDiffModel,
    HeightGHSSModel,
    HeightModel,
    CloverModel,
    CloverDiffDeadModel,
    CloverDiffDead2Model,
    CloverDiffDead3Model,
    CloverSumModel,
    CloverSumHeightModel,
]


def get_model_architecture(
    model_name,
    backbone_name,
    pretrained,
    in_channels,
    n_classes,
    emb_dim=128,
    aux_dim_reduction_factor=2,
    head_connection_type="direct",
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
    elif model_name == "height_model":
        model = HeightModel(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_channels=in_channels,
            n_classes=n_classes,
            emb_dim=emb_dim,
            aux_dim_reduction_factor=aux_dim_reduction_factor,
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
    elif model_name == "clover_model":
        model = CloverModel(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_channels=in_channels,
            n_classes=n_classes,
            emb_dim=emb_dim,
            aux_dim_reduction_factor=aux_dim_reduction_factor,
            head_connection_type=head_connection_type,
        )
    elif model_name == "clover_diffdead":
        model = CloverDiffDeadModel(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_channels=in_channels,
            n_classes=n_classes,
            emb_dim=emb_dim,
        )
    elif model_name == "clover_diffdead2":
        model = CloverDiffDead2Model(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_channels=in_channels,
            n_classes=n_classes,
            emb_dim=emb_dim,
        )
    elif model_name == "clover_diffdead3":
        model = CloverDiffDead3Model(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_channels=in_channels,
            n_classes=n_classes,
            emb_dim=emb_dim,
        )
    elif model_name == "clover_sum":
        model = CloverSumModel(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_channels=in_channels,
            n_classes=n_classes,
            emb_dim=emb_dim,
        )
    elif model_name == "clover_sum_height":
        model = CloverSumHeightModel(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_channels=in_channels,
            n_classes=n_classes,
            emb_dim=emb_dim,
            head_connection_type=head_connection_type,
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
            head_connection_type=self.config.head_connection_type,
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
