import timm
import torch
from torch import nn


class HeightModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        in_channels: int,
        n_classes: int,
        emb_dim: int = 32,
        aux_dim_reduction_factor: int = 2,
        dropout: float = 0.2,
        drop_path_rate: float = 0.3,
    ):
        super(HeightModel, self).__init__()
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )

        self.target_head = nn.Sequential(
            # nn.Linear(self.model.num_features, emb_dim),
            # nn.BatchNorm1d(emb_dim * 2),  # 入れると調整難しすぎる
            # nn.ReLU(),
            # nn.Linear(emb_dim * 2, emb_dim),
            # nn.BatchNorm1d(emb_dim),  # 入れると調整難しすぎる
            # nn.ReLU(),
            # nn.Linear(emb_dim, n_classes),
            nn.Linear(self.model.num_features, n_classes),
            nn.ReLU(),
            # nn.Dropout(dropout),
        )
        # aux targets(height, gshh) heads
        aux_head_dim = emb_dim // aux_dim_reduction_factor
        self.height_head = nn.Sequential(
            nn.Linear(self.model.num_features, aux_head_dim),
            # nn.BatchNorm1d(aux_head_dim),  # 入れると調整難しすぎる
            nn.ReLU(),
            nn.Linear(aux_head_dim, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        img = input["image"]
        emb = self.model(img)
        output = self.target_head(emb)
        height = self.height_head(emb)

        output = {"logits": output, "height": height}
        return output


if __name__ == "__main__":
    batch_size = 4
    model = HeightModel(
        backbone_name="tf_efficientnet_b0",
        pretrained=True,
        in_channels=3,
        n_classes=5,
        emb_dim=128,
        aux_dim_reduction_factor=2,
    )

    sample_input = {"image": torch.randn(batch_size, 3, 224, 224)}
    output = model(sample_input)
    print(output["logits"].shape)  # Expected output shape: (4, 18)

    print(output["height"].shape)  # Expected output shape: (4, 1)
