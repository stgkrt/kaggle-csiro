import timm
import torch
from torch import nn


class HeightGHSSModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        in_channels: int,
        n_classes: int,
        emb_dim: int = 128,
        aux_dim_reduction_factor: int = 2,
    ):
        super(HeightGHSSModel, self).__init__()
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
        )

        print(self.model.num_classes)
        self.target_head = nn.Sequential(
            nn.Linear(self.model.num_classes, emb_dim),
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim),
            nn.Linear(emb_dim, n_classes),
            nn.ReLU(),
        )
        # aux targets(height, gshh) heads
        aux_head_dim = emb_dim // aux_dim_reduction_factor
        self.height_head = nn.Sequential(
            nn.Linear(self.model.num_classes, aux_head_dim),
            nn.ReLU(),
            nn.BatchNorm1d(aux_head_dim),
            nn.Linear(aux_head_dim, 1),
        )
        self.gshh_head = nn.Sequential(
            nn.Linear(self.model.num_classes, aux_head_dim),
            nn.ReLU(),
            nn.BatchNorm1d(aux_head_dim),
            nn.Linear(aux_head_dim, 1),
        )

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        img = input["image"]
        emb = self.model(img)
        output = self.target_head(emb)
        height = self.height_head(emb)
        gshh = self.gshh_head(emb)

        output = {"logits": output, "height": height, "gshh": gshh}
        return output


if __name__ == "__main__":
    batch_size = 4
    model = HeightGHSSModel(
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
    print(output["gshh"].shape)  # Expected output shape: (4, 1)
