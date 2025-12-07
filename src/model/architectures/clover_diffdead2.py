import timm
import torch
from torch import nn


class CloverDiffDead2Model(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        in_channels: int,
        n_classes: int,
        emb_dim: int = 32,
        dropout: float = 0.2,
        drop_path_rate: float = 0.3,
    ):
        super(CloverDiffDead2Model, self).__init__()
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )

        self.target_emb = nn.Sequential(
            nn.Linear(self.model.num_features, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.target_head = nn.Sequential(
            nn.Linear(emb_dim, n_classes),
            nn.ReLU(),
        )
        self.clover_classification_head = nn.Sequential(
            nn.Linear(emb_dim, 1),
        )

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        img = input["image"]
        emb = self.model(img)
        emb = self.target_emb(emb)
        output = self.target_head(emb)
        # total = gdm - dead -> dead = total - gdm
        dead_from_total = output[:, 4] - output[:, 3]
        # gdm = green + clover -> dead = total - (green + clover)
        dead_from_sum = output[:, 4] - (output[:, 0] + output[:, 2])
        # 元々のchは補正要因として使う
        output_dead = ((dead_from_total + dead_from_sum) * 0.5) + output[:, 1]
        output = torch.cat(
            [output[:, 0:1], output_dead.unsqueeze(1), output[:, 2:5]],
            dim=1,
        )
        # clover classification head
        clover_output = self.clover_classification_head(emb)

        output = {"logits": output, "include_clover_preds": clover_output}
        return output


if __name__ == "__main__":
    batch_size = 4
    model = CloverDiffDead2Model(
        backbone_name="tf_efficientnet_b0",
        pretrained=True,
        in_channels=3,
        n_classes=5,
    )

    sample_input = {"image": torch.randn(batch_size, 3, 224, 224)}
    output = model(sample_input)
    print(output["logits"].shape)  # Expected output shape: (4, 5)
    print(output["include_clover_preds"].shape)  # Expected output shape: (4, 1)
