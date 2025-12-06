import timm
import torch
from torch import nn


class CloverModel(nn.Module):
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
        head_connection_type: str = "direct",
    ):
        super(CloverModel, self).__init__()
        self.head_connection_type = head_connection_type

        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )

        if head_connection_type == "direct":
            self.target_head = nn.Sequential(
                nn.Linear(self.model.num_features, n_classes),
                nn.ReLU(),
            )
            self.clover_classification_head = nn.Sequential(
                nn.Linear(self.model.num_features, 1),
            )
        elif head_connection_type == "class_head":
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
        else:
            raise ValueError(f"Invalid head_connection_type: {head_connection_type}")

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        img = input["image"]
        emb = self.model(img)

        if self.head_connection_type != "direct":
            emb = self.target_emb(emb)
        output = self.target_head(emb)
        clover_output = self.clover_classification_head(emb)

        output = {"logits": output, "include_clover_pred": clover_output}
        return output


if __name__ == "__main__":
    batch_size = 4
    model = CloverModel(
        backbone_name="tf_efficientnet_b0",
        pretrained=True,
        in_channels=3,
        n_classes=5,
        emb_dim=128,
        aux_dim_reduction_factor=2,
        head_connection_type="class_head",
    )

    sample_input = {"image": torch.randn(batch_size, 3, 224, 224)}
    output = model(sample_input)
    print(output["logits"].shape)  # Expected output shape: (4, 18)

    print(output["include_clover_pred"].shape)  # Expected output shape: (4, 1)
