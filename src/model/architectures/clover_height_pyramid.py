import timm
import torch
from torch import nn


class CloverHeightPyramidModel(nn.Module):
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
        super(CloverHeightPyramidModel, self).__init__()
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            drop_path_rate=drop_path_rate,
            features_only=True,
        )
        self.encoder_out_channels = self.model.feature_info.channels()
        self.feature_extractor = nn.ModuleList()
        for i in range(len(self.encoder_out_channels) - 1):
            in_ch = self.encoder_out_channels[i]
            self.feature_extractor.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, emb_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(emb_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.AdaptiveAvgPool2d(1),
                )
            )
        self.target_head = nn.Sequential(
            nn.Linear(emb_dim * (len(self.encoder_out_channels) - 1), n_classes),
            nn.ReLU(),
        )
        self.clover_classification_head = nn.Sequential(
            nn.Linear(emb_dim * (len(self.encoder_out_channels) - 1), 1),
        )
        self.height_head = nn.Sequential(
            nn.Linear(emb_dim * (len(self.encoder_out_channels) - 1), 1),
            nn.ReLU(),
        )

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        img = input["image"]
        features = self.model(img)
        pyramid_features = []
        for i, feature in enumerate(features[:-1]):
            x = self.feature_extractor[i](feature)  # (batch, emb_dim, 1, 1)
            x = x.view(x.size(0), -1)  # (batch, emb_dim)
            pyramid_features.append(x)
        # concated pyramid features = (batch, emb_dim * num_features)
        pyramid_features = torch.cat(pyramid_features, dim=1)
        output = self.target_head(pyramid_features)
        clover_output = self.clover_classification_head(pyramid_features)
        height = self.height_head(pyramid_features)

        output = {
            "logits": output,
            "include_clover_preds": clover_output,
            "height": height,
        }
        return output


if __name__ == "__main__":
    batch_size = 4
    model = CloverHeightPyramidModel(
        backbone_name="tf_efficientnet_b0",
        pretrained=True,
        in_channels=3,
        n_classes=5,
    )

    sample_input = {"image": torch.randn(batch_size, 3, 224, 224)}
    output = model(sample_input)
    print(output["logits"].shape)  # Expected output shape: (4, 5)
    print(output["include_clover_preds"].shape)  # Expected output shape: (4, 1)
