import timm
import torch
from torch import nn


class CloverHeightPyramidSumModel(nn.Module):
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
        super(CloverHeightPyramidSumModel, self).__init__()
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
            nn.Linear(emb_dim, n_classes - 2),
            nn.ReLU(),
        )
        self.clover_classification_head = nn.Sequential(
            nn.Linear(emb_dim, 1),
        )
        self.height_head = nn.Sequential(
            nn.Linear(emb_dim, 1),
            nn.ReLU(),
        )

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        img = input["image"]
        features = self.model(img)
        pyramid_features = None
        for i, feature in enumerate(features[:-1]):
            x = self.feature_extractor[i](feature)  # (batch, emb_dim, 1, 1)
            x = x.view(x.size(0), -1)  # (batch, emb_dim)
            if pyramid_features is None:
                pyramid_features = x
            else:
                pyramid_features += x
        output = self.target_head(pyramid_features)
        gdm = output[:, 0] + output[:, 2]
        # total = gdm + dead
        total = torch.sum(output, dim=1, keepdim=True)
        output = torch.cat(
            [output[:, 0:1], output[:, 1:2], output[:, 2:3], gdm.unsqueeze(1), total],
            dim=1,
        )
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
    model = CloverHeightPyramidSumModel(
        backbone_name="tf_efficientnet_b0",
        pretrained=True,
        in_channels=3,
        n_classes=5,
    )

    sample_input = {"image": torch.randn(batch_size, 3, 224, 224)}
    output = model(sample_input)
    print(output["logits"].shape)  # Expected output shape: (4, 5)
    print(output["include_clover_preds"].shape)  # Expected output shape: (4, 1)
