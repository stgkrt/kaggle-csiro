import numpy as np
import torch
import torch.nn as nn
from torchaudio.transforms import AmplitudeToDB, Spectrogram


class SpecNormalize(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x: (batch, channel, freq, time)
        min_ = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_ = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        return (x - min_) / (max_ - min_ + self.eps)


class SpecFeatureExtractor(nn.Module):
    def __init__(
        self,
        out_channels: int,
        height: int,
        hop_length: int,
        win_length: int | None = None,
        is_height_channel: bool = False,
    ):
        super().__init__()
        self.height = height
        self.out_channels = out_channels
        self.is_height_channel = is_height_channel
        n_fft = height * 2 - 1
        self.feature_extractor = nn.Sequential(
            Spectrogram(n_fft=n_fft, hop_length=None, win_length=None),
            AmplitudeToDB(top_db=80),
            SpecNormalize(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((self.out_channels, self.out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # output shape: (batch, sensor_num, height, time)
        img = self.feature_extractor(x)
        if self.is_height_channel:
            # Change shape to (batch_size, height, sensor, time)
            img = img.permute(0, 2, 1, 3)
        img = self.avg_pool(img)
        return img


class ResidualSECNNBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=3, drop=0.1):
        super(ResidualSECNNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_filters, out_filters, kernel_size, padding="same"),
            nn.BatchNorm2d(out_filters),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Conv2d(out_filters, out_filters, kernel_size, padding="same"),
            nn.BatchNorm2d(out_filters),
        )

        if in_filters != out_filters:
            self.shortcut = nn.Conv2d(in_filters, out_filters, 1)
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x if self.shortcut is None else self.shortcut(x)
        return self.block(x) + residual


class SpecModel(nn.Module):
    def __init__(
        self,
        imu_dim: int,
        n_classes: int = 18,
        default_emb_dim: int = 32,
        layer_num: int = 5,
        height: int = 64,
        hop_length: int = 8,
        win_length: int = 4,
        is_height_channel: bool = False,
    ):
        super().__init__()
        self.feature_extractor = SpecFeatureExtractor(
            out_channels=default_emb_dim,
            height=height,
            hop_length=hop_length,
            win_length=win_length,
            is_height_channel=is_height_channel,
        )

        self.imu_blocks = nn.ModuleList([])
        if is_height_channel:
            out_filters = height
            self.imu_blocks.append(
                ResidualSECNNBlock(
                    in_filters=height,
                    out_filters=out_filters,
                    kernel_size=3,
                    drop=0.2,
                )
            )
        else:
            out_filters = imu_dim
            self.imu_blocks.append(
                ResidualSECNNBlock(
                    in_filters=imu_dim,
                    out_filters=out_filters,
                    kernel_size=3,
                    drop=0.2,
                )
            )
        for _ in range(layer_num):
            in_filters = out_filters
            out_filters = in_filters * 2
            self.imu_blocks.append(
                ResidualSECNNBlock(
                    in_filters=in_filters,
                    out_filters=out_filters,
                    kernel_size=3,
                    drop=0.2,
                )
            )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier_head = nn.Sequential(
            nn.Linear(out_filters, out_filters * 2, bias=False),
            nn.BatchNorm1d(out_filters * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(out_filters * 2, n_classes, bias=False),
        )

    def forward(
        self, input_features: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        x = input_features["features"]  # (batch, time, sensor_num)
        x = x.permute(0, 2, 1)  # Change shape to (batch, sensor_num, time)
        # output shape: (batch, sensor_num, height, time)
        x = self.feature_extractor(x)

        # (batch, channels, height, time)
        for block in self.imu_blocks:
            x = block(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.classifier_head(x)
        output = {"logits": x}
        return output


if __name__ == "__main__":
    # Example usage
    model = SpecModel(imu_dim=11, n_classes=18, hop_length=8, win_length=4)
    input_tensor = torch.randn(8, 95, 11)  # (batch_size, imu_dim, sequence_length)
    input_tensor = {"features": input_tensor}
    output = model(input_tensor)
    print(
        output["logits"].shape
    )  # Should be (8, channels, time) after feature extraction and blocks
