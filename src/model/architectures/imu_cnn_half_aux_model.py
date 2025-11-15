import torch
import torch.nn as nn

from src.model.architectures.model_blocks import AttentionLayer, ResidualSECNNBlock


class SimpleCNNBlock(nn.Module):
    def __init__(
        self, in_filters, out_filters, kernel_size=3, pool_size=2, drop=0.2, stride=1
    ):
        super(SimpleCNNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_filters,
                out_filters,
                kernel_size,
                stride=stride,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(out_filters),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.block(x)


class ImuCNNHalfAuxModel(nn.Module):
    def __init__(
        self,
        imu_dim=11,
        n_classes=18,
        default_emb_dim=16,
        layer_num=5,
    ):
        super(ImuCNNHalfAuxModel, self).__init__()
        self.imu_dim = imu_dim
        self.default_emb_dim = default_emb_dim
        self.layer_num = layer_num
        self.divnum_imu_to_other = 2
        print(
            f"Initializing IMU Only CNNHalfAuxModel with"
            f"imu_dim={imu_dim}"
            f"default_emb_dim={default_emb_dim}, layer_num={layer_num}"
        )

        # imu blocks
        self.imu_blocks = nn.ModuleList([])
        self.imu_blocks.append(
            ResidualSECNNBlock(imu_dim, self.default_emb_dim, 3, drop=0.1)
        )
        imu_out_filters = self.default_emb_dim
        for _ in range(self.layer_num):
            imu_in_filters = imu_out_filters
            imu_out_filters = imu_in_filters * 2
            self.imu_blocks.append(
                ResidualSECNNBlock(
                    in_filters=imu_in_filters,
                    out_filters=imu_out_filters,
                    kernel_size=3,
                    drop=0.2,
                )
            )
        # Half imu blocks
        self.imu_half_blocks = nn.ModuleList([])
        self.imu_half_blocks.append(
            ResidualSECNNBlock(
                self.imu_dim,
                self.default_emb_dim // 2,
                3,
                drop=0.1,
            )
        )
        imu_half_in_filters = self.default_emb_dim // 2
        imu_half_out_filters = self.default_emb_dim // 2
        for i in range(self.layer_num):
            if i % 2 == 0:
                imu_half_in_filters = imu_half_out_filters
                imu_half_out_filters = imu_half_in_filters * 2
            else:
                imu_half_in_filters = imu_half_out_filters
            self.imu_half_blocks.append(
                ResidualSECNNBlock(
                    in_filters=imu_half_in_filters,
                    out_filters=imu_half_out_filters,
                    kernel_size=3,
                    drop=0.2,
                )
            )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        class_in_filters = imu_out_filters + imu_half_out_filters
        self.classifier_head = nn.Sequential(
            nn.Linear(class_in_filters, class_in_filters * 2, bias=False),
            nn.BatchNorm1d(class_in_filters * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(class_in_filters * 2, class_in_filters, bias=False),
            nn.BatchNorm1d(class_in_filters),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(class_in_filters, n_classes),
        )

        self.orient_head = nn.Sequential(
            nn.Linear(class_in_filters, class_in_filters // 2, bias=False),
            nn.BatchNorm1d(class_in_filters // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(class_in_filters // 2, 4),  # 4 orientations
        )
        self.behavior_head = nn.Sequential(
            nn.Linear(class_in_filters, class_in_filters // 2, bias=False),
            nn.BatchNorm1d(class_in_filters // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(class_in_filters // 2, 4),  # 4 behaviors
        )

    def _get_signal_features(self, input_features, blocks):
        x = input_features.permute(0, 2, 1)
        for block in blocks:
            x = block(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return x

    def forward(
        self, input_features: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        imu_features = input_features["imu_features"]
        imu = self._get_signal_features(imu_features, self.imu_blocks)
        # imu_half = imu_features[:, : imu_features.shape[1] // 2]
        imu_half = imu_features[:, -imu_features.shape[1] // 2 :]
        imu_half = self._get_signal_features(imu_half, self.imu_half_blocks)

        x = torch.cat([imu, imu_half], dim=-1)
        logits = self.classifier_head(x)
        orient_logits = self.orient_head(x)
        behavior_logits = self.behavior_head(x)
        outputs = {
            "logits": logits,
            "orientation": orient_logits,
            "behavior": behavior_logits,
        }

        return outputs


if __name__ == "__main__":
    # Example usage
    pad_len = 127
    imu_dim = 11  # Example IMU dimension
    tof_dim = 25  # Example ToF dimension
    thm_dim = 5  # Example thermal dimension
    n_classes = 18  # Example number of classes
    batch_size = 32

    simple_model = ImuCNNHalfAuxModel(imu_dim, n_classes)
    imu_input = torch.rand(batch_size, pad_len, imu_dim)
    dummy_input_simple = {
        "imu_features": imu_input,
    }

    output_simple = simple_model(dummy_input_simple)
    print("Output logits shape:", output_simple["logits"].shape)
    print("Output orientation shape:", output_simple["orientation"].shape)
    print("Output behavior shape:", output_simple["behavior"].shape)
