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


class BranchCNNLSTMModel(nn.Module):
    def __init__(
        self,
        imu_dim=11,
        tof_dim=25,
        thm_dim=5,
        n_classes=18,
        default_emb_dim=16,
        layer_num=5,
    ):
        super(BranchCNNLSTMModel, self).__init__()
        self.imu_dim = imu_dim
        self.tof_dim = tof_dim
        self.thm_dim = thm_dim
        self.default_emb_dim = default_emb_dim
        self.layer_num = layer_num
        self.divnum_imu_to_other = 2
        print(
            f"Initializing Branch CNN LSTM Model with"
            f"imu_dim={imu_dim}, tof_dim={tof_dim}, thm_dim={thm_dim}, "
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
            # imu_out_filters = imu_in_filters * 2
            imu_out_filters = imu_in_filters
            self.imu_blocks.append(
                ResidualSECNNBlock(
                    in_filters=imu_in_filters,
                    out_filters=imu_out_filters,
                    kernel_size=3,
                    drop=0.2,
                )
            )
        # tof blocks
        self.tof_blocks = nn.ModuleList([])
        self.tof_blocks.append(
            ResidualSECNNBlock(
                self.tof_dim,
                self.default_emb_dim // self.divnum_imu_to_other,
                3,
                drop=0.1,
            )
        )
        tof_in_filters = self.default_emb_dim // self.divnum_imu_to_other
        tof_out_filters = self.default_emb_dim // self.divnum_imu_to_other
        for i in range(self.layer_num):
            if i % 2 == 0:
                tof_in_filters = tof_out_filters
                # tof_out_filters = tof_in_filters * 2
                tof_out_filters = tof_in_filters
            else:
                tof_in_filters = tof_out_filters
            self.tof_blocks.append(
                ResidualSECNNBlock(
                    in_filters=tof_in_filters,
                    out_filters=tof_out_filters,
                    kernel_size=3,
                    drop=0.2,
                )
            )
        # thm blocks
        self.thm_blocks = nn.ModuleList([])
        self.thm_blocks.append(
            ResidualSECNNBlock(
                self.thm_dim,
                self.default_emb_dim // self.divnum_imu_to_other,
                3,
                drop=0.1,
            )
        )
        thm_in_filters = self.default_emb_dim // self.divnum_imu_to_other
        thm_out_filters = self.default_emb_dim // self.divnum_imu_to_other
        for i in range(self.layer_num):
            if i % 2 == 0:
                thm_in_filters = thm_out_filters
                # thm_out_filters = thm_in_filters * 2
                thm_out_filters = thm_in_filters
            else:
                thm_in_filters = thm_out_filters
            self.thm_blocks.append(
                ResidualSECNNBlock(
                    in_filters=thm_in_filters,
                    out_filters=thm_out_filters,
                    kernel_size=3,
                    drop=0.2,
                )
            )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        lstm_filters = imu_out_filters + tof_out_filters + thm_out_filters
        self.lstm = nn.LSTM(
            input_size=lstm_filters,
            hidden_size=lstm_filters // 2,
            bidirectional=True,
            batch_first=True,
        )
        self.gru = nn.GRU(
            input_size=lstm_filters,
            hidden_size=lstm_filters // 2,
            bidirectional=True,
            batch_first=True,
        )
        class_in_filters = lstm_filters
        self.classifier_head = nn.Sequential(
            nn.Linear(class_in_filters, class_in_filters * 2, bias=False),
            nn.BatchNorm1d(class_in_filters * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(class_in_filters * 2, class_in_filters, bias=False),
            nn.BatchNorm1d(class_in_filters),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.output_layer = nn.Linear(class_in_filters, n_classes)

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
        imu = self._get_signal_features(input_features["imu_features"], self.imu_blocks)
        tof = self._get_signal_features(input_features["tof_features"], self.tof_blocks)
        thm = self._get_signal_features(input_features["thm_features"], self.thm_blocks)

        x = torch.cat([imu, tof, thm], dim=-1)
        x, _ = self.lstm(x)
        x, _ = self.gru(x)

        classified = self.classifier_head(x)
        logits = self.output_layer(classified)
        outputs = {"logits": logits}  # Return logits in a dictionary

        return outputs


if __name__ == "__main__":
    # Example usage
    pad_len = 127
    imu_dim = 11  # Example IMU dimension
    tof_dim = 25  # Example ToF dimension
    thm_dim = 5  # Example thermal dimension
    n_classes = 18  # Example number of classes
    batch_size = 32

    simple_model = BranchCNNLSTMModel(imu_dim, tof_dim, thm_dim, n_classes)
    imu_input = torch.randn(batch_size, pad_len, imu_dim)
    tof_input = torch.randn(batch_size, pad_len, tof_dim)
    thm_input = torch.randn(batch_size, pad_len, thm_dim)  # Assuming
    dummy_input_simple = {
        "imu_features": imu_input,
        "tof_features": tof_input,
        "thm_features": thm_input,
    }

    output_simple = simple_model(dummy_input_simple)
    print("Simple Model Output shape:", output_simple["logits"].shape)
