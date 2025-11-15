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


class CNNLSTMModel(nn.Module):
    def __init__(
        self, imu_dim=11, n_classes=18, default_emb_dim=32, layer_num=5, stride=1
    ):
        super(CNNLSTMModel, self).__init__()
        self.imu_dim = imu_dim
        self.default_emb_dim = default_emb_dim
        self.layer_num = layer_num
        self.stride = stride
        print(
            f"Initializing CNNLSTMModel with imu_dim={imu_dim}, n_classes={n_classes}, "
            f"default_emb_dim={default_emb_dim}, layer_num={layer_num}, stride={stride}"
        )
        self.input_conv = nn.Conv1d(
            imu_dim, self.default_emb_dim, kernel_size=3, padding="same", bias=False
        )
        self.recurrent_layer = nn.LSTM(
            input_size=self.default_emb_dim,
            hidden_size=self.default_emb_dim,
            bidirectional=True,
            batch_first=True,
        )

        self.imu_blocks = nn.ModuleList([])
        self.imu_blocks.append(
            ResidualSECNNBlock(
                self.default_emb_dim * 2, self.default_emb_dim * 2, 3, drop=0.1
            )
        )
        emb_output_dim = self.default_emb_dim * 2
        for _ in range(self.layer_num):
            emb_input_dim = emb_output_dim
            emb_output_dim = emb_input_dim * 2
            self.imu_blocks.append(
                ResidualSECNNBlock(
                    in_filters=emb_input_dim,
                    out_filters=emb_output_dim,
                    kernel_size=3,
                    drop=0.2,
                )
            )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.classifier_head = nn.Sequential(
            nn.Linear(emb_output_dim, emb_output_dim * 2, bias=False),
            nn.BatchNorm1d(emb_output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(emb_output_dim * 2, emb_output_dim, bias=False),
            nn.BatchNorm1d(emb_output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.output_layer = nn.Linear(emb_output_dim, n_classes)

    def forward(
        self, input_features: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        x = input_features["features"]  # Shape: (batch_size, sequence_length, imu_dim)
        x = x.permute(0, 2, 1)
        x = self.input_conv(x)  # Shape: (batch_size, default_emb_dim, sequence_length)
        x = x.permute(0, 2, 1)
        x, _ = self.recurrent_layer(x)  # Shape: (batch, seq_len, 128)
        x = x.permute(
            0, 2, 1
        )  # Change shape to (batch_size, channels, sequence_length)

        for block in self.imu_blocks:
            x = block(x)

        x = self.avg_pool(x)  # Shape: (batch_size, channels, 1)
        x = self.flatten(x)  # Flatten to (batch_size, channels)
        classified = self.classifier_head(x)
        logits = self.output_layer(classified)
        outputs = {"logits": logits}  # Return logits in a dictionary

        return outputs


if __name__ == "__main__":
    # Example usage
    pad_len = 127
    imu_dim = 11  # Example IMU dimension
    tof_dim = 25  # Example ToF dimension
    n_classes = 18  # Example number of classes
    batch_size = 4

    cnn_lstm_model = CNNLSTMModel(imu_dim, n_classes)
    dummy_input_cnn_lstm = torch.randn(batch_size, pad_len, imu_dim)
    dummy_input_cnn_lstm = {"features": dummy_input_cnn_lstm}

    output_cnn_lstm = cnn_lstm_model(dummy_input_cnn_lstm)
    print("CNN-LSTM Model Output shape:", output_cnn_lstm["logits"].shape)
