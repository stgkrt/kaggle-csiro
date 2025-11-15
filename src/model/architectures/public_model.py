import torch
import torch.nn as nn

from src.model.architectures.model_blocks import AttentionLayer, ResidualSECNNBlock


class SimpleCNNBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=3, pool_size=2, drop=0.2):
        super(SimpleCNNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_filters, out_filters, kernel_size, padding="same", bias=False),
            nn.BatchNorm1d(out_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.block(x)


class PublicModel(nn.Module):
    def __init__(self, pad_len=127, imu_dim=11, tof_dim=25, n_classes=18):
        super(PublicModel, self).__init__()
        self.imu_dim = imu_dim
        self.tof_dim = tof_dim

        self.imu_branch_block1 = ResidualSECNNBlock(imu_dim, 64, 3, drop=0.1)
        self.imu_branch_block2 = ResidualSECNNBlock(64, 128, 5, drop=0.1)

        self.tof_branch_block1 = SimpleCNNBlock(tof_dim, 64, drop=0.2)
        self.tof_branch_block2 = SimpleCNNBlock(64, 128, drop=0.2)

        merged_cnn_features = 128 + 128  # 128 from IMU, 128 from TOF
        self.recurrent_layer = nn.LSTM(
            input_size=merged_cnn_features,
            hidden_size=128,
            bidirectional=True,
            batch_first=True,
        )
        recurrent_output_features = 128 * 2

        self.attention_layer = AttentionLayer(input_dim=recurrent_output_features)
        self.classifier_head = nn.Sequential(
            nn.Linear(recurrent_output_features, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.output_layer = nn.Linear(128, n_classes)

    def forward(self, input_features: dict[str, torch.Tensor]) -> torch.Tensor:
        x = input_features["features"]
        imu = x[:, :, : self.imu_dim]
        tof = x[:, :, self.imu_dim :]
        imu = imu.permute(0, 2, 1)
        tof = tof.permute(0, 2, 1)

        x1 = self.imu_branch_block1(imu)
        x1 = self.imu_branch_block2(x1)  # Shape: (batch, 128, new_seq_len)

        x2 = self.tof_branch_block1(tof)
        x2 = self.tof_branch_block2(x2)  # Shape: (batch, 128, new_seq_len)

        merged = torch.cat([x1, x2], dim=1)  # Shape: (batch, 256, new_seq_len)
        merged = merged.permute(0, 2, 1)

        recurrent_out, _ = self.recurrent_layer(merged)  # Shape: (batch, seq_len, 256)
        attention_out = self.attention_layer(recurrent_out)  # Shape: (batch, 256)

        classified = self.classifier_head(attention_out)
        logits = self.output_layer(classified)
        outputs = {"logits": logits}

        return outputs


class PublicIMUModel(nn.Module):
    def __init__(self, pad_len=127, imu_dim=11, n_classes=18):
        super(PublicIMUModel, self).__init__()
        hidden_size = 128
        self.imu_dim = imu_dim

        self.imu_branch_block1 = ResidualSECNNBlock(
            imu_dim, hidden_size // 2, 3, drop=0.1
        )
        self.imu_branch_block2 = ResidualSECNNBlock(
            hidden_size // 2, hidden_size, 5, drop=0.1
        )

        self.recurrent_layer = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
        )

        self.attention_layer = AttentionLayer(input_dim=hidden_size * 2)
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2, bias=False),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.output_layer = nn.Linear(hidden_size, n_classes)

    def forward(self, input_features: dict[str, torch.Tensor]) -> torch.Tensor:
        x = input_features["features"]
        x = x.permute(0, 2, 1)

        x = self.imu_branch_block1(x)
        x = self.imu_branch_block2(x)  # Shape: (batch, 128, new_seq_len)

        x = x.permute(0, 2, 1)

        recurrent_out, _ = self.recurrent_layer(x)  # Shape: (batch, seq_len, 128)
        attention_out = self.attention_layer(recurrent_out)  # Shape: (batch, 128)

        classified = self.classifier_head(attention_out)
        logits = self.output_layer(classified)
        outputs = {"logits": logits}  # Return logits in a dictionary

        return outputs


if __name__ == "__main__":
    # Example usage
    pad_len = 127
    imu_dim = 11  # Example IMU dimension
    tof_dim = 25  # Example ToF dimension
    n_classes = 18  # Example number of classes
    batch_size = 32

    # model = PublicModel(pad_len, imu_dim, tof_dim, n_classes)

    # # Create a dummy input tensor with shape (batch_size, pad_len, imu_dim + tof_dim)
    # dummy_input = torch.randn(batch_size, pad_len, imu_dim + tof_dim)
    # dummy_input = {"features": dummy_input}
    # # Forward pass
    # output = model(dummy_input)
    # print("Output shape:", output.shape)  # Should be (batch_size, n_classes)

    model_imu = PublicIMUModel(pad_len, imu_dim, n_classes)
    dummy_input_imu = torch.randn(batch_size, pad_len, imu_dim)
    dummy_input_imu = {"features": dummy_input_imu}
    output_imu = model_imu(dummy_input_imu)
    print(
        "IMU Model Output shape:", output_imu["logits"].shape
    )  # Should be (batch_size, n_classes)
