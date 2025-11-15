import torch
from torch import nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        se = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # -> (B, C)
        se = F.relu(self.fc1(se), inplace=True)  # -> (B, C//r)
        se = self.sigmoid(self.fc2(se)).unsqueeze(-1)  # -> (B, C, 1)
        return x * se


class ResNetSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, wd=1e-4):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        # SE
        self.se = SEBlock(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, padding=0, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)  # (B, out, L)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # (B, out, L)
        out = out + identity
        return self.relu(out)


class AttentionLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.score_fn = nn.Linear(feature_dim, 1, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (B, L, F)
        score = torch.tanh(self.score_fn(x))  # (B, L, 1)
        weights = self.softmax(score.squeeze(-1))  # (B, L)
        weights = weights.unsqueeze(-1)  # (B, L, 1)
        context = x * weights  # (B, L, F)
        return context.sum(dim=1)  # (B, F)


class GaussianNoise(nn.Module):
    """Add Gaussian noise to input tensor"""

    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.stddev
            return x + noise
        return x


class CMIBackbone(nn.Module):
    def __init__(self, imu_dim, thm_dim, tof_dim, **kwargs):
        super().__init__()
        self.imu_acc_branch = nn.Sequential(
            self.residual_feature_block(
                3,
                kwargs["imu1_channels"],
                kwargs["imu1_layers"],
                drop=kwargs["imu1_dropout"],
            ),
            self.residual_feature_block(
                kwargs["imu1_channels"],
                kwargs["imu2_channels"],
                kwargs["imu2_layers"],
                drop=kwargs["imu2_dropout"],
            ),
        )
        self.imu_rot_branch = nn.Sequential(
            self.residual_feature_block(
                4,
                kwargs["imu1_channels"],
                kwargs["imu1_layers"],
                drop=kwargs["imu1_dropout"],
            ),
            self.residual_feature_block(
                kwargs["imu1_channels"],
                kwargs["imu2_channels"],
                kwargs["imu2_layers"],
                drop=kwargs["imu2_dropout"],
            ),
        )
        self.imu_other_branch = nn.Sequential(
            self.residual_feature_block(
                imu_dim - 7,
                kwargs["imu1_channels"],
                kwargs["imu1_layers"],
                drop=kwargs["imu1_dropout"],
            ),
            self.residual_feature_block(
                kwargs["imu1_channels"],
                kwargs["imu2_channels"],
                kwargs["imu2_layers"],
                drop=kwargs["imu2_dropout"],
            ),
        )

        self.thm_branch1, self.tof_branch1 = self.init_thm_tof_branch(
            thm_dim // 5, tof_dim // 5, **kwargs
        )
        self.thm_branch2, self.tof_branch2 = self.init_thm_tof_branch(
            thm_dim // 5, tof_dim // 5, **kwargs
        )
        self.thm_branch3, self.tof_branch3 = self.init_thm_tof_branch(
            thm_dim // 5, tof_dim // 5, **kwargs
        )
        self.thm_branch4, self.tof_branch4 = self.init_thm_tof_branch(
            thm_dim // 5, tof_dim // 5, **kwargs
        )
        self.thm_branch5, self.tof_branch5 = self.init_thm_tof_branch(
            thm_dim // 5, tof_dim // 5, **kwargs
        )

        self.imu_proj = ResNetSEBlock(
            in_channels=3 * kwargs["imu2_channels"],
            out_channels=kwargs["imu2_channels"],
        )
        self.thm_proj = ResNetSEBlock(
            in_channels=5 * kwargs["thm2_channels"],
            out_channels=kwargs["thm2_channels"],
        )
        self.tof_proj = ResNetSEBlock(
            in_channels=5 * kwargs["tof2_channels"],
            out_channels=kwargs["tof2_channels"],
        )

        self.lstm = nn.LSTM(
            input_size=kwargs["imu2_channels"]
            + kwargs["thm2_channels"]
            + kwargs["tof2_channels"],
            hidden_size=kwargs["lstm_hidden_size"],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.gru = nn.GRU(
            input_size=kwargs["imu2_channels"]
            + kwargs["thm2_channels"]
            + kwargs["tof2_channels"],
            hidden_size=kwargs["gru_hidden_size"],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.noise = GaussianNoise(kwargs["gaussian_noise_rate"])
        self.dense = nn.Sequential(
            nn.Linear(
                kwargs["imu2_channels"]
                + kwargs["thm2_channels"]
                + kwargs["tof2_channels"],
                kwargs["dense_channels"],
            ),
            nn.ELU(),
        )

        self.attn = AttentionLayer(
            feature_dim=(kwargs["lstm_hidden_size"] + kwargs["gru_hidden_size"]) * 2
            + kwargs["dense_channels"]
        )  # lstm + gru + dense

    def feature_block(
        self, in_channels, out_channels, num_layers, pool_size=2, drop=0.3
    ):
        return nn.Sequential(
            *[
                ResNetSEBlock(in_channels=in_channels, out_channels=in_channels)
                for i in range(num_layers)
            ],
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size, ceil_mode=True),
            nn.Dropout(drop),
        )

    def residual_feature_block(
        self, in_channels, out_channels, num_layers, pool_size=2, drop=0.3
    ):
        return nn.Sequential(
            *[
                ResNetSEBlock(in_channels=in_channels, out_channels=in_channels)
                for i in range(num_layers)
            ],
            ResNetSEBlock(in_channels, out_channels, wd=1e-4),
            nn.MaxPool1d(pool_size, ceil_mode=True),
            nn.Dropout(drop),
        )

    def init_thm_tof_branch(self, thm_dim, tof_dim, **kwargs):
        thm_branch = nn.Sequential(
            self.feature_block(
                thm_dim,
                kwargs["thm1_channels"],
                kwargs["thm1_layers"],
                drop=kwargs["thm1_dropout"],
            ),
            self.feature_block(
                kwargs["thm1_channels"],
                kwargs["thm2_channels"],
                kwargs["thm2_layers"],
                drop=kwargs["thm2_dropout"],
            ),
        )
        tof_branch = nn.Sequential(
            self.feature_block(
                tof_dim,
                kwargs["tof1_channels"],
                kwargs["tof1_layers"],
                drop=kwargs["tof1_dropout"],
            ),
            self.feature_block(
                kwargs["tof1_channels"],
                kwargs["tof2_channels"],
                kwargs["tof2_layers"],
                drop=kwargs["tof2_dropout"],
            ),
        )
        return thm_branch, tof_branch

    def forward(self, imus, thms, tofs):
        imu_acc, imu_rot, imu_other = imus
        imu_acc_feat = self.imu_acc_branch(imu_acc.permute(0, 2, 1))
        imu_rot_feat = self.imu_rot_branch(imu_rot.permute(0, 2, 1))
        imu_other_feat = self.imu_other_branch(imu_other.permute(0, 2, 1))
        imu_feat = self.imu_proj(
            torch.cat([imu_acc_feat, imu_rot_feat, imu_other_feat], dim=1)
        )

        thm1, thm2, thm3, thm4, thm5 = thms
        tof1, tof2, tof3, tof4, tof5 = tofs

        thm1_feat = self.thm_branch1(thm1.permute(0, 2, 1))
        thm2_feat = self.thm_branch2(thm2.permute(0, 2, 1))
        thm3_feat = self.thm_branch3(thm3.permute(0, 2, 1))
        thm4_feat = self.thm_branch4(thm4.permute(0, 2, 1))
        thm5_feat = self.thm_branch5(thm5.permute(0, 2, 1))
        thm_feat = self.thm_proj(
            torch.cat([thm1_feat, thm2_feat, thm3_feat, thm4_feat, thm5_feat], dim=1)
        )

        tof1_feat = self.tof_branch1(tof1.permute(0, 2, 1))
        tof2_feat = self.tof_branch2(tof2.permute(0, 2, 1))
        tof3_feat = self.tof_branch3(tof3.permute(0, 2, 1))
        tof4_feat = self.tof_branch4(tof4.permute(0, 2, 1))
        tof5_feat = self.tof_branch5(tof5.permute(0, 2, 1))
        tof_feat = self.tof_proj(
            torch.cat([tof1_feat, tof2_feat, tof3_feat, tof4_feat, tof5_feat], dim=1)
        )

        feat = torch.cat([imu_feat, thm_feat, tof_feat], dim=1).permute(0, 2, 1)
        lstm_out, _ = self.lstm(feat)
        gru_out, _ = self.gru(feat)
        dense_out = self.dense(self.noise(feat))

        return self.attn(torch.cat([lstm_out, gru_out, dense_out], dim=-1))


class CMIModel(nn.Module):
    def __init__(
        self,
        imu_dim,
        thm_dim,
        tof_dim,
        target_classes_num,
        non_target_classes_num,
        **kwargs,
    ):
        super().__init__()
        self.backbone = CMIBackbone(imu_dim, thm_dim, tof_dim, **kwargs)
        self.target_classifier = nn.Sequential(
            nn.Linear(
                (kwargs["lstm_hidden_size"] + kwargs["gru_hidden_size"]) * 2
                + kwargs["dense_channels"],
                kwargs["cls_channels1"],
            ),
            nn.BatchNorm1d(kwargs["cls_channels1"]),
            nn.ReLU(),
            nn.Dropout(kwargs["cls_dropout1"]),
            nn.Linear(kwargs["cls_channels1"], kwargs["cls_channels2"]),
            nn.BatchNorm1d(kwargs["cls_channels2"]),
            nn.ReLU(),
            nn.Dropout(kwargs["cls_dropout2"]),
            nn.Linear(kwargs["cls_channels2"], target_classes_num),
        )
        self.non_target_classifier = nn.Sequential(
            nn.Linear(
                (kwargs["lstm_hidden_size"] + kwargs["gru_hidden_size"]) * 2
                + kwargs["dense_channels"],
                kwargs["cls_channels1"],
            ),
            nn.BatchNorm1d(kwargs["cls_channels1"]),
            nn.ReLU(),
            nn.Dropout(kwargs["cls_dropout1"]),
            nn.Linear(kwargs["cls_channels1"], kwargs["cls_channels2"]),
            nn.BatchNorm1d(kwargs["cls_channels2"]),
            nn.ReLU(),
            nn.Dropout(kwargs["cls_dropout2"]),
            nn.Linear(kwargs["cls_channels2"], non_target_classes_num),
        )

    def forward(self, imu, thm, tof):
        feat = self.backbone(imu, thm, tof)
        targets_y = self.target_classifier(feat)
        non_targets_y = self.non_target_classifier(feat)
        return torch.cat([targets_y, non_targets_y], dim=1)


model_function = CMIModel
model_args = {
    "imu1_channels": 128,
    "imu2_channels": 256,
    "imu1_dropout": 0.3,
    "imu2_dropout": 0.25,
    "imu1_layers": 0,
    "imu2_layers": 0,
    "thm1_channels": 32,
    "thm2_channels": 64,
    "thm1_dropout": 0.25,
    "thm2_dropout": 0.2,
    "thm1_layers": 0,
    "thm2_layers": 0,
    "tof1_channels": 256,
    "tof2_channels": 512,
    "tof1_dropout": 0.4,
    "tof2_dropout": 0.3,
    "tof1_layers": 0,
    "tof2_layers": 0,
    "lstm_hidden_size": 128,
    "gru_hidden_size": 128,
    "gaussian_noise_rate": 0.1,
    "dense_channels": 32,
    "cls_channels1": 256,
    "cls_dropout1": 0.2,
    "cls_channels2": 128,
    "cls_dropout2": 0.2,
    "target_classes_num": 8,
    "non_target_classes_num": 10,
}
