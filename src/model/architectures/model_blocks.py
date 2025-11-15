import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def time_sum(x):
    return torch.sum(x, dim=1)


def squeeze_last_axis(x):
    return x.squeeze(-1)  # or torch.squeeze(x, dim=-1)


def expand_last_axis(x):
    return x.unsqueeze(-1)  # or torch.unsqueeze(x, dim=-1)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze operation
        # For PyTorch (B, C, L), AdaptiveAvgPool1d(1) gives (B, C, 1)
        # Then squeeze to (B, C)
        b, c, _ = x.size()  # Get batch size and channels
        se = self.avg_pool(x).view(b, c)  # (B, C, 1) -> (B, C)

        # Excitation operation
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se).view(b, c, 1)  # Reshape to (B, C, 1) for multiplication

        return x * se  # Element-wise multiplication


class ResidualSECNNBlock(nn.Module):
    def __init__(
        self, in_filters, out_filters, kernel_size, pool_size=2, drop=0.3, wd=1e-4
    ):
        super(ResidualSECNNBlock, self).__init__()
        # Use a list to store sequential layers for repeated blocks
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_filters, out_filters, kernel_size, padding="same", bias=False),
            nn.BatchNorm1d(out_filters),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                out_filters, out_filters, kernel_size, padding="same", bias=False
            ),
            nn.BatchNorm1d(out_filters),
            nn.ReLU(inplace=True),
        )
        self.se_block = SEBlock(out_filters)  # Apply SE block

        # Shortcut connection
        self.shortcut_conv = None
        if in_filters != out_filters:
            self.shortcut_conv = nn.Sequential(
                nn.Conv1d(in_filters, out_filters, 1, padding="same", bias=False),
                nn.BatchNorm1d(out_filters),
            )

        self.dropout = nn.Dropout(drop)

        self.wd = wd

    def forward(self, x):
        shortcut = x

        x = self.conv_block(x)
        x = self.se_block(x)
        if self.shortcut_conv:
            shortcut = self.shortcut_conv(shortcut)

        x = x + shortcut  # Keras 'add'
        x = F.relu(x)  # Activation after addition

        x = self.dropout(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.score_dense = nn.Linear(input_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        score = self.score_dense(inputs)
        score = self.tanh(score)
        score = squeeze_last_axis(score)

        weights = F.softmax(score, dim=1)
        weights = expand_last_axis(weights)

        context = inputs * weights
        context = time_sum(context)

        return context
