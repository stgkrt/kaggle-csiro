import timm
import torch
from torch import nn


class CloverSumHeightNVDIModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        in_channels: int,
        n_classes: int,
        emb_dim: int = 32,
        dropout: float = 0.2,
        drop_path_rate: float = 0.3,
        head_connection_type: str = "direct",
    ):
        super(CloverSumHeightNVDIModel, self).__init__()
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        self.head_connection_type = head_connection_type
        self.target_emb = nn.Sequential(
            nn.Linear(self.model.num_features, emb_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
        )
        self.target_head = nn.Sequential(
            nn.Linear(emb_dim, n_classes - 2),
            nn.Softplus(),
        )
        self.clover_classification_head = nn.Sequential(
            nn.Linear(emb_dim, 1),
        )
        self.height_head = nn.Sequential(
            nn.Linear(emb_dim, 1),
            nn.Softplus(),
        )
        self.nvdi_head = nn.Sequential(
            nn.Linear(emb_dim, 1),
            nn.Softplus(),
        )

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        img = input["image"]
        emb = self.model(img)
        emb = self.target_emb(emb)
        output = self.target_head(emb)
        # gdmとtotalは和から求める
        # gdm = green + clover
        gdm = output[:, 0] + output[:, 2]
        # total = gdm + dead
        total = torch.sum(output, dim=1, keepdim=True)
        output = torch.cat(
            [output[:, 0:1], output[:, 1:2], output[:, 2:3], gdm.unsqueeze(1), total],
            dim=1,
        )
        # clover classification head
        clover_output = self.clover_classification_head(emb)
        height = self.height_head(emb)
        nvdi = self.nvdi_head(emb)

        output = {
            "logits": output,
            "include_clover_preds": clover_output,
            "height": height,
            "nvdi": nvdi,
        }
        return output


class CloverSumHeightNVDIFrozenModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        in_channels: int,
        n_classes: int,
        emb_dim: int = 32,
        dropout: float = 0.2,
        drop_path_rate: float = 0.3,
        head_connection_type: str = "direct",
    ):
        super(CloverSumHeightNVDIFrozenModel, self).__init__()
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        self.head_connection_type = head_connection_type
        self.target_emb = nn.Sequential(
            nn.Linear(self.model.num_features, emb_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
        )
        self.target_head = nn.Sequential(
            nn.Linear(emb_dim, n_classes - 2),
            nn.Softplus(),
        )
        self.clover_classification_head = nn.Sequential(
            nn.Linear(emb_dim, 1),
        )
        self.height_head = nn.Sequential(
            nn.Linear(emb_dim, 1),
            nn.Softplus(),
        )
        self.nvdi_head = nn.Sequential(
            nn.Linear(emb_dim, 1),
            nn.Softplus(),
        )

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        img = input["image"]
        with torch.no_grad():
            emb = self.model(img)
        emb = self.target_emb(emb)
        output = self.target_head(emb)
        # gdmとtotalは和から求める
        # gdm = green + clover
        gdm = output[:, 0] + output[:, 2]
        # total = gdm + dead
        total = torch.sum(output, dim=1, keepdim=True)
        output = torch.cat(
            [output[:, 0:1], output[:, 1:2], output[:, 2:3], gdm.unsqueeze(1), total],
            dim=1,
        )
        # clover classification head
        clover_output = self.clover_classification_head(emb)
        height = self.height_head(emb)
        nvdi = self.nvdi_head(emb)

        output = {
            "logits": output,
            "include_clover_preds": clover_output,
            "height": height,
            "nvdi": nvdi,
        }
        return output


if __name__ == "__main__":
    batch_size = 4
    model = CloverSumHeightNVDIModel(
        backbone_name="tf_efficientnet_b0",
        pretrained=True,
        in_channels=3,
        n_classes=5,
    )

    sample_input = {"image": torch.randn(batch_size, 3, 224, 224)}
    output = model(sample_input)
    print(output["logits"].shape)  # Expected output shape: (4, 5)
    print(output["include_clover_preds"].shape)  # Expected output shape: (4, 1)
