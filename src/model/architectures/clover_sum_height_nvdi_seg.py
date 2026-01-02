import timm
import torch
from torch import nn

from src.model.architectures.model_blocks import DecoderBlock


class CloverSumHeightNVDISegModel(nn.Module):
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
        segmentation_depth: int = 4,
    ):
        super(CloverSumHeightNVDISegModel, self).__init__()
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            drop_path_rate=drop_path_rate,
            features_only=True,
        )
        self.encoder_out_channels = self.model.feature_info.channels()
        self.segmentation_depth = segmentation_depth
        self.head_connection_type = head_connection_type
        self.target_emb = nn.Sequential(
            nn.Conv2d(self.encoder_out_channels[-1], emb_dim, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
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
        # segmentation head (下の層のfeatureをconcatしてup-sampleしていく)
        self.segmentation_depth = segmentation_depth
        self.segmentation_head = nn.ModuleList()
        seg_in_channels = self.encoder_out_channels[-1]
        for i in range(segmentation_depth):
            seg_out_channels = self.encoder_out_channels[-(i + 2)]
            self.segmentation_head.append(
                DecoderBlock(seg_in_channels, seg_out_channels),
            )
            seg_in_channels = seg_out_channels + self.encoder_out_channels[-(i + 2)]

        self.segmentation_head.append(nn.Conv2d(seg_in_channels, 1, kernel_size=1))

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        img = input["image"]
        features = self.model(img)
        emb = features[-1]
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
        # segmentation
        seg_input = features[-1]
        # u-net like upsampling with skip connections
        for i in range(self.segmentation_depth):
            output_feature = self.segmentation_head[i](seg_input)
            seg_input = torch.cat([output_feature, features[-(i + 2)]], dim=1)

        segmentation_output = self.segmentation_head[-1](seg_input)
        output = {
            "logits": output,
            "include_clover_preds": clover_output,
            "height": height,
            "nvdi": nvdi,
            "segmentation": segmentation_output,
        }
        return output


if __name__ == "__main__":
    batch_size = 4
    model = CloverSumHeightNVDISegModel(
        backbone_name="tf_efficientnet_b0",
        pretrained=True,
        in_channels=3,
        n_classes=5,
    )

    sample_input = {"image": torch.randn(batch_size, 3, 224, 224)}
    output = model(sample_input)
    print(output["logits"].shape)  # Expected output shape: (4, 5)
    print(output["include_clover_preds"].shape)  # Expected output shape: (4, 1)
