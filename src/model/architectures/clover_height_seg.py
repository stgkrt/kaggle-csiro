import timm
import torch
from torch import nn

from src.model.architectures.model_blocks import DecoderBlock


class CloverHeightSegModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        in_channels: int,
        n_classes: int,
        emb_dim: int = 32,
        dropout: float = 0.2,
        drop_path_rate: float = 0.3,
        segmentation_depth: int = 1,
    ):
        super(CloverHeightSegModel, self).__init__()
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
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.AdaptiveAvgPool2d(1),
                )
            )
        use_features = 2  # 最初と最後以外
        self.target_head = nn.Sequential(
            nn.Linear(emb_dim * use_features, n_classes - 2),
            # nn.ReLU(),
            nn.Softplus(),
        )
        self.clover_classification_head = nn.Sequential(
            nn.Linear(emb_dim * use_features, 1),
        )
        self.height_head = nn.Sequential(
            nn.Linear(emb_dim * use_features, 1),
            # nn.ReLU(),
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
        pyramid_features = []
        for i, feature in enumerate(features[:-1]):
            if i not in [0, len(features) - 2]:
                continue  # skip first and last feature
            x = self.feature_extractor[i](feature)  # (batch, emb_dim, 1, 1)
            x = x.view(x.size(0), -1)  # (batch, emb_dim)
            pyramid_features.append(x)
        # concated pyramid features = (batch, emb_dim * num_features)
        pyramid_features = torch.cat(pyramid_features, dim=1)
        output = self.target_head(pyramid_features)
        # gdmとtotalは和から求める
        # gdm = green + clover
        gdm = output[:, 0] + output[:, 2]
        # total = gdm + dead
        total = torch.sum(output, dim=1, keepdim=True)
        output = torch.cat(
            [output[:, 0:1], output[:, 1:2], output[:, 2:3], gdm.unsqueeze(1), total],
            dim=1,
        )
        clover_output = self.clover_classification_head(pyramid_features)
        height = self.height_head(pyramid_features)
        # segmentation head forward seg_input_shape
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
            "segmentation": segmentation_output,
        }
        return output


class CloverHeightSegSimpleModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        in_channels: int,
        n_classes: int,
        emb_dim: int = 32,
        dropout: float = 0.2,
        drop_path_rate: float = 0.3,
        segmentation_depth: int = 1,
    ):
        super(CloverHeightSegSimpleModel, self).__init__()
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            drop_path_rate=drop_path_rate,
            features_only=True,
        )

        self.encoder_out_channels = self.model.feature_info.channels()
        self.target_emb = nn.Sequential(
            nn.Conv2d(self.encoder_out_channels[-1], emb_dim, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.target_head = nn.Sequential(
            nn.Linear(emb_dim, n_classes - 2),
            # nn.ReLU(),
            nn.Softplus(),
        )
        self.clover_classification_head = nn.Sequential(
            nn.Linear(emb_dim, 1),
        )
        self.height_head = nn.Sequential(
            nn.Linear(emb_dim, 1),
            # nn.ReLU(),
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
        # concated pyramid features = (batch, emb_dim * num_features)
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
        clover_output = self.clover_classification_head(emb)
        height = self.height_head(emb)
        # segmentation head forward seg_input_shape
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
            "segmentation": segmentation_output,
        }
        return output


if __name__ == "__main__":
    batch_size = 4
    # model = CloverHeightSegSimpleModel(
    #     backbone_name="tf_efficientnet_b0",
    #     pretrained=True,
    #     in_channels=3,
    #     n_classes=5,
    #     segmentation_depth=4,
    # )
    model = CloverHeightSegModel(
        backbone_name="tf_efficientnet_b0",
        pretrained=True,
        in_channels=3,
        n_classes=5,
        segmentation_depth=4,
    )
    sample_input = {"image": torch.randn(batch_size, 3, 512, 512)}
    output = model(sample_input)
    print(output["logits"].shape)  # Expected output shape: (4, 5)
    print(output["include_clover_preds"].shape)  # Expected output shape: (4, 1)
    # shape is depends on segmentation depth
    print(output["segmentation"].shape)
