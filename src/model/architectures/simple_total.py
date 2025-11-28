import timm
import torch
from torch import nn


class SimpleTotalModel(nn.Module):
    def __init__(
        self, backbone_name: str, pretrained: bool, in_channels: int, n_classes: int
    ):
        super(SimpleTotalModel, self).__init__()
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=n_classes,
        )
        self.activateion = nn.ReLU()

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        img = input["image"]
        output = self.model(img)
        # 3chの出力を0, 1, 2 chのsumとのmeanにする
        output_total = output[:, 0:3].sum(dim=1)
        output[:, 3] = (output[:, 3] + output_total) * 0.5
        output = self.activateion(output)

        output = {"logits": output}
        return output


if __name__ == "__main__":
    batch_size = 4
    model = SimpleTotalModel(
        backbone_name="tf_efficientnet_b0",
        pretrained=True,
        in_channels=3,
        n_classes=18,
    )

    sample_input = {"image": torch.randn(batch_size, 3, 224, 224)}
    output = model(sample_input)
    print(output["logits"].shape)  # Expected output shape: (4, 18)
