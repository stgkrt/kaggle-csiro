import timm
import torch
from torch import nn


class SimpleCloverDiffModel(nn.Module):
    def __init__(
        self, backbone_name: str, pretrained: bool, in_channels: int, n_classes: int
    ):
        super(SimpleCloverDiffModel, self).__init__()
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
        # 1 ,2のsumを3chから引いたものを2ch目とmean
        output_green_dead = output[:, 1:3].sum(dim=1)
        output[:, 2] = (output[:, 2] + (output[:, 3] - output_green_dead)) * 0.5
        output = self.activateion(output)

        output = {"logits": output}
        return output


if __name__ == "__main__":
    batch_size = 4
    model = SimpleCloverDiffModel(
        backbone_name="tf_efficientnet_b0",
        pretrained=True,
        in_channels=3,
        n_classes=18,
    )

    sample_input = {"image": torch.randn(batch_size, 3, 224, 224)}
    output = model(sample_input)
    print(output["logits"].shape)  # Expected output shape: (4, 18)
