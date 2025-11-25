import torch
from torch import nn

from configs import LossConfig


class LossModule(nn.Module):
    def __init__(self, loss_config: LossConfig):
        super(LossModule, self).__init__()
        self.config = loss_config
        self.loss_name = loss_config.loss_name
        self.loss = self._set_loss()

    def forward(
        self, inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        preds = inputs["logits"]
        labels = targets["labels"]
        loss = self.loss(preds, labels)
        return loss.mean()

    def _set_loss(self) -> nn.Module:
        print("loss name", self.loss_name)
        if self.loss_name == "mse_loss":
            loss: nn._Loss = nn.MSELoss()
        else:
            raise NotImplementedError
        return loss
