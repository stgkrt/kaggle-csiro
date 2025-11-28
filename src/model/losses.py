import torch
from torch import nn

from configs import LossConfig


class WeightedMSELoss(nn.Module):
    def __init__(self, weights: torch.Tensor, device: torch.device):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights
        self.weights = self.weights.to(device)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = (inputs - targets) ** 2
        weighted_loss = loss * self.weights
        return weighted_loss.mean()


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
        elif self.loss_name == "smooth_l1":
            loss = nn.SmoothL1Loss()
        elif self.loss_name == "weighted_mse":
            weights = torch.tensor(self.config.mse_weights)
            loss = WeightedMSELoss(weights=weights, device=self.config.device)
        else:
            raise NotImplementedError
        return loss
