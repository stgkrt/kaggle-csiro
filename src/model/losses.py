import torch
from configs import LossConfig
from torch import nn


class WeightedMSELoss(nn.Module):
    def __init__(self, weights: torch.Tensor, device: torch.device):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights
        self.weights = self.weights.to(device)

    def forward(
        self, inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        preds = inputs["logits"]
        labels = targets["labels"]
        loss = (preds - labels) ** 2
        weighted_loss = loss * self.weights
        return weighted_loss.mean()


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(
        self, inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        preds = inputs["logits"]
        labels = targets["labels"]
        loss = self.mse_loss(preds, labels)
        return loss.mean()


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def forward(
        self, inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        preds = inputs["logits"]
        labels = targets["labels"]
        loss = self.smooth_l1_loss(preds, labels)
        return loss.mean()


class HeightGHSSLoss(nn.Module):
    def __init__(self, device: torch.device, aux_weight: float = 0.3):
        super(HeightGHSSLoss, self).__init__()
        self.aux_weight = torch.tensor(aux_weight).to(device)
        self.mse_loss = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def forward(
        self, inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        preds = inputs["logits"]
        height_preds = inputs["height"]
        gshh_preds = inputs["gshh"]

        labels = targets["labels"]
        height_labels = targets["height"]
        gshh_labels = targets["gshh"]

        target_loss = self.smooth_l1_loss(preds, labels)
        aux_height_loss = self.mse_loss(height_preds, height_labels)
        aux_gshh_loss = self.mse_loss(gshh_preds, gshh_labels)

        loss = target_loss + self.aux_weight * (aux_height_loss + aux_gshh_loss)
        return loss.mean()


class LossModule(nn.Module):
    def __init__(self, loss_config: LossConfig):
        super(LossModule, self).__init__()
        self.config = loss_config
        self.loss_name = loss_config.loss_name
        self.loss = self._set_loss()

    def forward(
        self, inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        loss = self.loss(inputs, targets)
        return loss

    def _set_loss(self) -> nn.Module:
        print("loss name", self.loss_name)
        if self.loss_name == "mse_loss":
            loss: nn._Loss = MSELoss()
        elif self.loss_name == "smooth_l1":
            loss = SmoothL1Loss()
        elif self.loss_name == "weighted_mse":
            weights = torch.tensor(self.config.mse_weights)
            loss = WeightedMSELoss(weights=weights, device=self.config.device)
        elif self.loss_name == "height_gshh_loss":
            weights = torch.tensor(self.config.mse_weights)
            loss = HeightGHSSLoss(
                device=self.config.device,
                aux_weight=self.config.aux_weight,
            )
        else:
            raise NotImplementedError
        return loss
