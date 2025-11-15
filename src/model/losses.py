import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

from src.config_dataclass import ArgParseModelConfig


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = inputs.sigmoid()

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        if torch.isnan(dice):
            print("input", torch.isnan(inputs))
            print("target", torch.isnan(targets))
            print("intersection", torch.isnan(intersection))
            raise RuntimeError

        return 1 - dice


class WeightedCELoss(nn.Module):
    def __init__(
        self,
    ):
        super(WeightedCELoss, self).__init__()

    def forward(
        self, inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ):
        labels = torch.argmax(targets["labels"], dim=1)
        loss = F.cross_entropy(
            inputs["logits"],
            labels,
            reduction="none",
        )

        return loss.mean()


class AuxTargetCELoss(nn.Module):
    def __init__(self, target_idxs=None, aux_weight=0.2):
        super(AuxTargetCELoss, self).__init__()
        self.target_idxs = torch.tensor(target_idxs, dtype=torch.long)
        self.target_idxs = self.target_idxs.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.aux_weight = aux_weight

    def forward(
        self, inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ):
        labels = torch.argmax(targets["labels"], dim=1)
        class_loss = F.cross_entropy(
            inputs["logits"],
            labels,
            reduction="none",
        )

        labels_is_target = torch.isin(labels, self.target_idxs)
        inputs_binary = inputs["logits"].argmax(dim=1)
        inputs_binary = torch.isin(inputs_binary, self.target_idxs)
        is_target_loss = F.binary_cross_entropy_with_logits(
            inputs_binary.float(),
            labels_is_target.float(),
            reduction="none",
        )
        loss = class_loss.mean() + self.aux_weight * is_target_loss.mean()

        return loss


class AuxOrientBehaviorLoss(nn.Module):
    def __init__(self, aux_weight=0.2):
        super(AuxOrientBehaviorLoss, self).__init__()
        self.aux_weight = aux_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self, inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ):
        labels = torch.argmax(targets["labels"], dim=1)
        class_loss = F.cross_entropy(
            inputs["logits"],
            labels,
            reduction="none",
        )
        orient_loss = self.bce_loss(
            inputs["orientation"].float(), targets["orientation"].float()
        )
        behavior_loss = self.bce_loss(
            inputs["behavior"].float(), targets["behavior"].float()
        )
        loss = class_loss.mean() + self.aux_weight * (
            orient_loss.mean() + behavior_loss.mean()
        )
        return loss


class WeightedAuxOrientBehaviorLoss(nn.Module):
    def __init__(self, aux_weight=0.2, target_gesture_dict_path=None):
        super(WeightedAuxOrientBehaviorLoss, self).__init__()
        self.aux_weight = aux_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.target_gesture_dict = joblib.load(target_gesture_dict_path)
        self.target_gesture_list = [
            "Above ear - pull hair",
            "Forehead - pull hairline",
            "Forehead - scratch",
            "Eyebrow - pull hair",
            "Eyelash - pull hair",
            "Neck - scratch",
            "Neck - pinch skin",
            "Cheek - pinch skin",
        ]
        weight_value = 1.5
        # weight_value = 2.0
        class_num = 18
        self.class_weight = torch.zeros(class_num)
        for key, value in self.target_gesture_dict.items():
            if value in self.target_gesture_list:
                self.class_weight[int(key)] = weight_value
            else:
                self.class_weight[int(key)] = 1.0
        self.class_weight = self.class_weight.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def forward(
        self, inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ):
        labels = torch.argmax(targets["labels"], dim=1)
        class_loss = F.cross_entropy(
            inputs["logits"], labels, reduction="none", weight=self.class_weight
        )
        orient_loss = self.bce_loss(
            inputs["orientation"].float(), targets["orientation"].float()
        )
        behavior_loss = self.bce_loss(
            inputs["behavior"].float(), targets["behavior"].float()
        )
        loss = class_loss.mean() + self.aux_weight * (
            orient_loss.mean() + behavior_loss.mean()
        )
        return loss


class SplitAuxOrientBehaviorLoss(nn.Module):
    def __init__(self, aux_weight=0.2):
        super(SplitAuxOrientBehaviorLoss, self).__init__()
        self.aux_weight = aux_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self, inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ):
        labels = torch.argmax(targets["labels"], dim=1)
        class_loss = F.cross_entropy(
            inputs["logits"],
            labels,
            reduction="none",
        )
        target_gesture_label = targets["labels"][:, :8].sum(dim=1, keepdim=True)
        nontarget_gesture_label = targets["labels"][:, 8:].sum(dim=1, keepdim=True)
        target_gesture_pred = inputs["logits"][:, :8].sum(dim=1, keepdim=True)
        nontarget_gesture_pred = inputs["logits"][:, 8:].sum(dim=1, keepdim=True)
        replace_label = torch.cat(
            [target_gesture_label, nontarget_gesture_label], dim=1
        )
        replace_pred = torch.cat([target_gesture_pred, nontarget_gesture_pred], dim=1)
        binary_loss = self.bce_loss(replace_pred, replace_label)

        orient_loss = self.bce_loss(
            inputs["orientation"].float(), targets["orientation"].float()
        )
        behavior_loss = self.bce_loss(
            inputs["behavior"].float(), targets["behavior"].float()
        )
        loss = class_loss.mean() + self.aux_weight * (
            orient_loss.mean() + behavior_loss.mean() + binary_loss.mean()
        )
        return loss


class BCEAuxNonTargetLoss(nn.Module):
    def __init__(self, aux_weight=0.2, target_gesture_dict_path=None, pos_weight=None):
        super(BCEAuxNonTargetLoss, self).__init__()
        self.aux_weight = aux_weight
        self.pos_weight = (
            torch.tensor(pos_weight, dtype=torch.float)
            if pos_weight is not None
            else None
        )
        if self.pos_weight is not None:
            self.pos_weight = self.pos_weight.to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

        self.target_gesture_dict = joblib.load(target_gesture_dict_path)
        self.target_gesture_list = [
            "Above ear - pull hair",
            "Forehead - pull hairline",
            "Forehead - scratch",
            "Eyebrow - pull hair",
            "Eyelash - pull hair",
            "Neck - scratch",
            "Neck - pinch skin",
            "Cheek - pinch skin",
        ]
        self.write_leg = ["Write name on leg"]
        self.other_non_target_idx = []
        self.target_idxs = []
        self.write_leg_idx = []
        for key, value in self.target_gesture_dict.items():
            if value in self.write_leg:
                self.write_leg_idx.append(int(key))
            if value in self.target_gesture_list:
                self.target_idxs.append(int(key))
            else:
                self.other_non_target_idx.append(int(key))
        self.target_idxs = torch.tensor(self.target_idxs)
        self.write_leg_idx = torch.tensor(self.write_leg_idx)
        self.other_non_target_idx = torch.tensor(self.other_non_target_idx)
        self.target_idxs = self.target_idxs.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ).long()
        self.write_leg_idx = self.write_leg_idx.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ).long()
        self.other_non_target_idx = self.other_non_target_idx.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ).long()
        self.non_target_idxs = torch.concat(
            [self.other_non_target_idx, self.write_leg_idx]
        )
        self.target_write_idxs = torch.concat([self.target_idxs, self.write_leg_idx])

    def forward(
        self, inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ):
        labels = targets["labels"]
        pred = inputs["logits"]
        # pred = torch.sigmoid(pred)
        # pred_argmax = pred.argmax(dim=1)
        # labels = labels.argmax(dim=1)
        # basic_ce_loss = F.cross_entropy(pred, labels, reduction="none")

        # target_gesture_pred = pred[:, self.target_idxs]
        # non_target_gesture_pred = pred[:, self.non_target_idxs].sum(dim=1)
        # pred = torch.concat(
        #     [target_gesture_pred, non_target_gesture_pred.unsqueeze(1)], dim=1
        # )
        # target_gesture_label = labels[:, self.target_idxs]
        # non_target_gesture_label = labels[:, self.non_target_idxs].sum(dim=1)
        # label = torch.concat(
        #     [target_gesture_label, non_target_gesture_label.unsqueeze(1)], dim=1
        # )

        # class_loss = self.bce_loss(pred.float(), label.float())

        # ce loss

        target_and_write_pred = pred[:, self.target_write_idxs]
        other_non_target_pred = pred[:, self.other_non_target_idx].sum(dim=1)
        replace_pred = torch.concat(
            [target_and_write_pred, other_non_target_pred.unsqueeze(1)], dim=1
        )
        target_and_write_label = labels[:, self.target_write_idxs]
        other_non_target_label = labels[:, self.other_non_target_idx].sum(dim=1)
        replace_label = torch.concat(
            [target_and_write_label, other_non_target_label.unsqueeze(1)], dim=1
        )
        # replace_pred = replace_pred.argmax(dim=1)
        replace_label = replace_label.argmax(dim=1)
        class_loss = F.cross_entropy(
            replace_pred.float(),
            replace_label,
            reduction="none",
        )
        # target_pred = target_pred.argmax(dim=1)
        # target_label = target_label.argmax(dim=1)
        # target_loss = F.cross_entropy(
        #     target_pred.float(),
        #     target_label.float(),
        #     reduction="none",
        # )
        # other non-target gestures
        # other_non_target_pred = pred[:, self.other_non_target_idx].sum(dim=1)
        # write_leg_pred = pred[:, self.write_leg_idx].sum(dim=1)
        # other_non_target_label = labels[:, self.other_non_target_idx].sum(dim=1)
        # write_leg_label = labels[:, self.write_leg_idx].sum(dim=1)
        # other_non_target_loss = self.bce_loss(
        #     other_non_target_pred.float(), other_non_target_label.float()
        # )

        orient_loss = self.bce_loss(
            inputs["orientation"].float(), targets["orientation"].float()
        )
        behavior_loss = self.bce_loss(
            inputs["behavior"].float(), targets["behavior"].float()
        )
        # class_loss = (
        #     other_non_target_loss.mean() + target_loss.mean() + write_leg_loss.mean()
        # )
        # print("basic class loss", basic_class_loss.mean())
        # print("class loss", class_loss.mean())
        # print("orientation loss", orient_loss.mean())
        # print("behavior loss", behavior_loss.mean())
        loss = class_loss.mean() + self.aux_weight * (
            orient_loss.mean() + behavior_loss.mean()
        )
        return loss


class BCEAuxNonTargetLossV2(nn.Module):
    def __init__(self, aux_weight=0.2, target_gesture_dict_path=None, pos_weight=None):
        super(BCEAuxNonTargetLossV2, self).__init__()
        self.aux_weight = aux_weight
        self.pos_weight = (
            torch.tensor(pos_weight, dtype=torch.float)
            if pos_weight is not None
            else None
        )
        if self.pos_weight is not None:
            self.pos_weight = self.pos_weight.to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

        self.target_gesture_dict = joblib.load(target_gesture_dict_path)
        self.target_gesture_list = [
            "Above ear - pull hair",
            "Forehead - pull hairline",
            "Forehead - scratch",
            "Eyebrow - pull hair",
            "Eyelash - pull hair",
            "Neck - scratch",
            "Neck - pinch skin",
            "Cheek - pinch skin",
        ]
        self.write_name = ["Write name on leg", "Write name in air"]
        self.wave_hello = ["Wave hello"]
        self.other_non_target_idx = []
        self.target_gesture_idxs = []
        self.write_name_idx = []
        self.wave_hello_idx = []
        for key, value in self.target_gesture_dict.items():
            if value in self.write_name:
                self.write_name_idx.append(int(key))
            elif value in self.wave_hello:
                self.wave_hello_idx.append(int(key))
            elif value in self.target_gesture_list:
                self.target_gesture_idxs.append(int(key))
            else:
                self.other_non_target_idx.append(int(key))
        self.target_gesture_idxs = torch.tensor(self.target_gesture_idxs)
        self.write_name_idx = torch.tensor(self.write_name_idx)
        self.wave_hello_idx = torch.tensor(self.wave_hello_idx)
        self.other_non_target_idx = torch.tensor(self.other_non_target_idx)
        self.target_gesture_idxs = self.target_gesture_idxs.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ).long()
        self.write_name_idx = self.write_name_idx.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ).long()
        self.wave_hello_idx = self.wave_hello_idx.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ).long()
        self.other_non_target_idx = self.other_non_target_idx.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ).long()

    def forward(
        self, inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ):
        labels = targets["labels"]
        pred = inputs["logits"]

        target_gesture_pred = pred[:, self.target_gesture_idxs]
        write_gesture_pred = pred[:, self.write_name_idx].sum(dim=1)
        wave_hello_pred = pred[:, self.wave_hello_idx].sum(dim=1)
        other_non_target_pred = pred[:, self.other_non_target_idx].sum(dim=1)
        replace_pred = torch.concat(
            [
                target_gesture_pred,
                write_gesture_pred.unsqueeze(1),
                wave_hello_pred.unsqueeze(1),
                other_non_target_pred.unsqueeze(1),
            ],
            dim=1,
        )
        target_gesture_label = labels[:, self.target_gesture_idxs]
        write_gesture_label = labels[:, self.write_name_idx].sum(dim=1)
        wave_hello_label = labels[:, self.wave_hello_idx].sum(dim=1)
        other_non_target_label = labels[:, self.other_non_target_idx].sum(dim=1)
        replace_label = torch.concat(
            [
                target_gesture_label,
                write_gesture_label.unsqueeze(1),
                wave_hello_label.unsqueeze(1),
                other_non_target_label.unsqueeze(1),
            ],
            dim=1,
        )
        # replace_pred = replace_pred.argmax(dim=1)
        replace_label = replace_label.argmax(dim=1)
        class_loss = F.cross_entropy(
            replace_pred.float(),
            replace_label,
            reduction="none",
        )
        orient_loss = self.bce_loss(
            inputs["orientation"].float(), targets["orientation"].float()
        )
        behavior_loss = self.bce_loss(
            inputs["behavior"].float(), targets["behavior"].float()
        )
        # class_loss = (
        #     other_non_target_loss.mean() + target_loss.mean() + write_leg_loss.mean()
        # )
        # print("basic class loss", basic_class_loss.mean())
        # print("class loss", class_loss.mean())
        # print("orientation loss", orient_loss.mean())
        # print("behavior loss", behavior_loss.mean())
        loss = class_loss.mean() + self.aux_weight * (
            orient_loss.mean() + behavior_loss.mean()
        )
        return loss


class AuxGestureTypeLoss(nn.Module):
    def __init__(self, aux_weight=0.2, target_gesture_dict_path=None):
        super(AuxGestureTypeLoss, self).__init__()
        print("AuxGestureTypeLoss init, aux_weight", aux_weight)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.aux_weight = aux_weight
        self.target_gesture_dict = joblib.load(target_gesture_dict_path)
        self.target_pull_list = [
            "Above ear - pull hair",
            "Forehead - pull hairline",
            "Eyebrow - pull hair",
            "Eyelash - pull hair",
        ]
        self.target_scratch_list = [
            "Forehead - scratch",
            "Neck - scratch",
        ]
        self.target_pinch_list = [
            "Neck - pinch skin",
            "Cheek - pinch skin",
        ]
        self.target_pull_idxs = []
        self.target_scratch_idxs = []
        self.target_pinch_idxs = []
        self.non_target_idxs = []
        for idx, gesture in self.target_gesture_dict.items():
            if gesture in self.target_pull_list:
                self.target_pull_idxs.append(idx)
            elif gesture in self.target_scratch_list:
                self.target_scratch_idxs.append(idx)
            elif gesture in self.target_pinch_list:
                self.target_pinch_idxs.append(idx)
            else:
                self.non_target_idxs.append(idx)

        self.target_pull_idxs = torch.tensor(
            self.target_pull_idxs, dtype=torch.long
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.target_scratch_idxs = torch.tensor(
            self.target_scratch_idxs, dtype=torch.long
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.target_pinch_idxs = torch.tensor(
            self.target_pinch_idxs, dtype=torch.long
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.non_target_idxs = torch.tensor(self.non_target_idxs, dtype=torch.long).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def forward(
        self, inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ):
        labels = torch.argmax(targets["labels"], dim=1)
        class_loss = F.cross_entropy(
            inputs["logits"],
            labels,
            reduction="none",
        )
        confidence = inputs["logits"].softmax(dim=1)
        pred_target_pull = confidence[:, self.target_pull_idxs].sum(dim=1)
        pred_target_scratch = confidence[:, self.target_scratch_idxs].sum(dim=1)
        pred_target_pinch = confidence[:, self.target_pinch_idxs].sum(dim=1)
        pred_non_target = confidence[:, self.non_target_idxs].sum(dim=1)

        labels_is_target_pull = torch.isin(labels, self.target_pull_idxs)
        labels_is_target_scratch = torch.isin(labels, self.target_scratch_idxs)
        labels_is_target_pinch = torch.isin(labels, self.target_pinch_idxs)
        labels_is_non_target = torch.isin(labels, self.non_target_idxs)

        is_target_pull_loss = self.bce_loss(
            pred_target_pull.float(),
            labels_is_target_pull.float(),
        )
        is_target_scratch_loss = self.bce_loss(
            pred_target_scratch.float(),
            labels_is_target_scratch.float(),
        )
        is_target_pinch_loss = self.bce_loss(
            pred_target_pinch.float(),
            labels_is_target_pinch.float(),
        )
        is_non_target_loss = self.bce_loss(
            pred_non_target.float(),
            labels_is_non_target.float(),
        )
        aux_loss = (
            is_target_pull_loss.mean()
            + is_target_scratch_loss.mean()
            + is_target_pinch_loss.mean()
            + is_non_target_loss.mean()
        )
        loss = class_loss.mean() + self.aux_weight * aux_loss

        return loss


class SoftTargetBCELoss(nn.Module):
    def __init__(self, target_gesture_dict_path=None, smoothing=0.1, pos_weight=None):
        super(SoftTargetBCELoss, self).__init__()
        self.pos_weight = (
            torch.tensor(pos_weight, dtype=torch.float)
            if pos_weight is not None
            else None
        )
        self.bce_loss = nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=self.pos_weight
        )
        self.smoothing = smoothing
        self.target_gesture_dict = joblib.load(target_gesture_dict_path)
        self.target_gesture_list = [
            "Above ear - pull hair",
            "Forehead - pull hairline",
            "Forehead - scratch",
            "Eyebrow - pull hair",
            "Eyelash - pull hair",
            "Neck - scratch",
            "Neck - pinch skin",
            "Cheek - pinch skin",
        ]
        self.target_one_hot = torch.zeros(
            len(self.target_gesture_dict), dtype=torch.float
        )
        self.non_target_one_hot = torch.zeros(
            len(self.target_gesture_dict), dtype=torch.float
        )
        for idx, gesture in self.target_gesture_dict.items():
            if gesture in self.target_gesture_list:
                self.target_one_hot[idx] = 1.0
            else:
                self.non_target_one_hot[idx] = 1.0
        self.target_one_hot = self.target_one_hot.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.non_target_one_hot = self.non_target_one_hot.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def forward(
        self, inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ):
        logtits = inputs["logits"]
        labels = targets["labels"]
        prediction = logtits.sigmoid()
        # labelがtargetにあるとき、ほかのtargetのlabelをsmoothingの値に設定し、
        # non_targetのときもほかのnon_targetのlabelをsmoothingの値に設定
        labels_soft = torch.zeros_like(prediction)
        isin_target = (labels * self.target_one_hot).sum(dim=1, keepdim=True)
        isin_non_target = (labels * self.non_target_one_hot).sum(dim=1, keepdim=True)

        target_one_hot = self.target_one_hot * isin_target
        non_target_one_hot = self.non_target_one_hot * isin_non_target

        labels_soft = torch.where(
            target_one_hot > 0,
            torch.ones_like(labels_soft) * self.smoothing,
            torch.zeros_like(labels_soft),
        )
        labels_soft = torch.where(
            non_target_one_hot > 0,
            torch.ones_like(labels_soft) * self.smoothing,
            labels_soft,
        )
        labels_soft = torch.where(
            labels > 0,
            torch.ones_like(labels_soft) * (1 - self.smoothing),
            labels_soft,
        )
        loss = self.bce_loss(
            logtits,
            labels_soft,
        )

        return loss.mean()


class BCEOrientBehaviorAuxLoss(nn.Module):
    def __init__(self, aux_weight=0.2, target_gesture_dict_path=None, pos_weight=None):
        super(BCEOrientBehaviorAuxLoss, self).__init__()
        self.aux_weight = aux_weight
        self.pos_weight = (
            torch.tensor(pos_weight, dtype=torch.float)
            if pos_weight is not None
            else None
        )
        if self.pos_weight is not None:
            self.pos_weight = self.pos_weight.to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

        self.class_bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self, inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ):
        class_loss = self.class_bce(inputs["logits"].float(), targets["labels"].float())
        orient_loss = self.bce_loss(
            inputs["orientation"].float(), targets["orientation"].float()
        )
        behavior_loss = self.bce_loss(
            inputs["behavior"].float(), targets["behavior"].float()
        )
        loss = class_loss.mean() + self.aux_weight * (
            orient_loss.mean() + behavior_loss.mean()
        )

        return loss


class LossModule(nn.Module):
    def __init__(self, loss_config: ArgParseModelConfig):
        super(LossModule, self).__init__()
        self.config = loss_config
        self.loss_name = loss_config.loss_name
        self.loss = self._set_loss()

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)

    def _set_loss(self) -> nn.Module:
        print("loss name", self.loss_name)
        if self.loss_name == "weighted_cross_entropy":
            loss: nn._Loss = WeightedCELoss()
        elif self.loss_name == "aux_target_ce":
            target_idxs_dict = joblib.load(self.config.target_gesture_dict_path)
            target_idxs = target_idxs_dict["target"]
            loss = AuxTargetCELoss(
                target_idxs=target_idxs, aux_weight=self.config.aux_weight
            )
        elif self.loss_name == "aux_gesture_type":
            loss = AuxGestureTypeLoss(
                aux_weight=self.config.aux_weight,
                target_gesture_dict_path=self.config.target_gesture_dict_path,
            )
        elif self.loss_name == "soft_target_bce":
            loss = SoftTargetBCELoss(
                target_gesture_dict_path=self.config.target_gesture_dict_path,
                smoothing=0.1,
                pos_weight=self.config.pos_weight,
            )
        elif self.loss_name == "aux_orient_behavior":
            loss = AuxOrientBehaviorLoss(aux_weight=self.config.aux_weight)
        elif self.loss_name == "bce_orient_behavior":
            loss = BCEOrientBehaviorAuxLoss(
                aux_weight=self.config.aux_weight,
                target_gesture_dict_path=self.config.target_gesture_dict_path,
                pos_weight=self.config.pos_weight,
            )
        elif self.loss_name == "bce_aux_non_target":
            loss = BCEAuxNonTargetLoss(
                aux_weight=self.config.aux_weight,
                target_gesture_dict_path=self.config.target_gesture_dict_path,
                pos_weight=self.config.pos_weight,
            )
        elif self.loss_name == "bce_aux_non_target_v2":
            loss = BCEAuxNonTargetLossV2(
                aux_weight=self.config.aux_weight,
                target_gesture_dict_path=self.config.target_gesture_dict_path,
                pos_weight=self.config.pos_weight,
            )
        elif self.loss_name == "split_aux":
            loss = SplitAuxOrientBehaviorLoss(aux_weight=self.config.aux_weight)
        elif self.loss_name == "weighted_aux_orient_behavior":
            loss = WeightedAuxOrientBehaviorLoss(
                aux_weight=self.config.aux_weight,
                target_gesture_dict_path=self.config.target_gesture_dict_path,
            )
        else:
            raise NotImplementedError
        return loss


if __name__ == "__main__":
    from pathlib import Path

    inv_target_gesture_dict = joblib.load(
        "/kaggle/working/encoders/inverse_gesture_dict.pkl"
    )

    # loss_name = "weighted_cross_entropy"  # Change this to test different losses
    # loss_name = "aux_target_ce"  # Change this to test different losses
    # loss_name = "aux_gesture_type"  # Change this to test different losses
    # loss_name = "soft_target_bce"  # Change this to test different losses
    # loss_name = "aux_orient_behavior"  # Change this to test different losses
    # loss_name = "bce_orient_behavior"
    # loss_name = "bce_aux_non_target"
    # loss_name = "split_aux"
    loss_name = "weighted_aux_orient_behavior"

    if loss_name == "weighted_cross_entropy":
        config = ArgParseModelConfig(
            loss_name="weighted_cross_entropy",
            pos_weight=None,
            target_gesture_dict_path=None,
            aux_weight=None,
        )
        loss = LossModule(config)
        pred = torch.randn(34, 18)  # Example predictions

        # target = torch.randn(34, 18)
        target = torch.randint(0, 18, (34,))  # Example targets
        target = F.one_hot(
            target, num_classes=18
        ).float()  # Convert to one-hot encoding

        # target = torch.randint(0, 18, (34,))  # Example targets
        inputs = {"logits": pred}
        targets = {"labels": target}
    elif loss_name == "aux_target_ce":
        config = ArgParseModelConfig(
            loss_name="aux_target_ce",
            pos_weight=None,
            target_gesture_dict_path=Path(
                "/kaggle/working/encoders/target_non_target_gesture_le.pkl"
            ),
            aux_weight=0.2,
        )
        loss = LossModule(config)
        pred = torch.randn(34, 18)
        target = torch.randint(0, 18, (34,))
        target = F.one_hot(
            target, num_classes=18
        ).float()  # Convert to one-hot encoding
        inputs = {"logits": pred}
        targets = {"labels": target}
        inputs["logits"] = inputs["logits"].to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        targets["labels"] = targets["labels"].to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    elif loss_name == "aux_gesture_type":
        config = ArgParseModelConfig(
            loss_name="aux_gesture_type",
            pos_weight=None,
            target_gesture_dict_path=Path(
                "/kaggle/working/encoders/inverse_gesture_dict.pkl"
            ),
            aux_weight=0.2,
        )
        loss = LossModule(config)
        pred = torch.randn(34, 18)
        target = torch.randint(0, 18, (34,))
        target = F.one_hot(target, num_classes=18).float()
        inputs = {"logits": pred}
        targets = {"labels": target}
        inputs["logits"] = inputs["logits"].to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        targets["labels"] = targets["labels"].to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    elif loss_name == "soft_target_bce":
        config = ArgParseModelConfig(
            loss_name="soft_target_bce",
            pos_weight=None,
            target_gesture_dict_path=Path(
                "/kaggle/working/encoders/inverse_gesture_dict.pkl"
            ),
            smoothing=0.1,
        )
        loss = LossModule(config)
        pred = torch.randn(34, 18)
        target = torch.randint(0, 18, (34,))
        target = F.one_hot(target, num_classes=18).float()
        inputs = {"logits": pred}
        targets = {"labels": target}
        inputs["logits"] = inputs["logits"].to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        targets["labels"] = targets["labels"].to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    elif loss_name == "aux_orient_behavior":
        config = ArgParseModelConfig(
            loss_name="aux_orient_behavior",
            pos_weight=None,
            target_gesture_dict_path=Path(
                "/kaggle/working/encoders/inverse_gesture_dict.pkl"
            ),
            aux_weight=0.2,
        )
        loss = LossModule(config)
        pred = {
            "logits": torch.randn(34, 18),
            "orientation": torch.randn(34, 4),
            "behavior": torch.randn(34, 4),
        }
        target = torch.randint(0, 18, (34,))
        target = F.one_hot(target, num_classes=18).float()
        orient_target = torch.randint(0, 4, (34,))
        behavior_target = torch.randint(0, 4, (34,))
        orient_target = F.one_hot(orient_target, num_classes=4).float()
        behavior_target = F.one_hot(behavior_target, num_classes=4).float()
        inputs = {
            "logits": pred["logits"],
            "orientation": pred["orientation"],
            "behavior": pred["behavior"],
        }
        targets = {
            "labels": target,
            "orientation": orient_target,
            "behavior": behavior_target,
        }
    elif loss_name == "bce_orient_behavior":
        config = ArgParseModelConfig(
            loss_name="bce_orient_behavior",
            pos_weight=None,
            target_gesture_dict_path=Path(
                "/kaggle/working/encoders/inverse_gesture_dict.pkl"
            ),
        )
        loss = LossModule(config)
        pred = {
            "logits": torch.randn(34, 18),
            "orientation": torch.randn(34, 4),
            "behavior": torch.randn(34, 4),
        }
        target = torch.randint(0, 18, (34,))
        target = F.one_hot(target, num_classes=18).float()
        orient_target = torch.randint(0, 4, (34,))
        behavior_target = torch.randint(0, 4, (34,))
        orient_target = F.one_hot(orient_target, num_classes=4).float()
        behavior_target = F.one_hot(behavior_target, num_classes=4).float()
        pred["logits"] = pred["logits"].to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        pred["orientation"] = pred["orientation"].to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        pred["behavior"] = pred["behavior"].to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        target = target.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        orient_target = orient_target.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        behavior_target = behavior_target.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        inputs = {
            "logits": pred["logits"],
            "orientation": pred["orientation"],
            "behavior": pred["behavior"],
        }
        targets = {
            "labels": target,
            "orientation": orient_target,
            "behavior": behavior_target,
        }
    elif loss_name == "bce_aux_non_target":
        config = ArgParseModelConfig(
            loss_name="bce_aux_non_target",
            pos_weight=None,
            target_gesture_dict_path=Path(
                "/kaggle/working/encoders/inverse_gesture_dict.pkl"
            ),
            aux_weight=0.2,
        )
        loss = LossModule(config)
        pred = {
            "logits": torch.randn(34, 18),
            "orientation": torch.randn(34, 4),
            "behavior": torch.randn(34, 4),
        }
        target = torch.randint(0, 18, (34,))
        target = F.one_hot(target, num_classes=18).float()
        orient_target = torch.randint(0, 4, (34,))
        behavior_target = torch.randint(0, 4, (34,))
        orient_target = F.one_hot(orient_target, num_classes=4).float()
        behavior_target = F.one_hot(behavior_target, num_classes=4).float()
        pred["logits"] = pred["logits"].to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        pred["orientation"] = pred["orientation"].to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        pred["behavior"] = pred["behavior"].to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        inputs = {
            "logits": pred["logits"],
            "orientation": pred["orientation"],
            "behavior": pred["behavior"],
        }
        target = target.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        orient_target = orient_target.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        behavior_target = behavior_target.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        targets = {
            "labels": target,
            "orientation": orient_target,
            "behavior": behavior_target,
        }
    elif loss_name == "bce_aux_non_target_v2":
        config = ArgParseModelConfig(
            loss_name="bce_aux_non_target_v2",
            pos_weight=None,
            target_gesture_dict_path=Path(
                "/kaggle/working/encoders/inverse_gesture_dict.pkl"
            ),
            aux_weight=0.2,
        )
        loss = LossModule(config)
        pred = {
            "logits": torch.randn(34, 18),
            "orientation": torch.randn(34, 4),
            "behavior": torch.randn(34, 4),
        }
        target = torch.randint(0, 18, (34,))
        target = F.one_hot(target, num_classes=18).float()
        orient_target = torch.randint(0, 4, (34,))
        behavior_target = torch.randint(0, 4, (34,))
        orient_target = F.one_hot(orient_target, num_classes=4).float()
        behavior_target = F.one_hot(behavior_target, num_classes=4).float()
        pred["logits"] = pred["logits"].to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        pred["orientation"] = pred["orientation"].to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        pred["behavior"] = pred["behavior"].to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        inputs = {
            "logits": pred["logits"],
            "orientation": pred["orientation"],
            "behavior": pred["behavior"],
        }
        target = target.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        orient_target = orient_target.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        behavior_target = behavior_target.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        targets = {
            "labels": target,
            "orientation": orient_target,
            "behavior": behavior_target,
        }
    elif loss_name == "split_aux":
        config = ArgParseModelConfig(
            loss_name="split_aux",
            pos_weight=None,
            aux_weight=0.2,
        )
        loss = LossModule(config)
        pred = {
            "logits": torch.randn(34, 18),
            "orientation": torch.randn(34, 4),
            "behavior": torch.randn(34, 4),
        }
        target = torch.randint(0, 18, (34,))
        target = F.one_hot(target, num_classes=18).float()
        orient_target = torch.randint(0, 4, (34,))
        behavior_target = torch.randint(0, 4, (34,))
        orient_target = F.one_hot(orient_target, num_classes=4).float()
        behavior_target = F.one_hot(behavior_target, num_classes=4).float()
        pred["logits"] = pred["logits"].to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        pred["orientation"] = pred["orientation"].to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        pred["behavior"] = pred["behavior"].to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        inputs = {
            "logits": pred["logits"],
            "orientation": pred["orientation"],
            "behavior": pred["behavior"],
        }
        target = target.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        orient_target = orient_target.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        behavior_target = behavior_target.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        targets = {
            "labels": target,
            "orientation": orient_target,
            "behavior": behavior_target,
        }
    elif loss_name == "weighted_aux_orient_behavior":
        config = ArgParseModelConfig(
            loss_name="weighted_aux_orient_behavior",
            pos_weight=None,
            aux_weight=0.2,
            target_gesture_dict_path=Path(
                "/kaggle/working/encoders/inverse_gesture_dict.pkl"
            ),
        )
        loss = LossModule(config)
        pred = {
            "logits": torch.randn(34, 18),
            "orientation": torch.randn(34, 4),
            "behavior": torch.randn(34, 4),
        }
        target = torch.randint(0, 18, (34,))
        target = F.one_hot(target, num_classes=18).float()
        orient_target = torch.randint(0, 4, (34,))
        behavior_target = torch.randint(0, 4, (34,))
        orient_target = F.one_hot(orient_target, num_classes=4).float()
        behavior_target = F.one_hot(behavior_target, num_classes=4).float()
        pred["logits"] = pred["logits"].to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        pred["orientation"] = pred["orientation"].to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        pred["behavior"] = pred["behavior"].to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        inputs = {
            "logits": pred["logits"],
            "orientation": pred["orientation"],
            "behavior": pred["behavior"],
        }
        target = target.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        orient_target = orient_target.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        behavior_target = behavior_target.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        targets = {
            "labels": target,
            "orientation": orient_target,
            "behavior": behavior_target,
        }

    print(loss(inputs, targets))
