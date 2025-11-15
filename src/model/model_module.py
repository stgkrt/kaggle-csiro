import os
from pathlib import Path
from typing import Any

import joblib
import polars as pl
import pytorch_lightning as L
import torch
from sklearn.metrics import f1_score
from timm.utils import ModelEmaV2
from torchmetrics import MeanMetric

from src.log_utils.pylogger import RankedLogger
from src.metrics.competition_metrics import CompetitionMetrics
from src.model.architectures.model_architectures import ModelArchitectures
from src.model.losses import LossModule


class ModelModule(L.LightningModule):
    def __init__(
        self,
        model_architectures: ModelArchitectures,
        criterion: LossModule,
        metrics: CompetitionMetrics,
        compile: bool,
        oof_dir: Path = Path("/kaggle/working/oof"),
        lr: float = 2e-4,
        weight_decay: float = 1e-3,
        max_epochs: int = 100,
        scheduler_t_0: int = 100,
        scheduler_t_mult: int = 1,
        scheduler_eta_min: float = 1e-9,
        ema_decay: float = 0.998,
        ema_enable: bool = True,
    ) -> None:
        super().__init__()
        self.model = model_architectures
        self.criterion = criterion
        self.metrics = metrics
        self.best_metrics = -torch.inf
        self.oof_dir = oof_dir
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.scheduler_t_0 = scheduler_t_0
        self.scheduler_t_mult = scheduler_t_mult
        self.scheduler_eta_min = scheduler_eta_min
        self.ema_decay = ema_decay
        self.ema_enable = ema_enable

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_loss = MeanMetric()
        self.valid_loss = MeanMetric()

        # EMA初期化
        if self.ema_enable:
            self.model_ema = ModelEmaV2(self.model, decay=self.ema_decay)
        else:
            self.model_ema = None
        # validの予測値と正解値を保存するための変数(device設定)
        if torch.cuda.is_available():
            self.accelarator = "cuda"
        else:
            self.accelarator = "cpu"
        self.valid_preds = torch.Tensor().to(self.accelarator)
        self.valid_labels = torch.Tensor().to(self.accelarator)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":  # type: ignore
            self.model = torch.compile(self.model)  # type: ignore
            self.model_ema = torch.compile(self.model_ema) if self.model_ema else None

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = self.model(x)
        return x

    def model_step(
        self, batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, targets = batch
        outputs = self.forward(inputs)
        preds = outputs["logits"]  # model output logits
        labels = targets["labels"]  # one-hot encoded labels
        loss = self.criterion(outputs, targets)  # calculate loss
        # preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

    def model_step_ema(
        self, batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Model step with EMA."""
        inputs, targets = batch
        # EMAモデルでの推論
        outputs = self.model_ema.module(inputs)  # type: ignore
        preds = outputs["logits"]  # model output logits
        labels = targets["labels"]  # one-hot encoded labels
        loss = self.criterion(outputs, targets)  # calculate loss
        # preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

    def training_step(
        self,
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, preds, labels = self.model_step(batch)
        self.train_loss(loss)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # EMAの更新
        if self.model_ema is not None:
            self.model_ema.update(self.model)

        return loss

    def validation_step(
        self, batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        # EMAモデルでの推論
        if self.model_ema is not None:
            loss, preds, labels = self.model_step_ema(batch)
        else:
            loss, preds, labels = self.model_step(batch)

        self.valid_preds = torch.cat((self.valid_preds, preds))
        self.valid_labels = torch.cat((self.valid_labels, labels))
        self.valid_loss(loss)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    # epochの終わりにmetricsのlogを出力
    def on_train_epoch_end(self) -> None:
        valid_labels = self.valid_labels.cpu()
        valid_preds = self.valid_preds.cpu()
        if valid_preds.shape[-1] == 18:
            metrics = self.metrics(valid_labels, valid_preds)
            self.log(
                "competition_metrics",
                metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            f1_macro = self.metrics.macro_f1
            f1_binary = self.metrics.binary_f1
            self.log(
                "val/f1_macro",
                f1_macro,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "val/f1_binary",
                f1_binary,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.save_best(metrics)
            accuracy = (
                (valid_preds.argmax(dim=1) == valid_labels.argmax(dim=1)).float().mean()
            )
            self.log(
                "val/accuracy",
                accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

            # preds/labelsの初期化
            oof = pl.DataFrame(
                {
                    **{
                        f"pred_{i}": self.valid_preds[:, i].cpu().numpy()
                        for i in range(self.valid_preds.shape[-1])
                    },
                    **{
                        f"target_{i}": self.valid_labels[:, i].cpu().numpy()
                        for i in range(self.valid_labels.shape[-1])
                    },
                }
            )
        else:
            valid_labels = valid_labels.numpy()
            valid_preds = valid_preds.numpy()
            valid_labels_idx = valid_labels.argmax(axis=1)
            valid_preds_idx = valid_preds.argmax(axis=1)
            metrics = f1_score(valid_labels_idx, valid_preds_idx, average="macro")
            self.log(
                "val/f1_macro",
                metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.save_best(metrics)
            oof = pl.DataFrame(
                {
                    **{
                        f"pred_{i}": valid_preds[:, i]
                        for i in range(valid_preds.shape[-1])
                    },
                    **{
                        f"target_{i}": valid_labels[:, i]
                        for i in range(valid_labels.shape[-1])
                    },
                }
            )

        oof.write_csv(os.path.join(self.oof_dir, "oof.csv"))
        self.valid_preds = torch.Tensor().to(self.accelarator)
        self.valid_labels = torch.Tensor().to(self.accelarator)
        if self.trainer.datamodule.dataset_name == "augmented_aux_mixup":
            self.trainer.train_dataloader.dataset.set_epoch(self.current_epoch)

        return super().on_train_epoch_end()

    def save_best(self, metrics: float) -> None:
        if metrics > self.best_metrics:
            self.best_metrics = metrics
            pred_cols = [f"pred_{i}" for i in range(self.valid_preds.shape[-1])]  # type: ignore
            target_cols = [f"target_{i}" for i in range(self.valid_labels.shape[-1])]  # type: ignore
            # oofの保存
            oof = pl.DataFrame(
                {
                    **{
                        col: self.valid_preds[:, i].cpu().numpy()
                        for i, col in enumerate(pred_cols)
                    },
                    **{
                        col: self.valid_labels[:, i].cpu().numpy()
                        for i, col in enumerate(target_cols)
                    },
                }
            )
            oof.write_csv(os.path.join(self.oof_dir, "best_oof.csv"))
            # save best weights
            if self.model_ema is not None:
                weights_path = os.path.join(self.oof_dir, "best_weights.pth")
                torch.save(self.model_ema.module.model.state_dict(), weights_path)
            else:
                weights_path = os.path.join(self.oof_dir, "best_weights.pth")
                torch.save(self.model.model.state_dict(), weights_path)
        self.log(
            "best_metrics",
            self.best_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    # 全epoch終了時にweightsを保存
    def on_train_end(self) -> None:
        if self.model_ema is not None:
            weights_path = os.path.join(self.oof_dir, "final_weights.pth")
            torch.save(self.model_ema.module.model.state_dict(), weights_path)
        else:
            weights_path = os.path.join(self.oof_dir, "final_weights.pth")
            torch.save(self.model.model.state_dict(), weights_path)

        weights_path = os.path.join(self.oof_dir, "final_weights_orig.pth")
        torch.save(self.model.model.state_dict(), weights_path)
        return super().on_train_end()

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore
        """Choose what optimizers and learning-rate schedulers
        to use in your optimization.
        Normally you'd need one.
        But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Returns:
            A dict containing the configured optimizers and learning-rate schedulers
            to be used for training.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=self.scheduler_t_0,
        #     T_mult=self.scheduler_t_mult,
        #     eta_min=self.scheduler_eta_min,
        #     last_epoch=-1,
        #     verbose=False,
        # )
        # cosine anealing LR
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.scheduler_t_0,
            eta_min=self.scheduler_eta_min,
            last_epoch=-1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def save_state_dict(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
        print(f"Model weights saved to {path}")
        return
