import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytorch_lightning as L
import torch
from timm.utils import ModelEmaV2
from torchmetrics import AUROC, MeanMetric

import wandb
from src.data.mixup_data import MixupCutmixWrapper
from src.metrics.competition_metrics import CompetitionMetrics, calculate_custom_metric
from src.model.architectures.model_architectures import ModelArchitectures
from src.model.losses import LossModule


class ModelModule(L.LightningModule):
    def __init__(
        self,
        model_architectures: ModelArchitectures,
        criterion: LossModule,
        metrics: CompetitionMetrics,
        valid_df: pd.DataFrame,
        target_cols: list[str],
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
        mixup_alpha: float = 1.0,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 0.5,
        cutmix_prob: float = 0.5,
        mixup_cutmixup_buffer_size: int = 4,
        mixup_cutmix_n_splits: int = 2,
    ) -> None:
        super().__init__()
        self.model = model_architectures
        self.criterion = criterion
        self.metrics = metrics

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False, ignore=["model_architectures", "criterion", "metrics"]
        )
        self.oof_dir = oof_dir
        os.makedirs(self.oof_dir, exist_ok=True)

        self.train_loss = MeanMetric()
        self.valid_loss = MeanMetric()

        # EMA初期化
        if self.hparams.ema_enable:
            self.model_ema = ModelEmaV2(self.model, decay=self.hparams.ema_decay)
        else:
            self.model_ema = None
        # validの予測値と正解値を保存するための変数(device設定)
        if torch.cuda.is_available():
            self.accelarator = "cuda"
        else:
            self.accelarator = "cpu"
        self.valid_preds = torch.Tensor().to(self.accelarator)
        self.valid_labels = torch.Tensor().to(self.accelarator)
        self.valid_height_preds: torch.Tensor | None = None
        self.valid_height_labels: torch.Tensor | None = None
        self.valid_clover_preds: torch.Tensor | None = None
        self.valid_clover_labels: torch.Tensor | None = None
        self.train_height_preds: torch.Tensor | None = None
        self.train_height_labels: torch.Tensor | None = None
        self.train_clover_preds: torch.Tensor | None = None
        self.train_clover_labels: torch.Tensor | None = None
        self.best_metrics = -float("inf")
        self.valid_df = valid_df
        self.target_cols = target_cols
        # mixup
        self.mixer = MixupCutmixWrapper(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            mixup_prob=mixup_prob,
            cutmix_prob=cutmix_prob,
            buffer_size=mixup_cutmixup_buffer_size,  # 全データセット対応
            n_splits=mixup_cutmix_n_splits,
        )

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":  # type: ignore
            self.model = torch.compile(self.model)  # type: ignore
            self.model_ema = torch.compile(self.model_ema) if self.model_ema else None

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = self.model(x)
        return x

    def model_step(
        self, batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        inputs, targets = batch
        outputs = self.forward(inputs)
        preds = outputs["logits"]  # model output logits
        labels = targets["labels"]  # one-hot encoded labels
        loss = self.criterion(outputs, targets)  # calculate loss
        return loss, preds, labels, outputs

    def model_step_ema(
        self, batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Model step with EMA."""
        inputs, targets = batch
        outputs = self.model_ema.module(inputs)  # type: ignore
        preds = outputs["logits"]  # model output logits
        labels = targets["labels"]  # one-hot encoded labels
        loss = self.criterion(outputs, targets)  # calculate loss
        return loss, preds, labels, outputs

    def training_step(
        self,
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        inputs, targets = batch
        inputs, targets = self.mixer(inputs, targets)
        loss, _, _, outputs = self.model_step((inputs, targets))
        self.train_loss(loss)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Collect height data
        height_preds = outputs.get("height")
        height_labels = targets.get("height")
        if height_preds is not None and height_labels is not None:
            if self.train_height_preds is None:
                self.train_height_preds = height_preds.detach()
                self.train_height_labels = height_labels.detach()
            else:
                self.train_height_preds = torch.cat(
                    (self.train_height_preds, height_preds.detach())
                )
                self.train_height_labels = torch.cat(
                    (self.train_height_labels, height_labels.detach())
                )

        # Collect clover data
        clover_preds = outputs.get("include_clover_preds")
        clover_labels = targets.get("include_clover_label")
        if clover_preds is not None and clover_labels is not None:
            if self.train_clover_preds is None:
                self.train_clover_preds = clover_preds.detach()
                self.train_clover_labels = clover_labels.detach()
            else:
                self.train_clover_preds = torch.cat(
                    (self.train_clover_preds, clover_preds.detach())
                )
                self.train_clover_labels = torch.cat(
                    (self.train_clover_labels, clover_labels.detach())
                )

        # EMAの更新
        if self.model_ema is not None:
            self.model_ema.update(self.model)

        return loss

    def validation_step(
        self, batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        inputs, targets = batch
        if self.model_ema is not None:
            loss, preds, labels, outputs = self.model_step_ema((inputs, targets))
        else:
            loss, preds, labels, outputs = self.model_step((inputs, targets))

        self.valid_preds = torch.cat((self.valid_preds, preds))
        self.valid_labels = torch.cat((self.valid_labels, labels))

        height_preds = outputs.get("height")
        height_labels = targets.get("height")
        if height_preds is not None and height_labels is not None:
            if self.valid_height_preds is None:
                self.valid_height_preds = height_preds.detach()
                self.valid_height_labels = height_labels.detach()
            else:
                self.valid_height_preds = torch.cat(
                    (self.valid_height_preds, height_preds.detach())
                )
                self.valid_height_labels = torch.cat(
                    (self.valid_height_labels, height_labels.detach())
                )
        clover_preds = outputs.get("include_clover_preds")
        clover_labels = targets.get("include_clover_label")
        if clover_preds is not None and clover_labels is not None:
            if self.valid_clover_preds is None:
                self.valid_clover_preds = clover_preds.detach()
                self.valid_clover_labels = clover_labels.detach()
            else:
                self.valid_clover_preds = torch.cat(
                    (self.valid_clover_preds, clover_preds.detach())
                )
                self.valid_clover_labels = torch.cat(
                    (self.valid_clover_labels, clover_labels.detach())
                )

        self.valid_loss(loss)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def _make_oof_df(self) -> pd.DataFrame:
        sample_ids_targets = self.valid_df["sample_id"].unique()
        sample_ids = [sid.split("_")[0] for sid in sample_ids_targets]
        sample_ids = list(sorted(set(sample_ids)))

        # Create submission dataframe
        submission_rows = []
        valid_labels = self.valid_labels.cpu().numpy()
        valid_preds = self.valid_preds.cpu().numpy()
        for i, sample_id in enumerate(sample_ids):
            for j, target_name in enumerate(self.target_cols):
                submission_rows.append(
                    {
                        "sample_id": f"{sample_id}__{target_name}",
                        "pred": valid_preds[i, j],
                        "target": valid_labels[i, j],
                    }
                )
        oof_df = pd.DataFrame(submission_rows)
        return oof_df

    # epochの終わりにmetricsのlogを出力
    def on_train_epoch_end(self) -> None:
        valid_labels = self.valid_labels.cpu()
        valid_preds = self.valid_preds.cpu()
        valid_preds = torch.clamp(valid_preds, min=0.0)  # 負の予測値を0にクリップ
        metrics = self.metrics(valid_labels, valid_preds)
        self.log(
            "competition_metrics",
            metrics["weighted_r2"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "val/r2_Dry_Clover_g",
            metrics["r2_Dry_Clover_g"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/r2_Dry_Dead_g",
            metrics["r2_Dry_Dead_g"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/r2_Dry_Green_g",
            metrics["r2_Dry_Green_g"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/r2_Dry_Total_g",
            metrics["r2_Dry_Total_g"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/r2_GDM_g",
            metrics["r2_GDM_g"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Log train height and clover metrics
        if self.train_height_preds is not None and self.train_height_labels is not None:
            self.height_preds = torch.clamp(self.train_height_preds, min=0.0)
            train_height_diff_mean = (
                self.train_height_preds - self.train_height_labels
            ).mean()
            self.log(
                "train_height/height_diff_mean",
                train_height_diff_mean,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        if self.train_clover_preds is not None and self.train_clover_labels is not None:
            auroc = AUROC(task="binary")
            clover_preds_cpu = self.train_clover_preds.cpu()
            clover_labels_cpu = self.train_clover_labels.cpu()
            auc_score = auroc(clover_preds_cpu, clover_labels_cpu.long())
            self.log(
                "train_clover/auc",
                auc_score,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        # Log val height and clover metrics
        if self.valid_height_preds is not None and self.valid_height_labels is not None:
            height_diff_mean = (
                self.valid_height_preds - self.valid_height_labels
            ).mean()
            self.log(
                "val_height/height_diff_mean",
                height_diff_mean,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        if self.valid_clover_preds is not None and self.valid_clover_labels is not None:
            auroc = AUROC(task="binary")
            clover_preds_cpu = self.valid_clover_preds.cpu()
            clover_labels_cpu = self.valid_clover_labels.cpu()
            auc_score = auroc(clover_preds_cpu, clover_labels_cpu.long())
            self.log(
                "val_clover/auc",
                auc_score,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        self.save_best(metrics["weighted_r2"])
        # preds/labelsの初期化
        sample_ids_targets = self.valid_df["sample_id"].unique()
        sample_ids = [sid.split("_")[0] for sid in sample_ids_targets]
        sample_ids = list(sorted(set(sample_ids)))

        # Create submission dataframe
        oof_df = self._make_oof_df()

        oof_path = self.oof_dir / "oof.csv"
        oof_df.to_csv(oof_path, index=False)
        # self._log_histogram()
        # self._log_scatter()

        # Reset valid preds and labels for next epoch
        self.valid_preds = torch.Tensor().to(self.accelarator)
        self.valid_labels = torch.Tensor().to(self.accelarator)
        self.valid_height_preds = None
        self.valid_height_labels = None
        self.valid_clover_preds = None
        self.valid_clover_labels = None
        self.train_height_preds = None
        self.train_height_labels = None
        self.train_clover_preds = None
        self.train_clover_labels = None

        # custom metricの計算とログ出力
        custom_metrics = calculate_custom_metric(oof_df, self.valid_df)
        for key, value in custom_metrics.items():
            self.log(
                f"{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        return super().on_train_epoch_end()

    def save_best(self, metrics: float) -> None:
        if metrics > self.best_metrics:
            self.best_metrics = metrics

            oof = self._make_oof_df()
            oof_path = self.oof_dir / "best_oof.csv"
            oof.to_csv(oof_path, index=False)
            # save best weights
            if self.model_ema is not None:
                weights_path = self.oof_dir / "best_weights.pth"
                torch.save(self.model_ema.module.model.state_dict(), weights_path)
            else:
                weights_path = self.oof_dir / "best_weights.pth"
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
            weights_path = self.oof_dir / "final_weights.pth"
            torch.save(self.model_ema.module.model.state_dict(), weights_path)
        else:
            weights_path = self.oof_dir / "final_weights.pth"
            torch.save(self.model.model.state_dict(), weights_path)

        weights_path = self.oof_dir / "final_weights_orig.pth"
        torch.save(self.model.model.state_dict(), weights_path)
        return super().on_train_end()

    def _log_histogram(self) -> None:
        valid_preds = self.valid_preds.cpu().numpy()
        # valid_labels = self.valid_labels.cpu().numpy()
        for i, target_name in enumerate(self.target_cols):
            hist_pred = wandb.Histogram(valid_preds[:, i])  # type: ignore
            self.logger.experiment.log({f"histogram_preds_{target_name}": hist_pred})
        return

    def _log_scatter(self) -> None:
        valid_preds = self.valid_preds.cpu().numpy()
        valid_labels = self.valid_labels.cpu().numpy()
        for i, target_name in enumerate(self.target_cols):
            data = [
                [x, y]
                for (x, y) in zip(valid_labels[:, i], valid_preds[:, i], strict=True)
            ]
            table = wandb.Table(data=data, columns=["target", "pred"])  # type: ignore
            wandb.log(  # type: ignore
                {
                    f"scatter_pred_vs_target_{target_name}": wandb.plot.scatter(  # type: ignore
                        table,
                        "target",
                        "pred",
                        title=f"Pred vs Target Scatter Plot for {target_name}",
                    )
                }
            )

        return

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
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=self.hparams.scheduler_t_0,
        #     T_mult=self.hparams.scheduler_t_mult,
        #     eta_min=self.hparams.scheduler_eta_min,
        #     last_epoch=-1,
        #     verbose=False,
        # )
        # cosine anealing LR
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.scheduler_t_0,
            eta_min=self.hparams.scheduler_eta_min,
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
