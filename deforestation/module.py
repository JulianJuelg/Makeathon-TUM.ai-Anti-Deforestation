"""LightningModule for deforestation segmentation training."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import torch.distributed as dist

import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, mask=None):
        preds = torch.sigmoid(logits)
        if mask is not None:
            preds = preds * mask
            targets = targets * mask

        intersection = 2.0 * (preds * targets).sum()
        union = preds.sum() + targets.sum()
        dice = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=None, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets, mask=None):
        targets = targets.float()
        
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        if mask is not None:
            bce_loss = (bce_loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            bce_loss = bce_loss.mean()

        return self.bce_weight * bce_loss + self.dice_weight * self.dice(logits, targets, mask=mask)

class DeforestationModule(pl.LightningModule):
    """Training wrapper for a spatial segmentation network."""

    def __init__(
        self,
        network: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        pos_weight: float | None = None,
        eval_threshold: float = 0.5,
        sweep_thresholds: bool = True,
        threshold_values: list[float] | None = None,
        prediction_dir: str = "predictions",
        submission_output_path: str = "submission/submission.geojson",
        min_area_ha: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["network"])

        self.net = network

        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.loss_fn = BCEDiceLoss(pos_weight=pw)

        metric_kwargs = dict(task="binary", threshold=0.5)
        self.train_f1  = torchmetrics.F1Score(**metric_kwargs)
        self.val_f1    = torchmetrics.F1Score(**metric_kwargs)
        self.val_prec  = torchmetrics.Precision(**metric_kwargs)
        self.val_rec   = torchmetrics.Recall(**metric_kwargs)
        self.eval_threshold = eval_threshold
        self.sweep_thresholds = sweep_thresholds
        self.threshold_values = threshold_values or [round(0.10 + 0.05 * i, 2) for i in range(17)]
        self.best_spatial_threshold = eval_threshold
        self.prediction_dir = prediction_dir
        self.submission_output_path = submission_output_path
        self.min_area_ha = min_area_ha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits with shape matching the network head."""
        return self.net(x)

    def training_step(self, batch, batch_idx: int):
        x, y, mask = batch
        logits = self(x)
        loss = self.loss_fn(logits, y, mask=mask)

        valid_logits = logits[mask.bool()]
        valid_y = y[mask.bool()]
        if len(valid_y) > 0:
            preds = torch.sigmoid(valid_logits)
            self.train_f1(preds, valid_y.int())

        self.log("train/loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y, mask = batch
        logits = self(x)
        loss = self.loss_fn(logits, y, mask=mask)

        valid_logits = logits[mask.bool()]
        valid_y = y[mask.bool()]
        if len(valid_y) > 0:
            preds = torch.sigmoid(valid_logits)
            self.val_f1(preds,   valid_y.int())
            self.val_prec(preds, valid_y.int())
            self.val_rec(preds,  valid_y.int())

        self.log("val/loss",      loss,          on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log("val/f1",        self.val_f1,   on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log("val/precision", self.val_prec, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/recall",    self.val_rec,  on_epoch=True, prog_bar=False, sync_dist=True)

    def on_validation_epoch_end(self):
        if not self.trainer or not hasattr(self.trainer, "datamodule"):
            return
        dm = self.trainer.datamodule
        if dm is None or not hasattr(dm, "val_tiles_resolved"):
            return

        epoch = self.current_epoch
        every_n = getattr(dm, "spatial_eval_every_n_epochs", 5)
        last_epoch = self.trainer.max_epochs - 1
        if epoch % every_n != 0 and epoch != last_epoch:
            return

        from deforestation.metrics import (
            TEST_TILES,
            compute_challenge_metrics,
            generate_combined_submission,
            visualize_test_tiles,
        )
        
        eval_years = [2025]
        thresholds = self.threshold_values if self.sweep_thresholds else self.eval_threshold

        metrics = None
        if self.trainer.is_global_zero:
            metrics = compute_challenge_metrics(
                module=self,
                val_tiles=dm.val_tiles_resolved,
                aef_dir=dm.aef_dir,
                labels_dir=dm.labels_dir,
                cache_dir=dm.cache_dir,
                target_years=eval_years,
                threshold=thresholds,
                selected_bands=dm.selected_bands,
                min_area_ha=self.min_area_ha,
                label_thresholds=getattr(dm, "label_thresholds", None),
                filter_mode=getattr(dm, "filter_mode", "strict"),
            )
        if dist.is_available() and dist.is_initialized():
            payload = [metrics]
            dist.broadcast_object_list(payload, src=0)
            metrics = payload[0]
        if metrics is None:
            return

        prog_keys = {"val/union_iou", "val/recall", "val/year_acc"}
        for k, v in metrics.items():
            self.log(k, v, prog_bar=(k in prog_keys), sync_dist=False)

        self.best_spatial_threshold = float(metrics.get("val/best_threshold", self.eval_threshold))

        if self.trainer.is_global_zero:
            print(
                f"\n[Spatial @ epoch {epoch}]"
                f"  IoU={metrics['val/union_iou']:.4f}"
                f"  Recall={metrics['val/recall']:.4f}"
                f"  FPR={metrics['val/fpr']:.4f}"
                f"  YearAcc={metrics['val/year_acc']:.4f}"
                f"  Thr={self.best_spatial_threshold:.2f}"
                f"  pred={metrics['val/pred_area_ha']:.2f}ha"
                f"  gt={metrics['val/gt_area_ha']:.2f}ha"
            )

        if not self.trainer.is_global_zero:
            return

        test_aef_dir = dm.aef_dir.replace("train", "test")
        visualize_test_tiles(
            module=self,
            test_tiles=TEST_TILES,
            aef_dir=test_aef_dir,
            cache_dir=dm.cache_dir,
            target_years=eval_years,
            threshold=self.best_spatial_threshold,
            selected_bands=dm.selected_bands,
            min_area_ha=self.min_area_ha,
        )
        submission = generate_combined_submission(
            module=self,
            test_tiles=TEST_TILES,
            aef_dir=test_aef_dir,
            cache_dir=dm.cache_dir,
            target_years=eval_years,
            threshold=self.best_spatial_threshold,
            selected_bands=dm.selected_bands,
            prediction_dir=self.prediction_dir,
            output_path=self.submission_output_path,
            min_area_ha=self.min_area_ha,
        )
        print(
            f"[Submission] wrote {len(submission['features'])} polygons"
            f" -> {self.submission_output_path}"
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.lr * 1e-2,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/union_iou",
                "interval": "epoch",
            },
        }
