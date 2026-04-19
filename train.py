"""Hydra training entrypoint.

Usage:
  python train.py
  python train.py trainer.devices=4
  python train.py "data.val_tiles=[48QVE_3_0,48QWD_2_2]"
  python train.py --multirun module.lr=1e-3,5e-4,1e-4
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import cast

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> float:
    orig = get_original_cwd()
    if orig not in sys.path:
        sys.path.insert(0, orig)

    def _abs(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(orig, p)

    def _unique_submission_path(base_path: str) -> str:
        root, ext = os.path.splitext(base_path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        unique_path = f"{root}_{ts}_pid{os.getpid()}{ext or '.geojson'}"
        os.makedirs(os.path.dirname(unique_path), exist_ok=True)
        return unique_path

    submission_output_path = _unique_submission_path(
        _abs(OmegaConf.select(cfg, "submission.output_path", default="submission/submission.geojson"))
    )

    cfg = cast(DictConfig, OmegaConf.merge(cfg, {
        "data": {
            "data_dir":  _abs(cfg.data.data_dir),
            "cache_dir": _abs(cfg.data.cache_dir),
        },
        "submission": {
            "prediction_dir": _abs(OmegaConf.select(cfg, "submission.prediction_dir", default="predictions")),
            "output_path": submission_output_path,
        },
    }))

    print(OmegaConf.to_yaml(cfg, resolve=True))
    pl.seed_everything(cfg.seed, workers=True)

    dm     = instantiate(cfg.data)
    net    = instantiate(cfg.model)
    module = instantiate(cfg.module, network=net)
    module.eval_threshold = float(OmegaConf.select(cfg, "eval.threshold", default=0.5))
    module.sweep_thresholds = bool(OmegaConf.select(cfg, "eval.sweep_thresholds", default=True))
    module.threshold_values = list(OmegaConf.select(
        cfg, "eval.threshold_values",
        default=[round(0.10 + 0.05 * i, 2) for i in range(17)]
    ))
    module.best_spatial_threshold = module.eval_threshold
    module.prediction_dir = str(OmegaConf.select(cfg, "submission.prediction_dir", default=_abs("predictions")))
    module.submission_output_path = str(OmegaConf.select(cfg, "submission.output_path", default=submission_output_path))
    module.min_area_ha = float(OmegaConf.select(cfg, "postprocess.min_area_ha", default=0.5))

    callbacks = [instantiate(cb) for cb in cfg.trainer.callbacks.values()]
    logger    = instantiate(cfg.trainer.logger)

    ckpt_dir = os.path.join(os.getcwd(), "checkpoints")
    for cb in callbacks:
        if hasattr(cb, "dirpath") and cb.dirpath is None:
            cb.dirpath = ckpt_dir

    devices = cfg.trainer.devices
    if isinstance(devices, int) and devices > 1:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = "auto"

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=devices,
        strategy=strategy,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        deterministic=cfg.trainer.deterministic,
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(module, datamodule=dm)

    val_f1 = float(trainer.callback_metrics.get("val/f1", float("nan")))
    print(f"\n✓ Done | best val/f1 = {val_f1:.4f}")
    print(f"  checkpoints → {ckpt_dir}")
    print(f"  tensorboard → tensorboard --logdir logs")
    return val_f1


if __name__ == "__main__":
    main()
