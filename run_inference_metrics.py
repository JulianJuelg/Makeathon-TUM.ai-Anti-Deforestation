"""Run full-tile inference metrics on train/val splits.

Examples:
  python run_inference_metrics.py checkpoint=checkpoints/epoch016-iou0.5966.ckpt
  python run_inference_metrics.py checkpoint=checkpoints/epoch016-iou0.5966.ckpt eval.threshold=0.35
  python run_inference_metrics.py checkpoint=checkpoints/epoch016-iou0.5966.ckpt eval.sweep_thresholds=true
  python run_inference_metrics.py checkpoint=checkpoints/epoch016-iou0.5966.ckpt inference.generate_submission=true

  # Run train/val metrics and write a timestamped submission using the val best threshold.
  python run_inference_metrics.py \
    checkpoint=checkpoints/epoch016-iou0.5966.ckpt \
    "inference.splits=[train,val]" \
    inference.generate_submission=true \
    inference.output_path=metrics/epoch016_iou05966_train_val.json

  # Try a looser submission setting for higher recall.
  python run_inference_metrics.py \
    checkpoint=checkpoints/epoch016-iou0.5966.ckpt \
    eval.threshold=0.35 \
    postprocess.min_area_ha=0.2 \
    "inference.splits=[train,val]" \
    inference.generate_submission=true \
    inference.output_path=metrics/epoch016_thr035_area03_train_val.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import cast

import hydra
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf


def _abs_path(base: str, path: str | None) -> str | None:
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(base, path)


def _unique_submission_path(base_path: str) -> str:
    root, ext = os.path.splitext(base_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_path = f"{root}_{ts}_pid{os.getpid()}{ext or '.geojson'}"
    os.makedirs(os.path.dirname(unique_path), exist_ok=True)
    return unique_path


def _resolve_splits(dm) -> dict[str, list[str]]:
    all_tiles = dm._discover_tiles()
    if not all_tiles:
        raise RuntimeError(f"No RADD tiles found under {dm.labels_dir}/radd/")

    train_tiles_cfg = getattr(dm, "_train_tiles", None)
    val_tiles_cfg = getattr(dm, "_val_tiles", None)
    if train_tiles_cfg is None and val_tiles_cfg is None:
        n_val = max(1, len(all_tiles) // 5)
        return {
            "train": all_tiles[:-n_val],
            "val": all_tiles[-n_val:],
        }

    return {
        "train": train_tiles_cfg or [],
        "val": val_tiles_cfg or [],
    }


def _load_module(cfg: DictConfig, checkpoint: str, device: torch.device):
    net = instantiate(cfg.model)
    module = instantiate(cfg.module, network=net)

    ckpt = torch.load(checkpoint, map_location="cpu")
    module.load_state_dict(ckpt["state_dict"], strict=True)

    module.eval_threshold = float(OmegaConf.select(cfg, "eval.threshold", default=0.5))
    module.sweep_thresholds = bool(OmegaConf.select(cfg, "eval.sweep_thresholds", default=False))
    module.threshold_values = list(OmegaConf.select(
        cfg,
        "eval.threshold_values",
        default=[round(0.10 + 0.05 * i, 2) for i in range(17)],
    ))
    module.best_spatial_threshold = module.eval_threshold
    module.min_area_ha = float(OmegaConf.select(cfg, "postprocess.min_area_ha", default=0.5))

    module.to(device)
    module.eval()
    return module


def _split_metric_keys(metrics: dict[str, float], split: str) -> dict[str, float]:
    renamed = {}
    for key, value in metrics.items():
        if key.startswith("val/"):
            renamed[f"{split}/{key[4:]}"] = value
        else:
            renamed[f"{split}/{key}"] = value
    return renamed


def _print_summary(split: str, metrics: dict[str, float]) -> None:
    prefix = f"{split}/"
    print(
        f"[{split}]"
        f" IoU={metrics[prefix + 'union_iou']:.4f}"
        f" Recall={metrics[prefix + 'recall']:.4f}"
        f" FPR={metrics[prefix + 'fpr']:.4f}"
        f" YearAcc={metrics[prefix + 'year_acc']:.4f}"
        f" Thr={metrics[prefix + 'best_threshold']:.2f}"
        f" pred={metrics[prefix + 'pred_area_ha']:.2f}ha"
        f" gt={metrics[prefix + 'gt_area_ha']:.2f}ha"
    )


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> dict[str, dict[str, float]]:
    orig = get_original_cwd()
    if orig not in sys.path:
        sys.path.insert(0, orig)

    checkpoint = OmegaConf.select(cfg, "checkpoint", default=None)
    if checkpoint is None:
        raise ValueError("Pass checkpoint=path/to/checkpoint.ckpt")
    checkpoint = cast(str, _abs_path(orig, str(checkpoint)))
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    cfg = cast(DictConfig, OmegaConf.merge(cfg, {
        "data": {
            "data_dir": _abs_path(orig, str(cfg.data.data_dir)),
            "cache_dir": _abs_path(orig, str(cfg.data.cache_dir)),
        },
        "submission": {
            "prediction_dir": _abs_path(orig, str(OmegaConf.select(cfg, "submission.prediction_dir", default="predictions"))),
            "output_path": _abs_path(orig, str(OmegaConf.select(cfg, "submission.output_path", default="submission/submission.geojson"))),
        },
    }))

    device_name = str(OmegaConf.select(cfg, "inference.device", default="auto"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)

    split_names = list(OmegaConf.select(cfg, "inference.splits", default=["train", "val"]))
    target_years = list(OmegaConf.select(cfg, "inference.target_years", default=[2025]))
    output_path = OmegaConf.select(cfg, "inference.output_path", default=None)
    output_path = _abs_path(orig, str(output_path)) if output_path is not None else None
    generate_submission = bool(OmegaConf.select(cfg, "inference.generate_submission", default=False))
    unique_submission_path = bool(OmegaConf.select(cfg, "inference.unique_submission_path", default=True))
    submission_threshold_split = str(OmegaConf.select(cfg, "inference.submission_threshold_split", default="val"))

    dm = instantiate(cfg.data)
    splits = _resolve_splits(dm)
    module = _load_module(cfg, checkpoint, device)

    thresholds = module.threshold_values if module.sweep_thresholds else module.eval_threshold
    print(f"checkpoint: {checkpoint}")
    print(f"device: {device}")
    print(f"target_years: {target_years}")
    print(f"thresholds: {thresholds}")
    print(f"min_area_ha: {module.min_area_ha}")
    print(f"filter_mode: {getattr(dm, 'filter_mode', 'strict')}")
    print(f"label_thresholds: {getattr(dm, 'label_thresholds', None)}")

    from deforestation.metrics import (
        TEST_TILES,
        compute_challenge_metrics,
        generate_combined_submission,
    )

    results: dict[str, dict[str, float]] = {}
    for split in split_names:
        if split not in splits:
            raise ValueError(f"Unknown split '{split}'. Expected one of {sorted(splits)}")
        tiles = splits[split]
        print(f"\nRunning {split} split on {len(tiles)} tiles: {tiles}")
        metrics = compute_challenge_metrics(
            module=module,
            val_tiles=tiles,
            aef_dir=dm.aef_dir,
            labels_dir=dm.labels_dir,
            cache_dir=dm.cache_dir,
            target_years=target_years,
            threshold=thresholds,
            selected_bands=dm.selected_bands,
            min_area_ha=module.min_area_ha,
            label_thresholds=getattr(dm, "label_thresholds", None),
            filter_mode=getattr(dm, "filter_mode", "strict"),
        )
        renamed = _split_metric_keys(metrics, split)
        results[split] = renamed
        _print_summary(split, renamed)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print(f"\nwrote metrics: {output_path}")

    if generate_submission:
        if submission_threshold_split in results:
            submission_threshold = float(
                results[submission_threshold_split][f"{submission_threshold_split}/best_threshold"]
            )
        elif "val" in results:
            submission_threshold = float(results["val"]["val/best_threshold"])
        else:
            submission_threshold = float(module.eval_threshold)

        test_tiles_cfg = OmegaConf.select(cfg, "inference.submission_tiles", default=None)
        test_tiles = list(test_tiles_cfg) if test_tiles_cfg is not None else TEST_TILES
        test_aef_dir = dm.aef_dir.replace("train", "test")
        submission_path = str(OmegaConf.select(cfg, "submission.output_path", default="submission/submission.geojson"))
        if unique_submission_path:
            submission_path = _unique_submission_path(submission_path)
        else:
            Path(submission_path).parent.mkdir(parents=True, exist_ok=True)

        submission = generate_combined_submission(
            module=module,
            test_tiles=test_tiles,
            aef_dir=test_aef_dir,
            cache_dir=dm.cache_dir,
            target_years=target_years,
            threshold=submission_threshold,
            selected_bands=dm.selected_bands,
            prediction_dir=str(OmegaConf.select(cfg, "submission.prediction_dir", default="predictions")),
            output_path=submission_path,
            min_area_ha=module.min_area_ha,
        )
        print(
            f"\n[Submission] wrote {len(submission['features'])} polygons"
            f" -> {submission_path}"
            f" (threshold={submission_threshold:.2f})"
        )

    return results


if __name__ == "__main__":
    main()
