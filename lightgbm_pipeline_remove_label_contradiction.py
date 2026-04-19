
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import joblib
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.warp import reproject
from shapely.geometry import shape
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

TILE_YEAR_RE = re.compile(r"^(?P<tile>.+)_(?P<year>\d{4})\.tiff$")
S2_FILE_RE = re.compile(r"^(?P<tile>.+)__s2_l2a_(?P<year>\d{4})_(?P<month>\d{1,2})\.tif$")
GLADL_YEARS = tuple(range(2021, 2026))
GLADS2_EPOCH = date(2019, 1, 1)
RADD_EPOCH = date(2014, 12, 31)

STATE_UNAVAILABLE = -1
STATE_NEG = 0
STATE_UNCERTAIN = 1
STATE_SOFT_POS = 2
STATE_HARD_POS = 3


@dataclass(frozen=True)
class TileFeatures:
    tile_id: str
    baseline_year: int
    comparison_year: int
    feature_cube: np.ndarray
    transform: Affine
    crs: object
    height: int
    width: int


@dataclass(frozen=True)
class TargetConfig:
    label_source: str = "consensus"
    min_confidence_glads2: int = 4
    min_confidence_gladl: int = 3
    min_confidence_radd: int = 3
    pre2020_min_confidence_glads2: int = 3
    pre2020_min_confidence_radd: int = 3


def log(msg: str) -> None:
    print(msg, flush=True)


def safe_dump_args(args: argparse.Namespace, out_path: Path) -> None:
    payload = {k: v for k, v in vars(args).items() if not callable(v)}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def list_tile_ids_from_metadata(geojson_path: Path) -> list[str] | None:
    if not geojson_path.exists():
        return None
    gdf = gpd.read_file(geojson_path)
    if "name" in gdf.columns:
        ids = gdf["name"].astype(str).tolist()
        if ids:
            return ids
    for col in ["tile_id", "id"]:
        if col in gdf.columns:
            ids = gdf[col].astype(str).tolist()
            if ids:
                return ids
    return None


def discover_aef_years(aef_dir: Path, split: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for path in sorted((aef_dir / split).glob("*.tiff")):
        m = TILE_YEAR_RE.match(path.name)
        if not m:
            continue
        out.setdefault(m.group("tile"), []).append(int(m.group("year")))
    for tile_id in list(out):
        out[tile_id] = sorted(set(out[tile_id]))
    return out


def discover_tile_ids(aef_dir: Path, split: str, metadata_root: Path | None = None) -> list[str]:
    if metadata_root is not None:
        ids = list_tile_ids_from_metadata(metadata_root / f"{split}_tiles.geojson")
        if ids:
            return ids
    return sorted(discover_aef_years(aef_dir, split).keys())


def read_aef(tile_id: str, year: int, aef_dir: Path, split: str) -> tuple[np.ndarray, Affine, object]:
    path = aef_dir / split / f"{tile_id}_{year}.tiff"
    with rasterio.open(path) as src:
        return src.read().astype(np.float32), src.transform, src.crs


def build_feature_cube(tile_id: str, split: str, aef_dir: Path, baseline_year: int, comparison_year: int) -> TileFeatures:
    base, transform, crs = read_aef(tile_id, baseline_year, aef_dir, split)
    comp, comp_transform, comp_crs = read_aef(tile_id, comparison_year, aef_dir, split)
    if base.shape != comp.shape or transform != comp_transform or crs != comp_crs:
        reproj = np.zeros_like(base, dtype=np.float32)
        for i in range(comp.shape[0]):
            reproject(source=comp[i], destination=reproj[i], src_transform=comp_transform, src_crs=comp_crs, dst_transform=transform, dst_crs=crs, resampling=Resampling.bilinear)
        comp = reproj
    delta = comp - base
    cube = np.concatenate([base, comp, delta], axis=0)
    _, h, w = cube.shape
    return TileFeatures(tile_id, baseline_year, comparison_year, cube, transform, crs, h, w)


def flatten_feature_cube(feature_cube: np.ndarray) -> np.ndarray:
    c, h, w = feature_cube.shape
    return np.moveaxis(feature_cube, 0, -1).reshape(h * w, c)


def valid_feature_mask(feature_cube: np.ndarray) -> np.ndarray:
    return np.isfinite(np.moveaxis(feature_cube, 0, -1)).all(axis=-1)


def day_offset_for_year_start(epoch: date, year: int) -> int:
    return (date(year, 1, 1) - epoch).days


def _reproject_single_band(src_path: Path, dst_shape: tuple[int, int], dst_transform: Affine, dst_crs: object, dtype) -> np.ndarray:
    with rasterio.open(src_path) as src:
        dst = np.zeros(dst_shape, dtype=dtype)
        reproject(source=src.read(1), destination=dst, src_transform=src.transform, src_crs=src.crs, dst_transform=dst_transform, dst_crs=dst_crs, resampling=Resampling.nearest)
    return dst


def read_glads2_state(tile_id: str, labels_dir: Path, dst_shape: tuple[int, int], dst_transform: Affine, dst_crs: object, comparison_year: int) -> np.ndarray | None:
    alert_path = labels_dir / "glads2" / f"glads2_{tile_id}_alert.tif"
    date_path = labels_dir / "glads2" / f"glads2_{tile_id}_alertDate.tif"
    if not (alert_path.exists() and date_path.exists()):
        return None

    alert = _reproject_single_band(alert_path, dst_shape, dst_transform, dst_crs, np.uint8)
    alert_date = _reproject_single_band(date_path, dst_shape, dst_transform, dst_crs, np.uint16)
    cutoff = day_offset_for_year_start(GLADS2_EPOCH, comparison_year + 1)

    state = np.full(dst_shape, STATE_UNAVAILABLE, dtype=np.int8)
    happened = (alert_date > 0) & (alert_date < cutoff)

    state[(alert == 0) | ((alert_date > 0) & (alert_date >= cutoff))] = STATE_NEG
    state[(alert == 1) & happened] = STATE_UNCERTAIN
    state[(alert == 2) & happened] = STATE_UNCERTAIN

    # relaxed version: GLAD-S2=3 counts as supportive, not contradiction
    state[(alert == 3) & happened] = STATE_HARD_POS
    state[(alert == 4) & happened] = STATE_HARD_POS
    return state


def read_radd_state(tile_id: str, labels_dir: Path, dst_shape: tuple[int, int], dst_transform: Affine, dst_crs: object, comparison_year: int) -> np.ndarray | None:
    path = labels_dir / "radd" / f"radd_{tile_id}_labels.tif"
    if not path.exists():
        return None

    raw = _reproject_single_band(path, dst_shape, dst_transform, dst_crs, np.int32)
    confidence = raw // 10000
    day_offset = raw % 10000
    cutoff = day_offset_for_year_start(RADD_EPOCH, comparison_year + 1)

    state = np.full(dst_shape, STATE_UNAVAILABLE, dtype=np.int8)
    happened = (day_offset > 0) & (day_offset < cutoff)

    state[(raw == 0) | ((day_offset > 0) & (day_offset >= cutoff))] = STATE_NEG
    state[(confidence == 2) & happened] = STATE_SOFT_POS
    state[(confidence >= 3) & happened] = STATE_HARD_POS
    return state


def read_gladl_state(tile_id: str, labels_dir: Path, dst_shape: tuple[int, int], dst_transform: Affine, dst_crs: object, comparison_year: int) -> np.ndarray | None:
    state = np.full(dst_shape, STATE_UNAVAILABLE, dtype=np.int8)
    any_seen = False
    any_alert_by_t = np.zeros(dst_shape, dtype=bool)
    any_confirmed_by_t = np.zeros(dst_shape, dtype=bool)
    any_probable_by_t = np.zeros(dst_shape, dtype=bool)

    for yy in GLADL_YEARS:
        if yy > comparison_year:
            continue
        alert_path = labels_dir / "gladl" / f"gladl_{tile_id}_alert{str(yy)[-2:]}.tif"
        if not alert_path.exists():
            continue
        any_seen = True
        alert = _reproject_single_band(alert_path, dst_shape, dst_transform, dst_crs, np.uint8)
        any_alert_by_t |= alert > 0
        any_probable_by_t |= alert == 2
        any_confirmed_by_t |= alert >= 3

    if not any_seen:
        return None

    state[~any_alert_by_t] = STATE_NEG
    state[any_probable_by_t] = STATE_SOFT_POS
    state[any_confirmed_by_t] = STATE_HARD_POS
    return state


def source_state_to_single_source_target(state: np.ndarray) -> np.ndarray:
    out = np.full(state.shape, -1, dtype=np.int8)
    out[state == STATE_NEG] = 0
    out[state == STATE_HARD_POS] = 1
    return out


def fuse_consensus_states(states: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.stack(states, axis=0)
    available = stack >= 0
    any_available = np.any(available, axis=0)

    any_neg = np.any(stack == STATE_NEG, axis=0)
    any_hard = np.any(stack == STATE_HARD_POS, axis=0)
    any_soft = np.any(stack == STATE_SOFT_POS, axis=0)
    any_nonzero_support = np.any((stack == STATE_HARD_POS) | (stack == STATE_SOFT_POS) | (stack == STATE_UNCERTAIN), axis=0)

    all_available_neg = any_available & np.all(np.where(available, stack == STATE_NEG, True), axis=0)

    # Only explicit 0 counts as disagreement
    disagreement = any_hard & any_neg

    # Positive requires at least one hard positive and no explicit negative
    positive = any_hard & (~any_neg)

    negative = all_available_neg

    # Uncertain = some nonzero evidence exists, but no hard positive and not all-negative
    uncertainty = any_available & (~positive) & (~negative) & (~disagreement) & any_nonzero_support

    out = np.full(stack.shape[1:], -1, dtype=np.int8)
    out[negative] = 0
    out[positive] = 1
    return out, disagreement, uncertainty


def _pre2020_positive_glads2(tile_id: str, labels_dir: Path, dst_shape: tuple[int, int], dst_transform: Affine, dst_crs: object, cfg: TargetConfig) -> np.ndarray | None:
    alert_path = labels_dir / "glads2" / f"glads2_{tile_id}_alert.tif"
    date_path = labels_dir / "glads2" / f"glads2_{tile_id}_alertDate.tif"
    if not (alert_path.exists() and date_path.exists()):
        return None
    alert = _reproject_single_band(alert_path, dst_shape, dst_transform, dst_crs, np.uint8)
    alert_date = _reproject_single_band(date_path, dst_shape, dst_transform, dst_crs, np.uint16)
    cutoff_2021 = day_offset_for_year_start(GLADS2_EPOCH, 2021)
    return (alert >= cfg.pre2020_min_confidence_glads2) & (alert_date > 0) & (alert_date < cutoff_2021)


def _pre2020_positive_radd(tile_id: str, labels_dir: Path, dst_shape: tuple[int, int], dst_transform: Affine, dst_crs: object, cfg: TargetConfig) -> np.ndarray | None:
    path = labels_dir / "radd" / f"radd_{tile_id}_labels.tif"
    if not path.exists():
        return None
    raw = _reproject_single_band(path, dst_shape, dst_transform, dst_crs, np.int32)
    confidence = raw // 10000
    day_offset = raw % 10000
    cutoff_2021 = day_offset_for_year_start(RADD_EPOCH, 2021)
    return (confidence >= cfg.pre2020_min_confidence_radd) & (day_offset > 0) & (day_offset < cutoff_2021)


def pre2020_exclusion_mask(tile_id: str, labels_dir: Path, dst_shape: tuple[int, int], dst_transform: Affine, dst_crs: object, cfg: TargetConfig) -> tuple[np.ndarray, list[str]]:
    masks = []
    used = []
    for name, fn in [("radd", _pre2020_positive_radd), ("glads2", _pre2020_positive_glads2)]:
        m = fn(tile_id, labels_dir, dst_shape, dst_transform, dst_crs, cfg)
        if m is not None:
            masks.append(m)
            used.append(name)
    if not masks:
        return np.zeros(dst_shape, dtype=bool), []
    return np.any(np.stack(masks, axis=0), axis=0), used


def read_training_target(tile_id: str, labels_dir: Path, dst_shape: tuple[int, int], dst_transform: Affine, dst_crs: object, cfg: TargetConfig, comparison_year: int, baseline_year: int = 2020) -> tuple[np.ndarray | None, list[str], np.ndarray, np.ndarray, np.ndarray]:
    pre2020_mask, pre2020_sources = pre2020_exclusion_mask(tile_id, labels_dir, dst_shape, dst_transform, dst_crs, cfg)

    if cfg.label_source == "glads2":
        state = read_glads2_state(tile_id, labels_dir, dst_shape, dst_transform, dst_crs, comparison_year)
        if state is None:
            return None, [], pre2020_mask, np.zeros(dst_shape, dtype=bool), np.zeros(dst_shape, dtype=bool)
        t = source_state_to_single_source_target(state)
        t[pre2020_mask] = -1
        return t, ["glads2"] + ([f"pre2020_excl:{'+'.join(pre2020_sources)}"] if pre2020_sources else []), pre2020_mask, np.zeros(dst_shape, dtype=bool), np.zeros(dst_shape, dtype=bool)

    if cfg.label_source == "gladl":
        state = read_gladl_state(tile_id, labels_dir, dst_shape, dst_transform, dst_crs, comparison_year)
        if state is None:
            return None, [], pre2020_mask, np.zeros(dst_shape, dtype=bool), np.zeros(dst_shape, dtype=bool)
        t = source_state_to_single_source_target(state)
        t[pre2020_mask] = -1
        return t, ["gladl"] + ([f"pre2020_excl:{'+'.join(pre2020_sources)}"] if pre2020_sources else []), pre2020_mask, np.zeros(dst_shape, dtype=bool), np.zeros(dst_shape, dtype=bool)

    if cfg.label_source == "radd":
        state = read_radd_state(tile_id, labels_dir, dst_shape, dst_transform, dst_crs, comparison_year)
        if state is None:
            return None, [], pre2020_mask, np.zeros(dst_shape, dtype=bool), np.zeros(dst_shape, dtype=bool)
        t = source_state_to_single_source_target(state)
        t[pre2020_mask] = -1
        return t, ["radd"] + ([f"pre2020_excl:{'+'.join(pre2020_sources)}"] if pre2020_sources else []), pre2020_mask, np.zeros(dst_shape, dtype=bool), np.zeros(dst_shape, dtype=bool)

    available_states = []
    names = []
    for name, fn in [("radd", read_radd_state), ("gladl", read_gladl_state), ("glads2", read_glads2_state)]:
        s = fn(tile_id, labels_dir, dst_shape, dst_transform, dst_crs, comparison_year)
        if s is not None:
            available_states.append(s)
            names.append(name)
    if not available_states:
        return None, [], pre2020_mask, np.zeros(dst_shape, dtype=bool), np.zeros(dst_shape, dtype=bool)
    combined, disagreement_mask, uncertainty_mask = fuse_consensus_states(available_states)
    combined[pre2020_mask] = -1
    disagreement_mask &= ~pre2020_mask
    uncertainty_mask &= ~pre2020_mask
    if pre2020_sources:
        names = names + [f"pre2020_excl:{'+'.join(pre2020_sources)}"]
    return combined, names, pre2020_mask, disagreement_mask, uncertainty_mask


def extract_tile_ids_from_glads2(labels_dir: Path) -> set[str]:
    return {p.name[len("glads2_"):-len("_alert.tif")] for p in (labels_dir / "glads2").glob("glads2_*_alert.tif")}


def extract_tile_ids_from_radd(labels_dir: Path) -> set[str]:
    return {p.name[len("radd_"):-len("_labels.tif")] for p in (labels_dir / "radd").glob("radd_*_labels.tif")}


def extract_tile_ids_from_gladl(labels_dir: Path) -> set[str]:
    out = set()
    for p in (labels_dir / "gladl").glob("gladl_*_alert*.tif"):
        name = p.name
        if "alertDate" in name:
            continue
        out.add(name.split("_alert")[0][len("gladl_"):])
    return out


def discover_trainable_tile_ids(labels_dir: Path, year_map: dict[str, list[int]], baseline_year: int, label_source: str) -> list[str]:
    aef_tiles = {t for t, years in year_map.items() if baseline_year in years and any(y > baseline_year for y in years)}
    glads2_tiles = extract_tile_ids_from_glads2(labels_dir)
    gladl_tiles = extract_tile_ids_from_gladl(labels_dir)
    radd_tiles = extract_tile_ids_from_radd(labels_dir)
    if label_source == "glads2":
        label_tiles = glads2_tiles
    elif label_source == "gladl":
        label_tiles = gladl_tiles
    elif label_source == "radd":
        label_tiles = radd_tiles
    elif label_source == "consensus":
        label_tiles = glads2_tiles | gladl_tiles | radd_tiles
    else:
        raise ValueError(label_source)
    return sorted(aef_tiles & label_tiles)


def stratified_sample_indices(y: np.ndarray, max_pos: int, max_neg: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if len(pos_idx) > max_pos:
        pos_idx = rng.choice(pos_idx, size=max_pos, replace=False)
    if len(neg_idx) > max_neg:
        neg_idx = rng.choice(neg_idx, size=max_neg, replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(idx)
    return idx


def comparison_years_for_tile(years: list[int], baseline_year: int) -> list[int]:
    return [y for y in sorted(years) if y > baseline_year]


def build_training_table(tile_ids: Sequence[str], aef_dir: Path, labels_dir: Path, year_map: dict[str, list[int]], baseline_year: int, target_cfg: TargetConfig, max_pos_per_pair: int, max_neg_per_pair: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    xs, ys, groups, rows = [], [], [], []
    pair_counter = 0
    dropped_disagreement_total = 0
    dropped_uncertainty_total = 0
    for tile_i, tile_id in enumerate(tile_ids):
        for comparison_year in comparison_years_for_tile(year_map.get(tile_id, []), baseline_year):
            feat = build_feature_cube(tile_id, "train", aef_dir, baseline_year, comparison_year)
            target, sources_used, pre2020_mask, disagreement_mask, uncertainty_mask = read_training_target(tile_id, labels_dir, (feat.height, feat.width), feat.transform, feat.crs, target_cfg, comparison_year, baseline_year)
            if target is None:
                continue
            feature_valid = valid_feature_mask(feat.feature_cube)
            labeled_valid = feature_valid & (target >= 0)
            y_all = target[labeled_valid].reshape(-1).astype(np.int8)
            if y_all.size == 0:
                continue
            x_all = np.moveaxis(feat.feature_cube, 0, -1)[labeled_valid].astype(np.float32)
            idx = stratified_sample_indices(y_all, max_pos=max_pos_per_pair, max_neg=max_neg_per_pair, seed=seed + 1000 * tile_i + comparison_year)
            if len(idx) == 0:
                continue
            xs.append(x_all[idx])
            ys.append(y_all[idx])
            groups.append(np.full(len(idx), tile_id, dtype=object))
            disagreement_count = int(disagreement_mask.sum())
            uncertainty_count = int(uncertainty_mask.sum())
            dropped_disagreement_total += disagreement_count
            dropped_uncertainty_total += uncertainty_count
            rows.append({
                "tile_id": tile_id,
                "baseline_year": baseline_year,
                "comparison_year": comparison_year,
                "pair_id": f"{tile_id}__{baseline_year}_{comparison_year}",
                "total_pixels": int(target.size),
                "feature_valid_pixels": int(feature_valid.sum()),
                "positive_pixels_total": int((target == 1).sum()),
                "negative_pixels_total": int((target == 0).sum()),
                "ignored_pixels_total": int((target < 0).sum()),
                "pre2020_excluded_pixels_total": int(pre2020_mask.sum()),
                "disagreement_ignored_pixels_total": disagreement_count,
                "uncertainty_ignored_pixels_total": uncertainty_count,
                "sampled_pixels": int(len(idx)),
                "sampled_pos": int((y_all[idx] == 1).sum()),
                "sampled_neg": int((y_all[idx] == 0).sum()),
                "sources_used": "+".join(sources_used),
            })
            pair_counter += 1
            log(f"Built {tile_id} {baseline_year}->{comparison_year}: n={len(idx):,}, pos={(y_all[idx] == 1).sum():,}, neg={(y_all[idx] == 0).sum():,}, ignored={(target < 0).sum():,}, pre2020_excl={int(pre2020_mask.sum()):,}, disagreement_ignored={disagreement_count:,}, uncertainty_ignored={uncertainty_count:,}, sources={'+'.join(sources_used)}")
    if not xs:
        raise RuntimeError("No training samples built")
    log(f"Built {pair_counter} tile-year training pairs")
    log(f"Dropped by consensus filters | disagreement={dropped_disagreement_total:,} uncertainty={dropped_uncertainty_total:,}")
    return np.concatenate(xs, 0), np.concatenate(ys, 0), np.concatenate(groups, 0), pd.DataFrame(rows)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.uint8)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": int(len(y_true))}


def make_lgbm(seed: int, n_estimators: int = 500):
    if lgb is None:
        raise ImportError("lightgbm is not installed; use --classifier linear or install lightgbm")
    return lgb.LGBMClassifier(objective="binary", n_estimators=n_estimators, learning_rate=0.05, num_leaves=64, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1, random_state=seed, class_weight="balanced", verbosity=-1)


def make_linear(seed: int):
    return LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=seed, n_jobs=None)


def make_classifier(classifier: str, seed: int, n_estimators: int = 500):
    if classifier == "lightgbm":
        return make_lgbm(seed, n_estimators)
    if classifier == "linear":
        return make_linear(seed)
    raise ValueError(classifier)


def predict_proba_positive(model, x: np.ndarray, num_iteration: int | None = None) -> np.ndarray:
    if lgb is not None and isinstance(model, lgb.LGBMClassifier):
        return model.predict_proba(x, num_iteration=num_iteration)[:, 1]
    return model.predict_proba(x)[:, 1]


def cross_validate_classifier(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int, n_splits: int, threshold: float, classifier: str) -> tuple[pd.DataFrame, int | None]:
    unique_tiles = np.unique(groups)
    actual_splits = min(n_splits, len(unique_tiles))
    if actual_splits < 2:
        raise ValueError("Need at least 2 labeled tiles for CV")
    rows, best_iterations = [], []
    cv = GroupKFold(n_splits=actual_splits)
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(x, y, groups), start=1):
        model = make_classifier(classifier, seed + fold_idx, 500)
        if classifier == "lightgbm":
            model.fit(x[train_idx], y[train_idx], eval_set=[(x[val_idx], y[val_idx])], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(50, verbose=False)])
            best_iter = int(model.best_iteration_ or model.n_estimators)
            best_iterations.append(best_iter)
        else:
            model.fit(x[train_idx], y[train_idx])
            best_iter = None
        y_prob = predict_proba_positive(model, x[val_idx], num_iteration=best_iter)
        metrics = compute_binary_metrics(y[val_idx], y_prob, threshold)
        metrics.update({"fold": fold_idx, "train_tiles": ",".join(sorted(np.unique(groups[train_idx]).tolist())), "val_tiles": ",".join(sorted(np.unique(groups[val_idx]).tolist())), "best_iteration": best_iter if best_iter is not None else -1, "classifier": classifier})
        rows.append(metrics)
        log(f"Fold {fold_idx}/{actual_splits} | precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} iou={metrics['iou']:.4f}")
    final_param = int(np.median(best_iterations)) if best_iterations else None
    return pd.DataFrame(rows), final_param


def summarize_cv_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"n_folds": int(len(fold_df)), "precision_mean": float(fold_df["precision"].mean()), "precision_std": float(fold_df["precision"].std(ddof=0)), "recall_mean": float(fold_df["recall"].mean()), "recall_std": float(fold_df["recall"].std(ddof=0)), "f1_mean": float(fold_df["f1"].mean()), "f1_std": float(fold_df["f1"].std(ddof=0)), "iou_mean": float(fold_df["iou"].mean()), "iou_std": float(fold_df["iou"].std(ddof=0))}])


def train_final_model(x: np.ndarray, y: np.ndarray, seed: int, classifier: str, n_estimators: int | None):
    model = make_lgbm(seed, int(n_estimators or 500)) if classifier == "lightgbm" else make_linear(seed)
    model.fit(x, y)
    return model

def feature_importance_df(model) -> pd.DataFrame:
    if lgb is not None and isinstance(model, lgb.LGBMClassifier):
        n_features = model.booster_.num_feature()
        return pd.DataFrame({
            "feature": [f"feature_{i:03d}" for i in range(n_features)],
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }).sort_values("importance_gain", ascending=False)
    coef = np.ravel(model.coef_)
    return pd.DataFrame({
        "feature": [f"feature_{i:03d}" for i in range(len(coef))],
        "coefficient": coef,
        "abs_coefficient": np.abs(coef),
    }).sort_values("abs_coefficient", ascending=False)

def save_binary_geotiff(mask: np.ndarray, reference: TileFeatures, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "driver": "GTiff",
        "height": reference.height,
        "width": reference.width,
        "count": 1,
        "dtype": "uint8",
        "crs": reference.crs,
        "transform": reference.transform,
        "compress": "deflate",
        "nodata": 0,
    }
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mask.astype(np.uint8), 1)


def save_probability_geotiff(proba: np.ndarray, reference: TileFeatures, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "driver": "GTiff",
        "height": reference.height,
        "width": reference.width,
        "count": 1,
        "dtype": "float32",
        "crs": reference.crs,
        "transform": reference.transform,
        "compress": "deflate",
        "nodata": np.nan,
    }
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(proba.astype(np.float32), 1)


def raster_mask_to_gdf(mask: np.ndarray, transform: Affine, crs: object, min_area_ha: float) -> gpd.GeoDataFrame:
    geoms = []
    for geom, value in shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=transform):
        if int(value) == 1:
            geoms.append(shape(geom))

    if not geoms:
        return gpd.GeoDataFrame({"value": []}, geometry=[], crs=crs).to_crs("EPSG:4326")

    gdf = gpd.GeoDataFrame({"value": [1] * len(geoms)}, geometry=geoms, crs=crs).to_crs("EPSG:4326")

    if min_area_ha > 0 and len(gdf):
        gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
        gdf = gdf.loc[((gdf_utm.area / 10_000.0) >= min_area_ha).values].reset_index(drop=True)

    return gdf



def run_train(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aef_dir = data_root / "aef-embeddings"
    labels_dir = data_root / "labels" / "train"
    metadata_root = data_root / "metadata"
    year_map = discover_aef_years(aef_dir, "train")
    requested_train_tile_ids = discover_tile_ids(aef_dir, "train", metadata_root if metadata_root.exists() else None)
    available_train_tile_ids = set(discover_trainable_tile_ids(labels_dir, year_map, args.baseline_year, args.label_source))
    train_tile_ids = [t for t in requested_train_tile_ids if t in available_train_tile_ids]
    log(f"Using {len(train_tile_ids)} train tiles with label_source={args.label_source}")
    target_cfg = TargetConfig(args.label_source, args.min_confidence_glads2, args.min_confidence_gladl, args.min_confidence_radd, args.pre2020_min_confidence_glads2, args.pre2020_min_confidence_radd)
    x, y, groups, summary = build_training_table(train_tile_ids, aef_dir, labels_dir, year_map, args.baseline_year, target_cfg, args.max_pos_per_pair, args.max_neg_per_pair, args.seed)
    log(f"Training pairs summary totals | pos={summary['positive_pixels_total'].sum():,} neg={summary['negative_pixels_total'].sum():,} ignored={summary['ignored_pixels_total'].sum():,} pre2020_excl={summary['pre2020_excluded_pixels_total'].sum():,} disagreement_ignored={summary['disagreement_ignored_pixels_total'].sum():,} uncertainty_ignored={summary['uncertainty_ignored_pixels_total'].sum():,} sampled={summary['sampled_pixels'].sum():,}")
    fold_df, final_n_estimators = cross_validate_classifier(x, y, groups, args.seed, args.cv_folds, args.eval_threshold, args.classifier)
    cv_summary_df = summarize_cv_metrics(fold_df)
    final_model = train_final_model(x, y, args.seed, args.classifier, final_n_estimators)
    model_path = output_dir / ("lightgbm_model.joblib" if args.classifier == "lightgbm" else "linear_model.joblib")
    joblib.dump(final_model, model_path)
    summary.to_csv(output_dir / "training_pair_summary.csv", index=False)
    fold_df.to_csv(output_dir / "cv_fold_metrics.csv", index=False)
    cv_summary_df.to_csv(output_dir / "cv_summary_metrics.csv", index=False)
    feature_importance_df(final_model).to_csv(output_dir / "feature_importance.csv", index=False)
    safe_dump_args(args, output_dir / "train_run_config.json")
    if args.save_train_preview:
        save_train_preview_artifacts(final_model, train_tile_ids, data_root, output_dir, aef_dir, labels_dir, year_map, args.baseline_year, target_cfg, args.threshold, args.min_area_ha, args.n_train_preview, args.seed, args.preview_year)
    row = cv_summary_df.iloc[0]
    log(f"CV summary: precision={row['precision_mean']:.4f}±{row['precision_std']:.4f} recall={row['recall_mean']:.4f}±{row['recall_std']:.4f} f1={row['f1_mean']:.4f}±{row['f1_std']:.4f} iou={row['iou_mean']:.4f}±{row['iou_std']:.4f}")

def predict_tile(model, tile_id: str, split: str, aef_dir: Path, year_map: dict[str, list[int]], baseline_year: int, threshold: float, output_dir: Path, save_proba: bool, min_area_ha: float, prediction_year: str) -> dict[str, object]:
    years = year_map[tile_id]
    comparison_year = max([y for y in years if y > baseline_year]) if prediction_year == "latest" else int(prediction_year)

    feat = build_feature_cube(tile_id, split, aef_dir, baseline_year, comparison_year)
    x = flatten_feature_cube(feat.feature_cube)
    valid = np.isfinite(x).all(axis=1)

    proba_flat = np.zeros(x.shape[0], dtype=np.float32)
    proba_flat[valid] = predict_proba_positive(model, x[valid]).astype(np.float32)
    proba = proba_flat.reshape(feat.height, feat.width)

    mask = (proba >= threshold).astype(np.uint8)

    raster_dir = output_dir / split / "rasters"
    vector_dir = output_dir / split / "geojson"

    if save_proba:
        save_probability_geotiff(proba, feat, raster_dir / f"{tile_id}_proba.tif")
    save_binary_geotiff(mask, feat, raster_dir / f"{tile_id}_mask.tif")

    gdf = raster_mask_to_gdf(mask, feat.transform, feat.crs, min_area_ha)
    vector_dir.mkdir(parents=True, exist_ok=True)

    features = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type not in {"Polygon", "MultiPolygon"}:
            continue
        features.append({
            "type": "Feature",
            "properties": {},
            "geometry": geom.__geo_interface__,
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(vector_dir / f"pred_{tile_id}.geojson", "w", encoding="utf-8") as f:
        json.dump(geojson, f)

    return {
        "tile_id": tile_id,
        "split": split,
        "comparison_year": comparison_year,
        "positive_pixels": int(mask.sum()),
        "height": feat.height,
        "width": feat.width,
    }


def run_predict(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aef_dir = data_root / "aef-embeddings"
    split = args.split
    year_map = discover_aef_years(aef_dir, split)
    tile_ids = sorted(year_map.keys())
    lightgbm_path = output_dir / "lightgbm_model.joblib"
    linear_path = output_dir / "linear_model.joblib"
    if lightgbm_path.exists():
        model = joblib.load(lightgbm_path)
    elif linear_path.exists():
        model = joblib.load(linear_path)
    else:
        raise FileNotFoundError(f"No saved model found in {output_dir}; expected {lightgbm_path.name} or {linear_path.name}")
    rows = []
    for i, tile_id in enumerate(tile_ids, start=1):
        if args.baseline_year not in year_map[tile_id] or not any(y > args.baseline_year for y in year_map[tile_id]):
            continue
        log(f"[{i}/{len(tile_ids)}] Predicting {split} tile {tile_id}")
        rows.append(predict_tile(model, tile_id, split, aef_dir, year_map, args.baseline_year, args.threshold, output_dir, args.save_probabilities, args.min_area_ha, args.prediction_year))
    pd.DataFrame(rows).to_csv(output_dir / f"{split}_prediction_summary.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AEF temporal-pairs pipeline with stricter consensus fusion and disagreement/uncertainty logging")
    sub = parser.add_subparsers(dest="command", required=True)
    def add_shared(p: argparse.ArgumentParser):
        p.add_argument("--data-root", required=True)
        p.add_argument("--output-dir", required=True)
        p.add_argument("--baseline-year", type=int, default=2020)
        p.add_argument("--seed", type=int, default=42)
    def add_trainish(p: argparse.ArgumentParser):
        p.add_argument("--classifier", default="lightgbm", choices=["lightgbm", "linear"])
        p.add_argument("--label-source", default="consensus", choices=["consensus", "radd", "gladl", "glads2"])
        p.add_argument("--min-confidence-glads2", type=int, default=4, choices=[1,2,3,4])
        p.add_argument("--min-confidence-gladl", type=int, default=3, choices=[2,3])
        p.add_argument("--min-confidence-radd", type=int, default=3, choices=[2,3])
        p.add_argument("--pre2020-min-confidence-glads2", type=int, default=3, choices=[1,2,3,4])
        p.add_argument("--pre2020-min-confidence-radd", type=int, default=3, choices=[2,3])
        p.add_argument("--max-pos-per-pair", type=int, default=50000)
        p.add_argument("--max-neg-per-pair", type=int, default=50000)
        p.add_argument("--cv-folds", type=int, default=5)
        p.add_argument("--eval-threshold", type=float, default=0.5)
        p.add_argument("--threshold", type=float, default=0.5)
        p.add_argument("--save-train-preview", action="store_true")
        p.add_argument("--n-train-preview", type=int, default=5)
        p.add_argument("--preview-year", default="latest")
        p.add_argument("--min-area-ha", type=float, default=0.0)
    train_p = sub.add_parser("train")
    add_shared(train_p)
    add_trainish(train_p)
    train_p.set_defaults(func=run_train)
    pred_p = sub.add_parser("predict")
    add_shared(pred_p)
    pred_p.add_argument("--split", default="test", choices=["train","test"])
    pred_p.add_argument("--threshold", type=float, default=0.5)
    pred_p.add_argument("--prediction-year", default="latest")
    pred_p.add_argument("--save-probabilities", action="store_true")
    pred_p.add_argument("--min-area-ha", type=float, default=0.0)
    pred_p.set_defaults(func=run_predict)
    both_p = sub.add_parser("fit-predict")
    add_shared(both_p)
    add_trainish(both_p)
    both_p.add_argument("--prediction-year", default="latest")
    both_p.add_argument("--save-probabilities", action="store_true")
    def run_both(args: argparse.Namespace):
        run_train(args)
        pred_args = argparse.Namespace(**{k:v for k,v in vars(args).items() if not callable(v)})
        pred_args.split = "test"
        run_predict(pred_args)
    both_p.set_defaults(func=run_both)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
