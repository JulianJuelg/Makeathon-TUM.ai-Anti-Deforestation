"""Data loading for patch-based deforestation segmentation."""
from __future__ import annotations

import datetime
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import rasterio
from rasterio.warp import reproject, Resampling

_DEFAULT_DATA_DIR = "data/makeathon-challenge"

_RADD_EPOCH   = datetime.date(2014, 12, 31)
_GLADS2_EPOCH = datetime.date(2019, 1, 1)
_BASELINE_YEAR = 2020
_GLADL_YEARS = range(2021, 2026)

STATE_UNAVAILABLE = -2
STATE_NEG = 0
STATE_HARD_POS = 1
STATE_SOFT_POS = 2
STATE_UNCERTAIN = 3

_DEFAULT_LABEL_THRESHOLDS = {
    "radd_positive_conf": 3,
    "radd_ignore_conf": 2,
    "glads2_min_conf": 3,
    "gladl_min_conf": 3,
}

_DEFAULT_FILTER_MODE = "strict"
_ALLOWED_FILTER_MODES = {"strict", "normal", "mild"}


def _resolve_label_thresholds(label_thresholds: dict | None = None) -> dict:
    resolved = dict(_DEFAULT_LABEL_THRESHOLDS)
    if label_thresholds:
        resolved.update({k: int(v) for k, v in label_thresholds.items()})
    return resolved


def _resolve_filter_mode(filter_mode: str | None = None) -> str:
    mode = (filter_mode or _DEFAULT_FILTER_MODE).strip().lower()
    if mode not in _ALLOWED_FILTER_MODES:
        raise ValueError(
            f"Unsupported filter_mode='{filter_mode}'. "
            f"Expected one of: {sorted(_ALLOWED_FILTER_MODES)}"
        )
    return mode


def _label_threshold_cache_key(label_thresholds: dict, filter_mode: str) -> str:
    return (
        f"mode_{filter_mode}_"
        f"raddp{label_thresholds['radd_positive_conf']}"
        f"_raddi{label_thresholds['radd_ignore_conf']}"
        f"_gs2{label_thresholds['glads2_min_conf']}"
        f"_gladl{label_thresholds['gladl_min_conf']}"
    )


def _days_to_year(days_array: np.ndarray, epoch: datetime.date) -> np.ndarray:
    day_to_year: dict[int, int] = {}
    for d in np.unique(days_array[days_array > 0]):
        day_to_year[int(d)] = (epoch + datetime.timedelta(days=int(d))).year
    year_arr = np.zeros_like(days_array)
    for d, y in day_to_year.items():
        year_arr[days_array == d] = y
    return year_arr


def _get_dst_grid(tile: str, aef_dir: str):
    """Read the target grid from the tile's 2020 AEF embedding."""
    path = os.path.join(aef_dir, f"{tile}_2020.tiff")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No AEF 2020 file for tile {tile} under {aef_dir}")
    with rasterio.open(path) as src:
        return src.transform, src.crs, (src.height, src.width)


def _post_baseline_mask(year_arr: np.ndarray) -> np.ndarray:
    """Return events that happened after the 2020 baseline imagery."""
    return year_arr > _BASELINE_YEAR


def _day_offset_for_year_start(epoch: datetime.date, year: int) -> int:
    return (datetime.date(year, 1, 1) - epoch).days


def _reproject_single_band(path: str, dst_transform, dst_crs, dst_shape, dtype) -> np.ndarray:
    with rasterio.open(path) as src:
        out = np.zeros(dst_shape, dtype=dtype)
        reproject(
            source=src.read(1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
    return out


def _read_glads2_state(tile: str, labels_dir: str, dst_transform, dst_crs, dst_shape,
                       comparison_year: int) -> np.ndarray | None:
    alert_path = os.path.join(labels_dir, "glads2", f"glads2_{tile}_alert.tif")
    date_path = os.path.join(labels_dir, "glads2", f"glads2_{tile}_alertDate.tif")
    if not (os.path.exists(alert_path) and os.path.exists(date_path)):
        return None

    alert = _reproject_single_band(alert_path, dst_transform, dst_crs, dst_shape, np.uint8)
    alert_date = _reproject_single_band(date_path, dst_transform, dst_crs, dst_shape, np.uint16)
    cutoff = _day_offset_for_year_start(_GLADS2_EPOCH, comparison_year + 1)
    happened = (alert_date > 0) & (alert_date < cutoff)

    state = np.full(dst_shape, STATE_UNAVAILABLE, dtype=np.int8)
    state[(alert == 0) | ((alert_date > 0) & (alert_date >= cutoff))] = STATE_NEG
    state[(alert == 1) & happened] = STATE_UNCERTAIN
    state[(alert == 2) & happened] = STATE_UNCERTAIN
    state[(alert == 3) & happened] = STATE_HARD_POS
    state[(alert == 4) & happened] = STATE_HARD_POS
    return state


def _read_radd_state(tile: str, labels_dir: str, dst_transform, dst_crs, dst_shape,
                     comparison_year: int) -> np.ndarray | None:
    path = os.path.join(labels_dir, "radd", f"radd_{tile}_labels.tif")
    if not os.path.exists(path):
        return None

    raw = _reproject_single_band(path, dst_transform, dst_crs, dst_shape, np.int32)
    confidence = raw // 10000
    day_offset = raw % 10000
    cutoff = _day_offset_for_year_start(_RADD_EPOCH, comparison_year + 1)
    happened = (day_offset > 0) & (day_offset < cutoff)

    state = np.full(dst_shape, STATE_UNAVAILABLE, dtype=np.int8)
    state[(raw == 0) | ((day_offset > 0) & (day_offset >= cutoff))] = STATE_NEG
    state[(confidence == 2) & happened] = STATE_SOFT_POS
    state[(confidence >= 3) & happened] = STATE_HARD_POS
    return state


def _read_gladl_state(tile: str, labels_dir: str, dst_transform, dst_crs, dst_shape,
                      comparison_year: int) -> np.ndarray | None:
    state = np.full(dst_shape, STATE_UNAVAILABLE, dtype=np.int8)
    any_seen = False
    any_alert_by_t = np.zeros(dst_shape, dtype=bool)
    any_probable_by_t = np.zeros(dst_shape, dtype=bool)
    any_confirmed_by_t = np.zeros(dst_shape, dtype=bool)

    for year in _GLADL_YEARS:
        if year > comparison_year:
            continue
        alert_path = os.path.join(labels_dir, "gladl", f"gladl_{tile}_alert{str(year)[-2:]}.tif")
        if not os.path.exists(alert_path):
            continue
        any_seen = True
        alert = _reproject_single_band(alert_path, dst_transform, dst_crs, dst_shape, np.uint8)
        any_alert_by_t |= alert > 0
        any_probable_by_t |= alert == 2
        any_confirmed_by_t |= alert >= 3

    if not any_seen:
        return None

    state[~any_alert_by_t] = STATE_NEG
    state[any_probable_by_t] = STATE_SOFT_POS
    state[any_confirmed_by_t] = STATE_HARD_POS
    return state


def _fuse_consensus_states(states: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.stack(states, axis=0)
    available = stack >= 0
    any_available = np.any(available, axis=0)

    any_neg = np.any(stack == STATE_NEG, axis=0)
    any_hard = np.any(stack == STATE_HARD_POS, axis=0)
    any_nonzero_support = np.any(
        (stack == STATE_HARD_POS) | (stack == STATE_SOFT_POS) | (stack == STATE_UNCERTAIN),
        axis=0,
    )
    all_available_neg = any_available & np.all(np.where(available, stack == STATE_NEG, True), axis=0)

    disagreement = any_hard & any_neg
    positive = any_hard & ~any_neg
    negative = all_available_neg
    uncertainty = any_available & ~positive & ~negative & ~disagreement & any_nonzero_support

    out = np.full(stack.shape[1:], -1, dtype=np.int8)
    out[negative] = 0
    out[positive] = 1
    return out, disagreement, uncertainty


def _pre2020_exclusion_mask(tile: str, labels_dir: str, dst_transform, dst_crs, dst_shape,
                            label_thresholds: dict) -> np.ndarray:
    masks = []

    radd_path = os.path.join(labels_dir, "radd", f"radd_{tile}_labels.tif")
    if os.path.exists(radd_path):
        raw = _reproject_single_band(radd_path, dst_transform, dst_crs, dst_shape, np.int32)
        confidence = raw // 10000
        day_offset = raw % 10000
        cutoff_2021 = _day_offset_for_year_start(_RADD_EPOCH, 2021)
        masks.append(
            (confidence >= label_thresholds["radd_positive_conf"])
            & (day_offset > 0)
            & (day_offset < cutoff_2021)
        )

    gs2_alert_path = os.path.join(labels_dir, "glads2", f"glads2_{tile}_alert.tif")
    gs2_date_path = os.path.join(labels_dir, "glads2", f"glads2_{tile}_alertDate.tif")
    if os.path.exists(gs2_alert_path) and os.path.exists(gs2_date_path):
        alert = _reproject_single_band(gs2_alert_path, dst_transform, dst_crs, dst_shape, np.uint8)
        alert_date = _reproject_single_band(gs2_date_path, dst_transform, dst_crs, dst_shape, np.uint16)
        cutoff_2021 = _day_offset_for_year_start(_GLADS2_EPOCH, 2021)
        masks.append(
            (alert >= label_thresholds["glads2_min_conf"])
            & (alert_date > 0)
            & (alert_date < cutoff_2021)
        )

    if not masks:
        return np.zeros(dst_shape, dtype=bool)
    return np.any(np.stack(masks, axis=0), axis=0)


def _build_consensus_targets(tile: str, dst_transform, dst_crs, dst_shape,
                             labels_dir: str, cache_dir: str | None = None,
                             label_thresholds: dict | None = None,
                             filter_mode: str = _DEFAULT_FILTER_MODE,
                             target_years: list[int] | None = None
                             ) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    label_thresholds = _resolve_label_thresholds(label_thresholds)
    filter_mode = _resolve_filter_mode(filter_mode)
    years = sorted(set(target_years or list(_GLADL_YEARS)))

    if cache_dir:
        cache_key = _label_threshold_cache_key(label_thresholds, filter_mode)
        years_key = "-".join(str(y) for y in years)
        cache_file = os.path.join(
            cache_dir,
            "def_year_v8_consensus_pre2020",
            cache_key,
            f"{tile}_{years_key}.npz",
        )
        if os.path.exists(cache_file):
            cached = np.load(cache_file)
            def_year = cached["def_year"]
            ignore_mask = cached["ignore_mask"].astype(bool)
            targets = {
                int(k.split("_")[1]): cached[k].astype(np.int8)
                for k in cached.files
                if k.startswith("target_") and k.split("_")[1].isdigit()
            }
            ignores = {
                int(k.split("_")[1]): cached[k].astype(bool)
                for k in cached.files
                if k.startswith("ignore_") and k.split("_")[1].isdigit()
            }
            return def_year, ignore_mask, targets, ignores

    pre2020_mask = _pre2020_exclusion_mask(
        tile, labels_dir, dst_transform, dst_crs, dst_shape, label_thresholds
    )
    def_year = np.full(dst_shape, 9999, dtype=np.int32)
    targets: dict[int, np.ndarray] = {}
    ignores: dict[int, np.ndarray] = {}
    max_target = np.full(dst_shape, -1, dtype=np.int8)

    for year in years:
        states = []
        for reader in (_read_radd_state, _read_gladl_state, _read_glads2_state):
            state = reader(tile, labels_dir, dst_transform, dst_crs, dst_shape, year)
            if state is not None:
                states.append(state)
        if not states:
            target = np.full(dst_shape, -1, dtype=np.int8)
            disagreement = np.zeros(dst_shape, dtype=bool)
            uncertainty = np.zeros(dst_shape, dtype=bool)
        else:
            target, disagreement, uncertainty = _fuse_consensus_states(states)

        target = target.copy()
        target[pre2020_mask] = -1
        disagreement &= ~pre2020_mask
        uncertainty &= ~pre2020_mask
        ignore = (target < 0) | disagreement | uncertainty
        target[ignore] = -1

        newly_positive = (target == 1) & (def_year == 9999)
        def_year[newly_positive] = year
        targets[year] = target
        ignores[year] = ignore
        max_target = target

    ignore_mask = max_target < 0

    if cache_dir:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        payload = {
            "def_year": def_year,
            "ignore_mask": ignore_mask.astype(np.uint8),
        }
        payload.update({f"target_{year}": target for year, target in targets.items()})
        payload.update({f"ignore_{year}": ignore.astype(np.uint8) for year, ignore in ignores.items()})
        np.savez_compressed(cache_file, **payload)
    return def_year, ignore_mask, targets, ignores


def _build_def_year_and_ignore(tile: str, dst_transform, dst_crs, dst_shape,
                               labels_dir: str, cache_dir: str | None = None,
                               label_thresholds: dict | None = None,
                               filter_mode: str = _DEFAULT_FILTER_MODE) -> tuple[np.ndarray, np.ndarray]:
    def_year, ignore_mask, _, _ = _build_consensus_targets(
        tile,
        dst_transform,
        dst_crs,
        dst_shape,
        labels_dir,
        cache_dir=cache_dir,
        label_thresholds=label_thresholds,
        filter_mode=filter_mode,
        target_years=list(_GLADL_YEARS),
    )
    return def_year, ignore_mask


def _load_aef_flat(tile: str, year: int, dst_transform, dst_crs, dst_shape,
                   aef_dir: str, cache_dir: str | None = None) -> np.ndarray | None:
    """Read and reproject AEF embeddings to the target grid as (H*W, C)."""
    if cache_dir:
        cache_file = os.path.join(cache_dir, "aef_flat_v2_aefgrid", f"{tile}_{year}.npy")
        if os.path.exists(cache_file):
            return np.load(cache_file, mmap_mode='r')

    path = os.path.join(aef_dir, f"{tile}_{year}.tiff")
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        bands = src.read().astype(np.float32)
        at, ac = src.transform, src.crs

    cube = np.full((64,) + dst_shape, np.nan, dtype=np.float32)
    for b in range(64):
        reproject(source=bands[b], destination=cube[b],
                  src_transform=at, src_crs=ac,
                  dst_transform=dst_transform, dst_crs=dst_crs,
                  resampling=Resampling.bilinear)
    result = cube.reshape(64, -1).T

    if cache_dir:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.save(cache_file, result)
    return result


class DeforestationPatchDataset(Dataset):
    """Patch dataset for U-Net training."""
    def __init__(self, tiles: list[str], target_years: list[int],
                 aef_dict: dict, def_year_dict: dict, ignore_mask_dict: dict,
                 target_by_year_dict: dict | None = None,
                 ignore_by_year_dict: dict | None = None,
                 patch_size: int = 128, max_patches_per_tile_year: int = 500,
                 require_both_classes: bool = True):
        self.patch_size = patch_size
        self.patches = []
        target_by_year_dict = target_by_year_dict or {}
        ignore_by_year_dict = ignore_by_year_dict or {}
        
        for tile in tiles:
            if tile not in def_year_dict:
                continue
            def_year = def_year_dict[tile]
            ignore_mask = ignore_mask_dict.get(tile, np.zeros_like(def_year, dtype=bool))
            
            for yr in target_years:
                if (tile, 2020) not in aef_dict or (tile, yr) not in aef_dict:
                    continue

                target_map = target_by_year_dict.get((tile, yr))
                year_ignore = ignore_by_year_dict.get((tile, yr), ignore_mask)
                valid = ~year_ignore
                margin = patch_size // 2
                valid[:margin, :] = False
                valid[-margin:, :] = False
                valid[:, :margin] = False
                valid[:, -margin:] = False
                
                if target_map is not None:
                    labels = (target_map == 1).astype(np.int32)
                    valid &= target_map >= 0
                else:
                    labels = (def_year <= yr).astype(np.int32)
                pos_idx = np.where(valid & (labels == 1))
                neg_idx = np.where(valid & (labels == 0))
                
                n_pos = len(pos_idx[0])
                n_neg = len(neg_idx[0])
                if require_both_classes and (n_pos == 0 or n_neg == 0):
                    continue
                if n_pos == 0 and n_neg == 0:
                    continue
                rng = np.random.default_rng(42 + yr)

                if require_both_classes:
                    n_pos_sample = min(n_pos, n_neg, max_patches_per_tile_year // 2)
                    n_neg_sample = n_pos_sample
                else:
                    n_pos_sample = min(n_pos, max_patches_per_tile_year // 2)
                    n_neg_sample = min(n_neg, max_patches_per_tile_year - n_pos_sample)
                    if n_neg_sample == 0:
                        n_pos_sample = min(n_pos, max_patches_per_tile_year)

                pos_sel = rng.choice(n_pos, n_pos_sample, replace=False) if n_pos_sample > 0 else []
                neg_sel = rng.choice(n_neg, n_neg_sample, replace=False) if n_neg_sample > 0 else []

                for idx in pos_sel:
                    self.patches.append((tile, yr, pos_idx[0][idx], pos_idx[1][idx]))
                for idx in neg_sel:
                    self.patches.append((tile, yr, neg_idx[0][idx], neg_idx[1][idx]))
                    
        self.aef_dict = aef_dict
        self.def_year_dict = def_year_dict
        self.ignore_mask_dict = ignore_mask_dict
        self.target_by_year_dict = target_by_year_dict
        self.ignore_by_year_dict = ignore_by_year_dict

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        tile, yr, y0, x0 = self.patches[idx]
        
        aef2020 = self.aef_dict[(tile, 2020)]  # (64, H, W)
        aefyr   = self.aef_dict[(tile, yr)]    # (64, H, W)
        def_year = self.def_year_dict[tile]    # (H, W)
        ignore_mask = self.ignore_mask_dict.get(tile)
        target_map = self.target_by_year_dict.get((tile, yr))
        year_ignore = self.ignore_by_year_dict.get((tile, yr), ignore_mask)
        
        r0 = y0 - self.patch_size // 2
        r1 = r0 + self.patch_size
        c0 = x0 - self.patch_size // 2
        c1 = c0 + self.patch_size
        
        x_2020 = aef2020[:, r0:r1, c0:c1]
        x_yr   = aefyr[:, r0:r1, c0:c1]
        y_def  = def_year[r0:r1, c0:c1]
        target = target_map[r0:r1, c0:c1] if target_map is not None else None
        ignore = year_ignore[r0:r1, c0:c1] if year_ignore is not None else np.zeros_like(y_def, dtype=bool)
        
        # Valid mask: True if both AEF2020 and AEF_yr have no NaNs across all channels
        mask = np.all(np.isfinite(x_2020), axis=0) & np.all(np.isfinite(x_yr), axis=0)
        mask &= ~ignore
        mask = mask[None, :, :] # (1, P, P)
        
        delta = x_yr - x_2020
        x = np.concatenate([x_2020, x_yr, delta], axis=0).astype(np.float32)  # (192, P, P)
        if target is not None:
            y = (target == 1).astype(np.float32)[None, :, :]
        else:
            y = (y_def <= yr).astype(np.float32)[None, :, :]  # (1, P, P)
        
        x = np.nan_to_num(x)  # Fill NaNs
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)


# ── LightningDataModule ───────────────────────────────────────────────────────
class DeforestationDataModule(pl.LightningDataModule):
    """LightningDataModule for patch-based segmentation training."""

    def __init__(
        self,
        data_dir: str = _DEFAULT_DATA_DIR,
        train_tiles: list[str] | None = None,
        val_tiles: list[str] | None = None,
        target_years: list[int] | None = None,
        max_pixels_per_tile_year: int = 50_000,
        patch_size: int = 128,
        selected_bands: list[int] | None = None,
        filter_mode: str = _DEFAULT_FILTER_MODE,
        label_thresholds: dict | None = None,
        batch_size: int = 4096,
        num_workers: int = 4,
        cache_dir: str = "cache",
        spatial_eval_every_n_epochs: int = 5,
    ):
        super().__init__()
        self.data_dir   = data_dir
        self.labels_dir = os.path.join(data_dir, "labels", "train")
        self.aef_dir    = os.path.join(data_dir, "aef-embeddings", "train")
        self.cache_dir  = cache_dir
        self.spatial_eval_every_n_epochs = spatial_eval_every_n_epochs

        self._train_tiles = train_tiles
        self._val_tiles   = val_tiles
        self.target_years = target_years or [2021, 2022, 2023, 2024, 2025]
        self.max_pixels   = max_pixels_per_tile_year
        self.patch_size   = patch_size
        self.selected_bands = selected_bands
        self.filter_mode = _resolve_filter_mode(filter_mode)
        self.label_thresholds = _resolve_label_thresholds(label_thresholds)
        self.batch_size   = batch_size
        self.num_workers  = num_workers

        self.train_dataset: DeforestationPatchDataset | None = None
        self.val_dataset: DeforestationPatchDataset | None = None
        self.val_tiles_resolved: list[str] = []

        self.save_hyperparameters()

    def _discover_tiles(self) -> list[str]:
        pattern = os.path.join(self.labels_dir, "radd", "radd_*_labels.tif")
        return sorted([
            os.path.basename(f).replace("radd_", "").replace("_labels.tif", "")
            for f in glob.glob(pattern)
        ])

    def setup(self, stage: str | None = None):
        all_tiles = self._discover_tiles()
        if not all_tiles:
            raise RuntimeError(f"No RADD tiles found under {self.labels_dir}/radd/")

        if self._train_tiles is None and self._val_tiles is None:
            n_val = max(1, len(all_tiles) // 5)
            train_tiles = all_tiles[:-n_val]
            val_tiles   = all_tiles[-n_val:]
        else:
            train_tiles = self._train_tiles or []
            val_tiles   = self._val_tiles   or []

        print(f"[DataModule] Train tiles ({len(train_tiles)}): {train_tiles}")
        print(f"[DataModule] Val   tiles ({len(val_tiles)}):   {val_tiles}")
        self.val_tiles_resolved = val_tiles

        if stage in ("fit", None):
            aef_dict, def_year_dict, ignore_mask_dict, target_by_year_dict, ignore_by_year_dict = self._build_patch_arrays(train_tiles + val_tiles)
            self.train_dataset = DeforestationPatchDataset(
                train_tiles, self.target_years, aef_dict, def_year_dict, ignore_mask_dict,
                target_by_year_dict=target_by_year_dict,
                ignore_by_year_dict=ignore_by_year_dict,
                patch_size=self.patch_size,
                max_patches_per_tile_year=self.max_pixels,
                require_both_classes=True)
            self.val_dataset = DeforestationPatchDataset(
                val_tiles, self.target_years, aef_dict, def_year_dict, ignore_mask_dict,
                target_by_year_dict=target_by_year_dict,
                ignore_by_year_dict=ignore_by_year_dict,
                patch_size=self.patch_size,
                max_patches_per_tile_year=self.max_pixels,
                require_both_classes=False)
            print(f"[DataModule] Train patches={len(self.train_dataset)}")
            print(f"[DataModule] Val   patches={len(self.val_dataset)}")

    def _build_patch_arrays(self, tiles: list[str]) -> tuple[dict, dict, dict, dict, dict]:
        from tqdm import tqdm
        aef_dict = {}
        def_year_dict = {}
        ignore_mask_dict = {}
        target_by_year_dict = {}
        ignore_by_year_dict = {}
        
        for tile in tqdm(tiles, desc="Loading Spatial Arrays into RAM"):
            try:
                dst_transform, dst_crs, dst_shape = _get_dst_grid(tile, self.aef_dir)
            except FileNotFoundError as e:
                print(f"  [WARN] {e}"); continue
                
            def_year, ignore_mask, targets, ignores = _build_consensus_targets(
                tile, dst_transform, dst_crs, dst_shape,
                self.labels_dir, cache_dir=self.cache_dir,
                label_thresholds=self.label_thresholds,
                filter_mode=self.filter_mode,
                target_years=self.target_years,
            )
            def_year_dict[tile] = def_year
            ignore_mask_dict[tile] = ignore_mask
            for yr, target in targets.items():
                target_by_year_dict[(tile, yr)] = target
            for yr, ignore in ignores.items():
                ignore_by_year_dict[(tile, yr)] = ignore
            
            aef_2020_flat = _load_aef_flat(tile, 2020, dst_transform, dst_crs, dst_shape,
                                      self.aef_dir, cache_dir=self.cache_dir)
            if aef_2020_flat is not None:
                if self.selected_bands is not None:
                    aef_2020_flat = aef_2020_flat[:, self.selected_bands]
                aef_dict[(tile, 2020)] = aef_2020_flat.T.reshape(aef_2020_flat.shape[1], dst_shape[0], dst_shape[1])
                
            for yr in self.target_years:
                aef_yr_flat = _load_aef_flat(tile, yr, dst_transform, dst_crs, dst_shape,
                                        self.aef_dir, cache_dir=self.cache_dir)
                if aef_yr_flat is not None:
                    if self.selected_bands is not None:
                        aef_yr_flat = aef_yr_flat[:, self.selected_bands]
                    aef_dict[(tile, yr)] = aef_yr_flat.T.reshape(aef_yr_flat.shape[1], dst_shape[0], dst_shape[1])
                    
        return aef_dict, def_year_dict, ignore_mask_dict, target_by_year_dict, ignore_by_year_dict

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )
