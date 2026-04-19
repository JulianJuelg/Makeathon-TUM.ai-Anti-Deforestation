"""Challenge metrics and visualization helpers.

Metric definitions, aggregated over validation tiles:

  Union IoU     = area(pred_polygon_union ∩ gt_polygon_union)
                  / area(pred_polygon_union ∪ gt_polygon_union)
  Recall        = area(pred_polygon_union ∩ gt_polygon_union) / area(gt_polygon_union)
  FPR (FDR)     = area(pred - gt) / area(pred)
  Year Accuracy = |correctly_dated| / |pred ∪ temporal_gt|

The primary IoU metric uses vector polygon union area, matching the benchmark
description. Year accuracy remains pixel-based because it evaluates the first
predicted deforestation year.
"""
from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
import numpy as np
import torch
import geopandas as gpd
import rasterio
from rasterio.warp import transform_geom

from deforestation.datamodule import (
    _build_def_year_and_ignore,
    _get_dst_grid,
    _load_aef_flat,
)

TEST_TILES = ["18NVJ_1_6", "18NYH_2_1", "33NTE_5_1", "47QMA_6_2", "48PWA_0_6"]


def _normalise_thresholds(threshold: float | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(threshold, (list, tuple)):
        return [float(t) for t in threshold]
    return [float(threshold)]


def _empty_feature_collection() -> dict:
    return {"type": "FeatureCollection", "features": []}


def _normalise_nan(band: np.ndarray, p_lo: int = 2, p_hi: int = 98) -> np.ndarray:
    valid_pixels = band[np.isfinite(band)]
    if len(valid_pixels) == 0:
        return np.zeros_like(band)
    lo, hi = np.percentile(valid_pixels, [p_lo, p_hi])
    return np.nan_to_num(np.clip((band - lo) / (hi - lo + 1e-6), 0, 1), nan=0.0)


def _polygonize_with_submission_utils(
    mask: np.ndarray,
    transform,
    crs,
    min_area_ha: float,
) -> dict:
    """Use the exact challenge submission utility for polygonization/filtering."""
    from submission_utils import raster_to_geojson

    with tempfile.TemporaryDirectory() as tmpdir:
        raster_path = Path(tmpdir) / "mask.tif"
        write_prediction_raster(mask.astype(np.uint8), transform, crs, raster_path)
        try:
            return raster_to_geojson(
                raster_path=raster_path,
                output_path=None,
                min_area_ha=min_area_ha,
            )
        except ValueError:
            return _empty_feature_collection()


def _plot_submission_polygons(
    ax,
    geojson: dict,
    transform,
    dst_crs,
    out_shape: tuple[int, int],
    *,
    facecolor: str = "#e31a1c",
    edgecolor: str = "#7f0000",
    background: str = "#d3d3d3",
) -> None:
    from matplotlib.patches import Polygon as MplPolygon
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape

    ax.set_facecolor(background)
    ax.set_xlim(0, out_shape[1])
    ax.set_ylim(out_shape[0], 0)
    ax.set_aspect("equal")

    def add_polygon(poly: Polygon) -> None:
        exterior = [~transform * xy[:2] for xy in poly.exterior.coords]
        ax.add_patch(
            MplPolygon(
                exterior,
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=0.8,
                alpha=0.85,
            )
        )
        for interior in poly.interiors:
            hole = [~transform * xy[:2] for xy in interior.coords]
            ax.add_patch(
                MplPolygon(
                    hole,
                    closed=True,
                    facecolor=background,
                    edgecolor=edgecolor,
                    linewidth=0.4,
                    alpha=1.0,
                )
            )

    for feature in geojson.get("features", []):
        geom_4326 = feature.get("geometry")
        if geom_4326 is None:
            continue
        geom_dst = shape(transform_geom("EPSG:4326", dst_crs, geom_4326))
        if isinstance(geom_dst, Polygon):
            add_polygon(geom_dst)
        elif isinstance(geom_dst, MultiPolygon):
            for poly in geom_dst.geoms:
                add_polygon(poly)
        elif isinstance(geom_dst, GeometryCollection):
            for geom in geom_dst.geoms:
                if isinstance(geom, Polygon):
                    add_polygon(geom)
                elif isinstance(geom, MultiPolygon):
                    for poly in geom.geoms:
                        add_polygon(poly)


def _union_geojson_geometry(geojson: dict, dst_crs):
    from shapely.geometry import GeometryCollection, shape
    from shapely.ops import unary_union

    geometries = []
    for feature in geojson.get("features", []):
        geom_4326 = feature.get("geometry")
        if geom_4326 is None:
            continue
        geom = shape(transform_geom("EPSG:4326", dst_crs, geom_4326))
        if not geom.is_valid:
            geom = geom.buffer(0)
        if not geom.is_empty:
            geometries.append(geom)
    if not geometries:
        return GeometryCollection()
    return unary_union(geometries)


def _update_totals_from_union_polygons(
    totals: dict[str, float],
    pred_geojson: dict,
    gt_geojson: dict,
    dst_crs,
) -> None:
    pred_union = _union_geojson_geometry(pred_geojson, dst_crs)
    gt_union = _union_geojson_geometry(gt_geojson, dst_crs)

    # Compute areas in a projected metric CRS (m²), not in geographic degrees.
    if pred_union.is_empty and gt_union.is_empty:
        pred_area = 0.0
        gt_area = 0.0
        intersection = 0.0
    else:
        geom_list = []
        labels = []
        if not pred_union.is_empty:
            geom_list.append(pred_union)
            labels.append("pred")
        if not gt_union.is_empty:
            geom_list.append(gt_union)
            labels.append("gt")

        gdf = gpd.GeoDataFrame({"kind": labels}, geometry=geom_list, crs=dst_crs)
        if gdf.crs is not None and gdf.crs.is_geographic:
            utm_crs = gdf.estimate_utm_crs()
            if utm_crs is not None:
                gdf = gdf.to_crs(utm_crs)

        pred_metric = gdf[gdf["kind"] == "pred"].geometry.iloc[0] if (gdf["kind"] == "pred").any() else None
        gt_metric = gdf[gdf["kind"] == "gt"].geometry.iloc[0] if (gdf["kind"] == "gt").any() else None

        pred_area = float(pred_metric.area) if pred_metric is not None else 0.0
        gt_area = float(gt_metric.area) if gt_metric is not None else 0.0
        if pred_metric is None or gt_metric is None or pred_metric.is_empty or gt_metric.is_empty:
            intersection = 0.0
        else:
            intersection = float(pred_metric.intersection(gt_metric).area)

    totals["intersection"] += intersection
    totals["pred_area"] += pred_area
    totals["gt_area"] += gt_area


@torch.no_grad()
def _predict_valid_probs(
    module,
    aef_2020: np.ndarray,
    aef_yr: np.ndarray,
    valid: np.ndarray,
    dst_shape: tuple[int, int],
    selected_bands: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return probabilities only for valid pixels plus their flat indices."""
    device = next(module.parameters()).device
    idx_valid = np.where(valid)[0]

    a2020_2d = aef_2020.T.reshape(64, dst_shape[0], dst_shape[1])
    ayr_2d = aef_yr.T.reshape(64, dst_shape[0], dst_shape[1])
    if selected_bands is not None:
        a2020_2d = a2020_2d[selected_bands]
        ayr_2d = ayr_2d[selected_bands]
    delta_2d = ayr_2d - a2020_2d
    x_2d = np.concatenate([a2020_2d, ayr_2d, delta_2d], axis=0).astype(np.float32)
    x_t = torch.from_numpy(np.nan_to_num(x_2d)).unsqueeze(0).to(device)

    logit = module(x_t)
    probs_full = torch.sigmoid(logit).squeeze().cpu().numpy().ravel()
    return probs_full[idx_valid], idx_valid


def _update_year_totals(
    totals: dict[str, int],
    pred_year_flat: np.ndarray,
    def_year: np.ndarray,
    both_valid: np.ndarray,
    target_years: list[int],
) -> None:
    def_year_flat = def_year.ravel()
    for yr in target_years:
        gt_yr = (def_year_flat == yr) & both_valid
        pred_yr = (pred_year_flat == yr) & both_valid
        totals["correctly_dated"] += int((gt_yr & pred_yr).sum())
        totals["year_denom"] += int((gt_yr | pred_yr).sum())


def _finalise_totals(totals: dict[str, float], prefix: str = "val") -> dict[str, float]:
    eps = 1e-8
    intersection = totals["intersection"]
    pred_area = totals["pred_area"]
    gt_area = totals["gt_area"]
    iou = intersection / (pred_area + gt_area - intersection + eps)
    recall = intersection / (gt_area + eps)
    fpr = (pred_area - intersection) / (pred_area + eps)
    year_acc = totals["correctly_dated"] / (totals["year_denom"] + eps)
    return {
        f"{prefix}/union_iou": float(iou),
        f"{prefix}/recall": float(recall),
        f"{prefix}/fpr": float(fpr),
        f"{prefix}/year_acc": float(year_acc),
        f"{prefix}/pred_area_m2": float(pred_area),
        f"{prefix}/gt_area_m2": float(gt_area),
        f"{prefix}/intersection_area_m2": float(intersection),
        f"{prefix}/pred_area_ha": float(pred_area / 10_000),
        f"{prefix}/gt_area_ha": float(gt_area / 10_000),
    }


def _new_totals() -> dict[str, float]:
    return {
        "intersection": 0.0,
        "pred_area": 0.0,
        "gt_area": 0.0,
        "correctly_dated": 0.0,
        "year_denom": 0.0,
    }


@torch.no_grad()
def compute_challenge_metrics(
    module,           # DeforestationModule (pl.LightningModule)
    val_tiles: list[str],
    aef_dir: str,
    labels_dir: str,
    cache_dir: str | None,
    target_years: list[int],
    threshold: float | list[float] = 0.5,
    selected_bands: list[int] | None = None,
    min_area_ha: float = 0.5,
    label_thresholds: dict | None = None,
    filter_mode: str = "strict",
) -> dict[str, float]:
    """Run full-tile validation inference and compute challenge metrics."""
    module.eval()

    thresholds = _normalise_thresholds(threshold)
    totals_by_threshold = {t: _new_totals() for t in thresholds}
    max_year = max(target_years)
    visual_payload = None

    for tile in val_tiles:
        try:
            dst_transform, dst_crs, dst_shape = _get_dst_grid(tile, aef_dir)
        except FileNotFoundError:
            continue

        def_year, ignore_mask = _build_def_year_and_ignore(
            tile, dst_transform, dst_crs, dst_shape,
            labels_dir, cache_dir=cache_dir,
            label_thresholds=label_thresholds,
            filter_mode=filter_mode,
        )
        gt_any_raw = (def_year < 9999) & (def_year <= max_year) & ~ignore_mask  # scored gt_union mask
        gt_geojson = _polygonize_with_submission_utils(
            gt_any_raw.astype(np.uint8), dst_transform, dst_crs, min_area_ha
        )

        aef_2020 = _load_aef_flat(tile, 2020, dst_transform, dst_crs, dst_shape,
                                   aef_dir, cache_dir=cache_dir)
        if aef_2020 is None:
            continue

        valid_base = np.all(np.isfinite(aef_2020), axis=1) & ~ignore_mask.ravel()

        pred_year_by_threshold = {
            t: np.full(dst_shape[0] * dst_shape[1], 9999, dtype=np.int32)
            for t in thresholds
        }

        for yr in sorted(target_years):
            aef_yr = _load_aef_flat(tile, yr, dst_transform, dst_crs, dst_shape,
                                     aef_dir, cache_dir=cache_dir)
            if aef_yr is None:
                continue

            valid = valid_base & np.all(np.isfinite(aef_yr), axis=1)
            if not valid.any():
                continue

            probs, idx_valid = _predict_valid_probs(
                module, aef_2020, aef_yr, valid, dst_shape, selected_bands=selected_bands
            )

            for t, pred_year_flat in pred_year_by_threshold.items():
                pred_positive = probs >= t
                not_predicted = pred_year_flat[idx_valid] == 9999
                newly_deforested = idx_valid[pred_positive & not_predicted]
                pred_year_flat[newly_deforested] = yr

        for t, pred_year_flat in pred_year_by_threshold.items():
            pred_any_raw = (pred_year_flat < 9999).reshape(dst_shape) & ~ignore_mask
            pred_geojson = _polygonize_with_submission_utils(
                pred_any_raw.astype(np.uint8), dst_transform, dst_crs, min_area_ha
            )
            _update_totals_from_union_polygons(
                totals_by_threshold[t],
                pred_geojson,
                gt_geojson,
                dst_crs,
            )
            _update_year_totals(
                totals_by_threshold[t],
                pred_year_flat,
                def_year,
                valid_base,
                target_years,
            )
            if visual_payload is None:
                visual_payload = {
                    "tile": tile,
                    "aef_2020": aef_2020,
                    "dst_shape": dst_shape,
                    "dst_transform": dst_transform,
                    "dst_crs": dst_crs,
                    "gt_geojson": gt_geojson,
                    "pred_geojson_by_threshold": {},
                }
            if visual_payload is not None and visual_payload["tile"] == tile:
                visual_payload["pred_geojson_by_threshold"][t] = pred_geojson

    metrics_by_threshold = {
        t: _finalise_totals(totals, prefix="val")
        for t, totals in totals_by_threshold.items()
    }
    best_threshold = max(thresholds, key=lambda t: metrics_by_threshold[t]["val/union_iou"])
    metrics = dict(metrics_by_threshold[best_threshold])
    metrics["val/best_threshold"] = float(best_threshold)

    for t, vals in metrics_by_threshold.items():
        suffix = f"{t:.2f}"
        metrics[f"val/union_iou@{suffix}"] = vals["val/union_iou"]
        metrics[f"val/recall@{suffix}"] = vals["val/recall"]
        metrics[f"val/fpr@{suffix}"] = vals["val/fpr"]

    if visual_payload is not None and module.logger is not None:
        import matplotlib.pyplot as plt

        aef_2020_2d = visual_payload["aef_2020"].T.reshape(64, *visual_payload["dst_shape"])
        aef_rgb = np.stack([_normalise_nan(aef_2020_2d[i]) for i in [7, 13, 52]], axis=-1)
        pred_geojson = visual_payload["pred_geojson_by_threshold"].get(best_threshold, _empty_feature_collection())
        gt_geojson = visual_payload["gt_geojson"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(aef_rgb)
        axes[0].set_title(f"Embedding (AEF 2020)\nTile {visual_payload['tile']}", fontsize=12)
        axes[0].axis("off")

        _plot_submission_polygons(
            axes[1],
            pred_geojson,
            visual_payload["dst_transform"],
            visual_payload["dst_crs"],
            visual_payload["dst_shape"],
        )
        axes[1].set_title(
            f"Prediction GeoJSON polygons\nthreshold={best_threshold:.2f}",
            fontsize=12,
        )
        axes[1].axis("off")

        _plot_submission_polygons(
            axes[2],
            gt_geojson,
            visual_payload["dst_transform"],
            visual_payload["dst_crs"],
            visual_payload["dst_shape"],
        )
        axes[2].set_title("Ground truth GeoJSON polygons", fontsize=12)
        axes[2].axis("off")
        plt.tight_layout()

        experiment = getattr(module.logger, "experiment", None)
        if experiment is not None:
            experiment.add_figure("val/spatial_visualization", fig, global_step=module.current_epoch)
        plt.close(fig)

    return metrics

@torch.no_grad()
def visualize_test_tiles(
    module,
    test_tiles: list[str],
    aef_dir: str,
    cache_dir: str | None,
    target_years: list[int],
    threshold: float = 0.5,
    selected_bands: list[int] | None = None,
    min_area_ha: float = 0.5,
):
    """Run full-tile test inference and log polygon visualizations."""
    if module.logger is None or getattr(module.logger, "experiment", None) is None:
        return

    module.eval()

    import matplotlib.pyplot as plt

    def normalise_nan(band, p_lo=2, p_hi=98):
        valid_pixels = band[np.isfinite(band)]
        if len(valid_pixels) == 0: return np.zeros_like(band)
        lo, hi = np.percentile(valid_pixels, [p_lo, p_hi])
        return np.nan_to_num(np.clip((band - lo) / (hi - lo + 1e-6), 0, 1), nan=0.0)

    for tile in test_tiles:
        try:
            dst_transform, dst_crs, dst_shape = _get_dst_grid(tile, aef_dir)
        except FileNotFoundError:
            continue

        aef_2020 = _load_aef_flat(tile, 2020, dst_transform, dst_crs, dst_shape,
                                   aef_dir, cache_dir=cache_dir)
        if aef_2020 is None:
            continue

        valid_base = np.all(np.isfinite(aef_2020), axis=1)
        pred_year_flat = np.full(dst_shape[0] * dst_shape[1], 9999, dtype=np.int32)

        for yr in sorted(target_years):
            aef_yr = _load_aef_flat(tile, yr, dst_transform, dst_crs, dst_shape,
                                     aef_dir, cache_dir=cache_dir)
            if aef_yr is None:
                continue

            valid = valid_base & np.all(np.isfinite(aef_yr), axis=1)
            if not valid.any():
                continue

            probs, idx_valid = _predict_valid_probs(
                module, aef_2020, aef_yr, valid, dst_shape, selected_bands=selected_bands
            )
            pred_positive = probs >= threshold
            newly_deforested = idx_valid[pred_positive & (pred_year_flat[idx_valid] == 9999)]
            pred_year_flat[newly_deforested] = yr

        pred_any = pred_year_flat < 9999
        pred_geojson = _polygonize_with_submission_utils(
            pred_any.reshape(dst_shape).astype(np.uint8),
            dst_transform,
            dst_crs,
            min_area_ha,
        )

        aef_2020_2d = aef_2020.T.reshape(64, dst_shape[0], dst_shape[1])
        aef_rgb = np.stack([normalise_nan(aef_2020_2d[i]) for i in [7, 13, 52]], axis=-1)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(aef_rgb)
        axes[0].set_title(f"Input (AEF 2020)\nTest Tile {tile}", fontsize=12)
        axes[0].axis("off")

        _plot_submission_polygons(
            axes[1],
            pred_geojson,
            dst_transform,
            dst_crs,
            dst_shape,
        )
        axes[1].set_title(f"Prediction GeoJSON polygons\nTest Tile {tile}", fontsize=12)
        axes[1].axis("off")
        
        plt.tight_layout()
        module.logger.experiment.add_figure(f"test_visualization/{tile}", fig, global_step=module.current_epoch)
        plt.close(fig)


@torch.no_grad()
def predict_tile_mask(
    module,
    tile: str,
    aef_dir: str,
    cache_dir: str | None,
    target_years: list[int],
    threshold: float,
    selected_bands: list[int] | None = None,
) -> tuple[np.ndarray, object, object]:
    """Predict a binary deforestation mask for one tile."""
    module.eval()
    dst_transform, dst_crs, dst_shape = _get_dst_grid(tile, aef_dir)
    aef_2020 = _load_aef_flat(tile, 2020, dst_transform, dst_crs, dst_shape,
                              aef_dir, cache_dir=cache_dir)
    if aef_2020 is None:
        return np.zeros(dst_shape, dtype=np.uint8), dst_transform, dst_crs

    valid_base = np.all(np.isfinite(aef_2020), axis=1)
    pred_year_flat = np.full(dst_shape[0] * dst_shape[1], 9999, dtype=np.int32)

    for yr in sorted(target_years):
        aef_yr = _load_aef_flat(tile, yr, dst_transform, dst_crs, dst_shape,
                                aef_dir, cache_dir=cache_dir)
        if aef_yr is None:
            continue
        valid = valid_base & np.all(np.isfinite(aef_yr), axis=1)
        if not valid.any():
            continue

        probs, idx_valid = _predict_valid_probs(
            module, aef_2020, aef_yr, valid, dst_shape, selected_bands=selected_bands
        )
        newly_deforested = idx_valid[(probs >= threshold) & (pred_year_flat[idx_valid] == 9999)]
        pred_year_flat[newly_deforested] = yr

    return (pred_year_flat < 9999).reshape(dst_shape).astype(np.uint8), dst_transform, dst_crs


def write_prediction_raster(
    mask: np.ndarray,
    transform,
    crs,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f"{output_path.stem}.tmp-{uuid.uuid4().hex}.tif"
    try:
        with rasterio.open(
            tmp_path,
            "w",
            driver="GTiff",
            height=mask.shape[0],
            width=mask.shape[1],
            count=1,
            dtype="uint8",
            crs=crs,
            transform=transform,
            compress="lzw",
            nodata=0,
        ) as dst:
            dst.write(mask.astype(np.uint8), 1)
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return output_path


@torch.no_grad()
def generate_combined_submission(
    module,
    test_tiles: list[str],
    aef_dir: str,
    cache_dir: str | None,
    target_years: list[int],
    threshold: float,
    selected_bands: list[int] | None,
    prediction_dir: str | Path,
    output_path: str | Path,
    min_area_ha: float = 0.5,
) -> dict:
    """Write per-tile binary rasters and one combined GeoJSON submission."""
    from submission_utils import raster_to_geojson

    prediction_dir = Path(prediction_dir)
    output_path = Path(output_path)
    features = []

    for tile in test_tiles:
        try:
            mask, transform, crs = predict_tile_mask(
                module=module,
                tile=tile,
                aef_dir=aef_dir,
                cache_dir=cache_dir,
                target_years=target_years,
                threshold=threshold,
                selected_bands=selected_bands,
            )
        except FileNotFoundError:
            continue

        raster_path = write_prediction_raster(
            mask,
            transform,
            crs,
            prediction_dir / f"{tile}_{max(target_years)}_binary.tif",
        )

        try:
            tile_geojson = raster_to_geojson(
                raster_path=raster_path,
                output_path=None,
                min_area_ha=min_area_ha,
            )
        except ValueError:
            tile_geojson = _empty_feature_collection()

        for feature in tile_geojson.get("features", []):
            feature.setdefault("properties", {})
            feature["properties"]["tile_id"] = tile
            features.append(feature)

    combined = {"type": "FeatureCollection", "features": features}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(combined, f)
    return combined
