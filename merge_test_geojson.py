#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_geojsons(input_dir: Path, output_path: Path, time_step: int | None = None) -> None:
    all_features = []

    files = sorted(input_dir.glob("pred_*.geojson"))
    if not files:
        raise RuntimeError(f"No pred_*.geojson files found in {input_dir}")

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            gj = json.load(f)

        if gj.get("type") != "FeatureCollection":
            raise ValueError(f"{path} is not a FeatureCollection")

        for feat in gj.get("features", []):
            geom = feat.get("geometry")
            if geom is None:
                continue

            gtype = geom.get("type")
            if gtype not in {"Polygon", "MultiPolygon"}:
                continue

            props = {}
            if time_step is not None:
                props["time_step"] = time_step

            all_features.append({
                "type": "Feature",
                "properties": props,
                "geometry": geom,
            })

    merged = {
        "type": "FeatureCollection",
        "features": all_features,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f)

    print(f"Saved merged GeoJSON to: {output_path}")
    print(f"Merged {len(files)} files")
    print(f"Total features: {len(all_features)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-tile prediction GeoJSONs into one submission file")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing pred_*.geojson files")
    parser.add_argument("--output-path", type=Path, required=True, help="Output merged GeoJSON path")
    parser.add_argument("--time-step", type=int, default=None, help="Optional YYMM time_step to attach to every feature")
    args = parser.parse_args()

    merge_geojsons(args.input_dir, args.output_path, args.time_step)

    print("merge script started", flush=True)


if __name__ == "__main__":
    main()