#!/usr/bin/env python3
"""Prepare all-five-variable clustering inputs in [sample, hour, station, variable] order.

By default, reuse the original Step 4 imputed archive: its values and sample
eligibility already contain all five variables. Use --rebuild to repeat the
original preparation from QC observations, without exporting a duplicate large
hourly CSV. Neither mode modifies the original Step 4 outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE_SCRIPT = ROOT / "4_prepare_imputed_weather_regime_samples.py"
SOURCE_DIR = ROOT / "4_imputed_weather_regime_samples"
OUTPUT_DIR = ROOT / "4_v2_imputed_weather_regime_samples"
VARIABLES = ["road_surface_temp_c_mean", "air_temp_c_mean", "dewpoint_c_mean",
             "humidity_mean", "wind_speed_mean"]
AXES = ["sample", "hour_offset", "station", "variable"]
ARCHIVE_NAME = "4_v2_multivariable_clustering_samples.npz"
INDEX_NAME = "4_v2_clustering_sample_index.csv"


def load_base():
    spec = importlib.util.spec_from_file_location("original_step4", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_existing(source_dir):
    archive = source_dir / "4_clustering_samples_7h_60stations.npz"
    index = source_dir / "4_clustering_sample_index.csv"
    if not archive.exists() or not index.exists():
        raise FileNotFoundError("Original Step 4 archive/index not found. Run Step 4 first or use --rebuild.")
    with np.load(archive, allow_pickle=False) as handle:
        payload = {name: handle[name] for name in ["X", "imputation_mask", "sample_hour_utc",
                   "station_ids", "variables", "hour_offsets"]}
    # The legacy local-time member has object dtype. Reconstruct ISO strings
    # from UTC instead of requiring pickle to load a timestamp column.
    local = pd.DatetimeIndex(pd.to_datetime(payload["sample_hour_utc"], utc=True)).tz_convert("America/Chicago")
    payload["sample_hour_iowa_local"] = np.asarray(local.strftime("%Y-%m-%d %H:%M:%S%z").tolist(), dtype=str)
    sample_index = pd.read_csv(index)
    provenance = {"mode": "reuse_original_imputed_archive", "input_archive": str(archive.resolve()),
                  "input_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                  "input_index": str(index.resolve())}
    return payload, sample_index, provenance


def rebuild():
    base = load_base()
    if base.VARIABLES != VARIABLES or base.HISTORY_HOURS != 6:
        raise ValueError("Original preparation settings no longer match the documented five-variable seven-hour design.")
    data, station_file = base.load_inputs()
    ids = sorted(station_file[base.STATION_COLUMN].astype(str).unique())
    coordinates = data.groupby(base.STATION_COLUMN).agg(latitude=("latitude", "median"),
                                                       longitude=("longitude", "median")).reindex(ids)
    if not np.isfinite(coordinates.to_numpy()).all():
        raise ValueError("Station coordinates must be finite for spatial imputation.")
    print("Building arrays and checking pre-imputation hourly coverage...", flush=True)
    hours, observed = base.build_arrays(data, ids)
    _, _, endpoints = base.eligible_hours_and_samples(hours, observed)
    if not len(endpoints):
        raise ValueError("No eligible seven-hour windows.")
    print("Estimating development climatology and imputing with the original Step 4 rules...", flush=True)
    climatology = base.training_climatology(hours, observed)
    distance = base.haversine_matrix(coordinates.latitude.to_numpy(), coordinates.longitude.to_numpy())
    completed, sources = base.impute_values(hours, observed, climatology, distance)
    offsets = np.arange(-6, 1, dtype=np.int8)
    selected_times = endpoints[:, None] + offsets[None, :]
    selected_hours = hours[endpoints]
    local = selected_hours.tz_localize("UTC").tz_convert(base.IOWA_TIMEZONE)
    payload = {"X": completed[selected_times].astype(np.float32),
               "imputation_mask": sources[selected_times] != 0,
               "imputation_source": sources[selected_times],
               "sample_hour_utc": selected_hours.to_numpy(dtype="datetime64[ns]"),
               "sample_hour_iowa_local": local.strftime("%Y-%m-%d %H:%M:%S%z").to_numpy(dtype=str),
               "station_ids": np.asarray(ids), "variables": np.asarray(VARIABLES), "hour_offsets": offsets}
    sample_index = pd.DataFrame({"sample_id": np.arange(len(endpoints)),
        "sample_hour_utc": selected_hours, "sample_hour_iowa_local": payload["sample_hour_iowa_local"],
        "sample_year": local.year, "split": np.where(local.year == base.TEST_YEAR, "test", "development")})
    return payload, sample_index, {"mode": "rebuild_using_original_imputation_functions",
        "input_modeling_table": str(base.MODELING_TABLE), "input_station_list": str(base.STATION_LIST),
        "original_preparation_script_sha256": hashlib.sha256(BASE_SCRIPT.read_bytes()).hexdigest(),
        "imputation_source_codes": base.SOURCE_NAMES,
        "note": "Same eligibility/imputation rules as Step 4; no duplicate station-hour CSV exported."}


def validate(payload, index):
    x = payload["X"]
    mask = payload["imputation_mask"]
    if x.ndim != 4 or x.shape[1:] != (7, 60, 5) or not len(x):
        raise ValueError(f"Expected nonempty [sample,7,60,5], got {x.shape}.")
    if mask.shape != x.shape or mask.dtype != np.bool_ or not np.isfinite(x).all():
        raise ValueError("Missing/nonfinite values or invalid imputation mask.")
    if payload["variables"].astype(str).tolist() != VARIABLES:
        raise ValueError("Unexpected variable order.")
    if not np.array_equal(payload["hour_offsets"], np.arange(-6, 1)):
        raise ValueError("Expected hour offsets t-6 through t.")
    if len(payload["station_ids"]) != 60 or len(set(payload["station_ids"].astype(str))) != 60:
        raise ValueError("Expected 60 unique station IDs.")
    if len(index) != len(x) or index.sample_id.duplicated().any():
        raise ValueError("Sample-index length or ID uniqueness mismatch.")
    if any(len(payload[name]) != len(x) for name in ["sample_hour_utc", "sample_hour_iowa_local"]):
        raise ValueError("Timestamp length mismatch.")
    times = pd.DatetimeIndex(pd.to_datetime(payload["sample_hour_utc"], utc=True))
    index_times = pd.DatetimeIndex(pd.to_datetime(index.sample_hour_utc, utc=True))
    if not times.equals(index_times) or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Index/array timestamps do not match or are not unique and chronological.")
    local = times.tz_convert("America/Chicago")
    if not np.isin(local.year, [2016, 2017, 2018]).all():
        raise ValueError("Unexpected sample years.")
    expected_split = np.where(local.year == 2018, "test", "development")
    if not np.array_equal(index["split"].to_numpy(), expected_split):
        raise ValueError("Index split does not match local issue-time year.")
    if "imputation_source" in payload:
        sources = payload["imputation_source"]
        if sources.shape != x.shape or not np.array_equal(sources != 0, mask):
            raise ValueError("Imputation source codes and mask disagree.")


def export(payload, index, output, provenance, overwrite=False):
    validate(payload, index)
    if output.resolve() == SOURCE_DIR.resolve():
        raise ValueError("Choose a separate output directory, not the original Step 4 directory.")
    if (output / ARCHIVE_NAME).exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output / ARCHIVE_NAME}. Use --overwrite to replace V2 outputs.")
    output.mkdir(parents=True, exist_ok=True)
    x, mask = payload["X"], payload["imputation_mask"]
    enriched = dict(payload, axis_names=np.asarray(AXES),
                    schema_version=np.asarray(2, dtype=np.int16),
                    feature_representation=np.asarray("all five variables in original units; C-order flattening"))
    with tempfile.NamedTemporaryFile(dir=output, suffix=".npz", delete=False) as temporary:
        temp_path = Path(temporary.name)
        try:
            np.savez_compressed(temporary, **enriched)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    temp_path.replace(output / ARCHIVE_NAME)
    index = index.copy()
    index["imputed_cell_count"] = mask.sum(axis=(1, 2, 3))
    index["total_cell_count"] = np.prod(x.shape[1:])
    index["imputed_cell_pct"] = 100*index.imputed_cell_count/index.total_cell_count
    for j, variable in enumerate(VARIABLES):
        index[f"{variable}_imputed_cell_pct"] = 100*mask[:, :, :, j].mean(axis=(1, 2))
    index.to_csv(output / INDEX_NAME, index=False)
    feature_rows = [{"flattened_feature_index": k, "hour_offset": int(payload["hour_offsets"][h]),
                     "rwis_id": str(payload["station_ids"][s]), "variable": VARIABLES[v],
                     "hour_axis_index": h, "station_axis_index": s, "variable_axis_index": v}
                    for k, (h, s, v) in enumerate(np.ndindex(x.shape[1:]))]
    pd.DataFrame(feature_rows).to_csv(output / "4_v2_clustering_feature_order.csv", index=False)
    rows = []
    for split in ["development", "test"]:
        selected = index.split.eq(split).to_numpy()
        if not selected.any():
            continue
        for j, variable in enumerate(VARIABLES):
            values = x[selected, :, :, j]
            rows.append({"split": split, "variable": variable, "sample_count": int(selected.sum()),
                         "cell_count": values.size, "mean": float(values.mean(dtype=np.float64)),
                         "standard_deviation": float(values.std(dtype=np.float64)),
                         "minimum": float(values.min()), "maximum": float(values.max()),
                         "imputed_cell_pct": float(100*mask[selected, :, :, j].mean())})
    pd.DataFrame(rows).to_csv(output / "4_v2_variable_summary.csv", index=False)
    summary = {"sample_tensor_shape": list(x.shape), "axis_names": AXES,
        "single_sample_shape": list(x.shape[1:]), "variables": VARIABLES, "feature_count": 2100,
        "hour_offsets": payload["hour_offsets"].tolist(), "station_count": 60,
        "development_sample_count": int(index.split.eq("development").sum()),
        "test_sample_count": int(index.split.eq("test").sum()),
        "mean_sample_imputed_cell_pct": float(index.imputed_cell_pct.mean()),
        "normalization_applied": False, "pca_applied": False,
        "flattening": "X.reshape(len(X), -1), C order: hour, station, variable",
        "overlapping_windows_note": "Variable summaries count window cells, including repeated station-hours across overlapping samples.",
        "downstream_requirement": "Step 5 currently extracts RST only. A multivariable clustering implementation must retain all five variables and fit scaling/PCA on development data only.",
        "provenance": provenance}
    (output / "4_v2_preparation_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-directory", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--rebuild", action="store_true", help="Repeat original preparation from raw QC observations.")
    parser.add_argument("--overwrite", action="store_true", help="Replace V2 outputs only.")
    args = parser.parse_args()
    payload, index, provenance = rebuild() if args.rebuild else load_existing(args.source_directory)
    export(payload, index, args.output_directory, provenance, args.overwrite)
    print("Saved all-five-variable clustering inputs to:", args.output_directory)


if __name__ == "__main__":
    main()
