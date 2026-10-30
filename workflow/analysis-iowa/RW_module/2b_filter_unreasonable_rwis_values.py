#!/usr/bin/env python3
"""Quality-control the hourly RWIS modeling table produced by script 2.

Values are set to missing only when they violate broad physical limits or are
simultaneously inconsistent with nearby stations and the same station's
surrounding hours. The script never removes a complete station-hour row.

Run:
    python3 2_build_rwis_station_hourly_modeling_table.py
    python3 2b_filter_unreasonable_rwis_values.py

Scripts 3 and 4 read this script's cleaned output directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_FILE = (
    ROOT
    / "2a_modeling_outputs"
    / "2_rwis_station_hourly_modeling_table_2016-01-01_to_2018-12-31.csv"
)
OUTPUT_DIR = ROOT / "2b_quality_controlled_outputs"
CLEANED_FILE = (
    OUTPUT_DIR
    / "2b_rwis_station_hourly_modeling_table_qc_2016-01-01_to_2018-12-31.csv"
)

TIME_COLUMN = "hour_utc"
STATION_COLUMN = "rwis_id"
NEIGHBOR_COUNT = 8
MAX_NEIGHBOR_DISTANCE_KM = 250.0
MIN_VALID_NEIGHBORS = 4
TEMPORAL_RADIUS_HOURS = 3
ROBUST_Z_THRESHOLD = 8.0

# Limits are deliberately broad. Spatial-temporal checks identify values that
# remain physically possible but are inconsistent with the network and the
# station's own adjacent observations.
VARIABLE_RULES = {
    "air_temp_c_mean": {
        "minimum": -50.0,
        "maximum": 50.0,
        "minimum_absolute_deviation": 12.0,
        "mad_floor": 0.5,
    },
    "dewpoint_c_mean": {
        "minimum": -60.0,
        "maximum": 40.0,
        "minimum_absolute_deviation": 15.0,
        "mad_floor": 0.5,
    },
    "humidity_mean": {
        "minimum": 0.0,
        "maximum": 100.0,
        "minimum_absolute_deviation": 35.0,
        "mad_floor": 2.0,
    },
    "wind_speed_mean": {
        "minimum": 0.0,
        "maximum": 60.0,
        "minimum_absolute_deviation": 20.0,
        "mad_floor": 1.0,
    },
    "tfs0_c_mean": {
        "minimum": -50.0,
        "maximum": 60.0,
        "minimum_absolute_deviation": 12.0,
        "mad_floor": 0.5,
    },
    "tfs1_c_mean": {
        "minimum": -50.0,
        "maximum": 60.0,
        "minimum_absolute_deviation": 12.0,
        "mad_floor": 0.5,
    },
    "tfs2_c_mean": {
        "minimum": -50.0,
        "maximum": 60.0,
        "minimum_absolute_deviation": 12.0,
        "mad_floor": 0.5,
    },
    "tfs3_c_mean": {
        "minimum": -50.0,
        "maximum": 60.0,
        "minimum_absolute_deviation": 12.0,
        "mad_floor": 0.5,
    },
}


def celsius_to_fahrenheit(values):
    return values * 9.0 / 5.0 + 32.0


def haversine_matrix(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat = np.radians(latitude.astype(float))
    lon = np.radians(longitude.astype(float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2.0) ** 2 + (
        np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    )
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def station_neighbors(station_metadata: pd.DataFrame) -> dict[str, list[str]]:
    station_ids = station_metadata.index.astype(str).to_numpy()
    distances = haversine_matrix(
        station_metadata["latitude"].to_numpy(),
        station_metadata["longitude"].to_numpy(),
    )
    result = {}
    for index, station_id in enumerate(station_ids):
        order = np.argsort(distances[index])
        valid = [
            int(j)
            for j in order
            if j != index and distances[index, j] <= MAX_NEIGHBOR_DISTANCE_KM
        ][:NEIGHBOR_COUNT]
        result[station_id] = station_ids[valid].tolist()
    return result


def robust_reference(frame: pd.DataFrame, rule: dict, axis: int):
    median = frame.median(axis=axis, skipna=True)
    if axis == 1:
        absolute = frame.sub(median, axis=0).abs()
    else:
        absolute = frame.sub(median, axis=1).abs()
    mad = absolute.median(axis=axis, skipna=True).clip(lower=rule["mad_floor"])
    return median, mad


def temporal_reference(pivot: pd.DataFrame, rule: dict):
    shifted = []
    timestamp_series = pd.Series(pivot.index, index=pivot.index)
    for offset in range(-TEMPORAL_RADIUS_HOURS, TEMPORAL_RADIUS_HOURS + 1):
        if offset == 0:
            continue
        frame = pivot.shift(offset)
        source_time = timestamp_series.shift(offset)
        consecutive = (timestamp_series - source_time).abs().eq(
            pd.Timedelta(hours=abs(offset))
        )
        shifted.append(frame.where(consecutive, np.nan, axis=0))
    values = np.stack([frame.to_numpy(dtype=float) for frame in shifted], axis=2)
    reference = np.nanmedian(values, axis=2)
    absolute = np.abs(values - reference[:, :, None])
    mad = np.nanmedian(absolute, axis=2)
    mad = np.maximum(mad, rule["mad_floor"])
    valid_count = np.isfinite(values).sum(axis=2)
    return reference, mad, valid_count


def quality_control_variable(
    data: pd.DataFrame,
    variable: str,
    rule: dict,
    neighbors: dict[str, list[str]],
) -> tuple[pd.Series, pd.DataFrame]:
    pivot = data.pivot_table(
        index=TIME_COLUMN,
        columns=STATION_COLUMN,
        values=variable,
        aggfunc="first",
    ).sort_index()
    original = pivot.copy()

    physical = original.lt(rule["minimum"]) | original.gt(rule["maximum"])
    working = original.mask(physical)

    spatial_reference = pd.DataFrame(index=working.index, columns=working.columns, dtype=float)
    spatial_mad = pd.DataFrame(index=working.index, columns=working.columns, dtype=float)
    spatial_count = pd.DataFrame(index=working.index, columns=working.columns, dtype=float)
    for station in working.columns:
        available_neighbors = [s for s in neighbors.get(str(station), []) if s in working.columns]
        neighbor_values = working[available_neighbors]
        median, mad = robust_reference(neighbor_values, rule, axis=1)
        spatial_reference[station] = median
        spatial_mad[station] = mad
        spatial_count[station] = neighbor_values.notna().sum(axis=1)

    temporal_ref, temporal_mad, temporal_count = temporal_reference(working, rule)
    temporal_reference_frame = pd.DataFrame(
        temporal_ref, index=working.index, columns=working.columns
    )
    temporal_mad_frame = pd.DataFrame(
        temporal_mad, index=working.index, columns=working.columns
    )
    temporal_count_frame = pd.DataFrame(
        temporal_count, index=working.index, columns=working.columns
    )

    spatial_deviation = (working - spatial_reference).abs()
    spatial_z = 0.6745 * spatial_deviation / spatial_mad
    spatial_outlier = (
        spatial_count.ge(MIN_VALID_NEIGHBORS)
        & spatial_deviation.gt(rule["minimum_absolute_deviation"])
        & spatial_z.gt(ROBUST_Z_THRESHOLD)
    )

    temporal_deviation = (working - temporal_reference_frame).abs()
    temporal_z = 0.6745 * temporal_deviation / temporal_mad_frame
    temporal_outlier = (
        temporal_count_frame.ge(MIN_VALID_NEIGHBORS)
        & temporal_deviation.gt(rule["minimum_absolute_deviation"])
        & temporal_z.gt(ROBUST_Z_THRESHOLD)
    )

    combined_outlier = working.notna() & spatial_outlier & temporal_outlier
    rejected = physical | combined_outlier

    long_original = original.stack(dropna=False).rename("original_value").reset_index()
    long_rejected = rejected.stack(dropna=False).rename("rejected").reset_index()
    long_physical = physical.stack(dropna=False).rename("physical_limit_flag").reset_index()
    long_spatial = spatial_outlier.stack(dropna=False).rename("spatial_outlier_flag").reset_index()
    long_temporal = temporal_outlier.stack(dropna=False).rename("temporal_outlier_flag").reset_index()
    long_spatial_ref = spatial_reference.stack(dropna=False).rename("spatial_reference").reset_index()
    long_temporal_ref = temporal_reference_frame.stack(dropna=False).rename("temporal_reference").reset_index()
    audit = long_original
    for frame in [long_rejected, long_physical, long_spatial, long_temporal, long_spatial_ref, long_temporal_ref]:
        audit = audit.merge(frame, on=[TIME_COLUMN, STATION_COLUMN], how="left")
    audit = audit[audit["rejected"] & audit["original_value"].notna()].copy()
    audit.insert(2, "variable", variable)
    audit["rejection_reason"] = np.where(
        audit["physical_limit_flag"],
        "physical_limit",
        "spatial_and_temporal_outlier",
    )

    cleaned_long = working.mask(combined_outlier).stack(dropna=False).rename("cleaned_value")
    keyed = data.set_index([TIME_COLUMN, STATION_COLUMN]).index
    cleaned_values = cleaned_long.reindex(keyed).to_numpy()
    return pd.Series(cleaned_values, index=data.index), audit


def clear_temperature_family(data: pd.DataFrame, mean_column: str) -> None:
    rejected = data[mean_column].isna() & data[f"__original_{mean_column}"].notna()
    if mean_column == "air_temp_c_mean":
        columns = [
            "air_temp_c_mean", "air_temp_c_min", "air_temp_c_max",
            "air_temp_f_mean", "air_temp_f_min", "air_temp_f_max",
        ]
    elif mean_column == "dewpoint_c_mean":
        columns = [
            "dewpoint_c_mean", "dewpoint_c_min", "dewpoint_c_max",
            "dewpoint_f_mean", "dewpoint_f_min", "dewpoint_f_max",
        ]
    else:
        return
    data.loc[rejected, [column for column in columns if column in data]] = np.nan


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_FILE}\nRun script 2 first."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Reading:", INPUT_FILE, flush=True)
    data = pd.read_csv(INPUT_FILE)
    data[TIME_COLUMN] = pd.to_datetime(data[TIME_COLUMN], errors="coerce")
    if "hour_local" not in data.columns:
        data.insert(
            1,
            "hour_local",
            pd.to_datetime(data[TIME_COLUMN], utc=True)
            .dt.tz_convert("America/Chicago")
            .dt.tz_localize(None),
        )
    data[STATION_COLUMN] = data[STATION_COLUMN].astype(str)
    data = data.sort_values([TIME_COLUMN, STATION_COLUMN]).reset_index(drop=True)

    missing_rule_columns = sorted(set(VARIABLE_RULES).difference(data.columns))
    if missing_rule_columns:
        raise ValueError(f"Missing QC variables: {missing_rule_columns}")

    station_metadata = (
        data.groupby(STATION_COLUMN)
        .agg(latitude=("latitude", "median"), longitude=("longitude", "median"))
    )
    neighbors = station_neighbors(station_metadata)

    audits = []
    for variable, rule in VARIABLE_RULES.items():
        print(f"Filtering {variable}...", flush=True)
        data[f"__original_{variable}"] = data[variable]
        cleaned, audit = quality_control_variable(data, variable, rule, neighbors)
        data[variable] = cleaned
        audits.append(audit)

    # Dew point should not materially exceed simultaneously observed air
    # temperature. A 2 C tolerance allows normal sensor disagreement.
    dewpoint_inconsistent = (
        data["dewpoint_c_mean"].notna()
        & data["air_temp_c_mean"].notna()
        & data["dewpoint_c_mean"].gt(data["air_temp_c_mean"] + 2.0)
    )
    if dewpoint_inconsistent.any():
        extra = data.loc[
            dewpoint_inconsistent,
            [TIME_COLUMN, STATION_COLUMN, "dewpoint_c_mean", "air_temp_c_mean"],
        ].copy()
        extra = extra.rename(
            columns={
                "dewpoint_c_mean": "original_value",
                "air_temp_c_mean": "spatial_reference",
            }
        )
        extra["variable"] = "dewpoint_c_mean"
        extra["rejected"] = True
        extra["physical_limit_flag"] = False
        extra["spatial_outlier_flag"] = False
        extra["temporal_outlier_flag"] = False
        extra["temporal_reference"] = np.nan
        extra["rejection_reason"] = "dewpoint_above_air_temperature"
        audits.append(extra)
        data.loc[dewpoint_inconsistent, "dewpoint_c_mean"] = np.nan

    clear_temperature_family(data, "air_temp_c_mean")
    clear_temperature_family(data, "dewpoint_c_mean")

    # Reconstruct F values for retained hourly means.
    data["air_temp_f_mean"] = celsius_to_fahrenheit(data["air_temp_c_mean"])
    data["dewpoint_f_mean"] = celsius_to_fahrenheit(data["dewpoint_c_mean"])
    for sensor in range(4):
        c_column = f"tfs{sensor}_c_mean"
        f_column = f"tfs{sensor}_f_mean"
        data[f_column] = celsius_to_fahrenheit(data[c_column])

    tfs_c = [f"tfs{sensor}_c_mean" for sensor in range(4)]
    tfs_f = [f"tfs{sensor}_f_mean" for sensor in range(4)]
    data["road_surface_sensor_count"] = data[tfs_c].notna().sum(axis=1)
    data["road_surface_temp_c_mean"] = data[tfs_c].mean(axis=1, skipna=True)
    data["road_surface_temp_f_mean"] = data[tfs_f].mean(axis=1, skipna=True)
    no_surface_sensor = data["road_surface_sensor_count"].eq(0)
    data.loc[no_surface_sensor, ["road_surface_temp_c_mean", "road_surface_temp_f_mean"]] = np.nan

    original_columns = [column for column in data if column.startswith("__original_")]
    data["qc_rejected_value_count"] = sum(
        (
            data[f"__original_{variable}"].notna()
            & data[variable].isna()
        ).astype(int)
        for variable in VARIABLE_RULES
    )
    data["qc_any_rejection"] = data["qc_rejected_value_count"].gt(0)
    data = data.drop(columns=original_columns)

    audit = pd.concat(audits, ignore_index=True, sort=False)
    if not audit.empty:
        local_lookup = data[[TIME_COLUMN, STATION_COLUMN, "hour_local"]].drop_duplicates()
        audit = audit.merge(
            local_lookup, on=[TIME_COLUMN, STATION_COLUMN], how="left"
        )
        ordered = ["hour_local", TIME_COLUMN, STATION_COLUMN, "variable"]
        audit = audit[[*ordered, *[c for c in audit.columns if c not in ordered]]]
    audit = audit.sort_values([TIME_COLUMN, STATION_COLUMN, "variable"])
    data.to_csv(CLEANED_FILE, index=False)
    audit.to_csv(OUTPUT_DIR / "2b_rejected_values_audit.csv", index=False)

    counts = audit.groupby(["variable", "rejection_reason"]).size().rename("count").reset_index()
    counts.to_csv(OUTPUT_DIR / "2b_rejection_summary.csv", index=False)
    summary = {
        "input_file": str(INPUT_FILE),
        "cleaned_file": str(CLEANED_FILE),
        "station_hour_rows": int(len(data)),
        "stations": int(data[STATION_COLUMN].nunique()),
        "hourly_events": int(data[TIME_COLUMN].nunique()),
        "rejected_value_count": int(len(audit)),
        "station_hour_rows_with_any_rejection": int(data["qc_any_rejection"].sum()),
        "rules": VARIABLE_RULES,
        "spatial_neighbors": NEIGHBOR_COUNT,
        "maximum_neighbor_distance_km": MAX_NEIGHBOR_DISTANCE_KM,
        "temporal_radius_hours": TEMPORAL_RADIUS_HOURS,
        "robust_z_threshold": ROBUST_Z_THRESHOLD,
    }
    (OUTPUT_DIR / "2b_quality_control_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nRejection summary:")
    print(counts.to_string(index=False))
    print("\nSaved cleaned table:", CLEANED_FILE)


if __name__ == "__main__":
    main()
