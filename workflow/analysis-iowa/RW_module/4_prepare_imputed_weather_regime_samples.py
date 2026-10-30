#!/usr/bin/env python3
"""Prepare complete 60-station, seven-hour samples for weather clustering.

Eligibility is assessed before imputation. Each retained hour must have a
complete five-variable observation vector at no fewer than 80% of the fixed
60 RWIS stations. A clustering sample ending at t is retained only when all
hours t-6,...,t satisfy that rule.

Missing values are filled without using 2018 to estimate climatology:
  1. direct spatial IDW of same-hour observations;
  2. the station's most recent completed anomaly within six hours;
  3. the 2016-2017 station/month/hour climatology as a final fallback.

The compressed NPZ output is the preferred input for the clustering script.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
IOWA_TIMEZONE = "America/Chicago"
MODELING_TABLE = (
    ROOT
    / "2b_quality_controlled_outputs"
    / "2b_rwis_station_hourly_modeling_table_qc_2016-01-01_to_2018-12-31.csv"
)
STATION_LIST = (
    ROOT
    / "3_most_overlapped_rwis_outputs"
    / "3_rwis_stations_reporting_all_weather_variables_every_year.csv"
)
OUTPUT_DIR = ROOT / "4_imputed_weather_regime_samples"

EVENT_COLUMN = "hour_utc"
STATION_COLUMN = "rwis_id"
VARIABLES = [
    "road_surface_temp_c_mean",
    "air_temp_c_mean",
    "dewpoint_c_mean",
    "humidity_mean",
    "wind_speed_mean",
]
MIN_HOURLY_STATION_COVERAGE = 0.80
HISTORY_HOURS = 6
IDW_NEIGHBORS = 5
IDW_POWER = 2.0
IDW_MAX_DISTANCE_KM = 200.0
MIN_IDW_NEIGHBORS = 3
TRAIN_END_YEAR = 2017
TEST_YEAR = 2018

SOURCE_NAMES = {
    0: "observed",
    1: "spatial_idw",
    2: "temporal_previous",
    3: "climatology",
}
IMPUTATION_BOUNDS = {
    "road_surface_temp_c_mean": (-50.0, 60.0),
    "air_temp_c_mean": (-50.0, 50.0),
    "dewpoint_c_mean": (-60.0, 40.0),
    "humidity_mean": (0.0, 100.0),
    "wind_speed_mean": (0.0, 60.0),
}


def haversine_matrix(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat = np.radians(latitude.astype(float))
    lon = np.radians(longitude.astype(float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2.0) ** 2 + (
        np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    )
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not MODELING_TABLE.exists():
        raise FileNotFoundError(f"Modeling table not found: {MODELING_TABLE}")
    if not STATION_LIST.exists():
        raise FileNotFoundError(
            f"Station list not found: {STATION_LIST}\n"
            "Run python3 3_most_overlapped_rwis.py first."
        )

    stations = pd.read_csv(STATION_LIST)
    station_ids = stations[STATION_COLUMN].dropna().astype(str).drop_duplicates()
    if len(station_ids) != 60:
        raise ValueError(
            f"Expected 60 stations in {STATION_LIST.name}, found {len(station_ids)}."
        )

    usecols = [
        EVENT_COLUMN,
        STATION_COLUMN,
        "latitude",
        "longitude",
        "road_surface_sensor_count",
        *VARIABLES,
    ]
    data = pd.read_csv(MODELING_TABLE, usecols=usecols)
    data[EVENT_COLUMN] = pd.to_datetime(data[EVENT_COLUMN], errors="coerce")
    data[STATION_COLUMN] = data[STATION_COLUMN].astype(str)
    data = data[
        data[STATION_COLUMN].isin(station_ids) & data[EVENT_COLUMN].notna()
    ].copy()
    data = data.sort_values([EVENT_COLUMN, STATION_COLUMN])
    return data, stations


def build_arrays(data: pd.DataFrame, station_ids: list[str]):
    hours = pd.DatetimeIndex(sorted(data[EVENT_COLUMN].unique()))
    shape = (len(hours), len(station_ids), len(VARIABLES))
    values = np.full(shape, np.nan, dtype=np.float32)

    for variable_index, variable in enumerate(VARIABLES):
        pivot = data.pivot_table(
            index=EVENT_COLUMN,
            columns=STATION_COLUMN,
            values=variable,
            aggfunc="first",
        ).reindex(index=hours, columns=station_ids)
        values[:, :, variable_index] = pivot.to_numpy(dtype=np.float32)

    # RST is valid only when at least one pavement sensor supplied the mean.
    sensor_count = data.pivot_table(
        index=EVENT_COLUMN,
        columns=STATION_COLUMN,
        values="road_surface_sensor_count",
        aggfunc="first",
    ).reindex(index=hours, columns=station_ids).to_numpy(dtype=float)
    values[:, :, 0][~(sensor_count >= 1)] = np.nan
    return hours, values


def training_climatology(
    hours: pd.DatetimeIndex,
    observed: np.ndarray,
) -> np.ndarray:
    """Return [time, station, variable] climatological expected values."""
    train = hours.year <= TRAIN_END_YEAR
    n_time, n_station, n_variable = observed.shape
    climatology = np.full_like(observed, np.nan, dtype=np.float32)
    month_hour = pd.MultiIndex.from_arrays([hours.month, hours.hour])

    global_month_hour = []
    global_medians = []
    for variable_index in range(n_variable):
        global_frame = pd.DataFrame(
            observed[train, :, variable_index], index=hours[train]
        )
        stacked = global_frame.stack()
        global_month_hour.append(
            stacked.groupby(
                [
                    stacked.index.get_level_values(0).month,
                    stacked.index.get_level_values(0).hour,
                ]
            ).median()
        )
        global_medians.append(
            float(np.nanmedian(observed[train, :, variable_index]))
        )

    for station_index in range(n_station):
        for variable_index in range(n_variable):
            series = pd.Series(observed[:, station_index, variable_index], index=hours)
            train_series = series.loc[train]
            station_mh = train_series.groupby(
                [train_series.index.month, train_series.index.hour]
            ).median()
            station_median = float(train_series.median())

            expected = np.empty(n_time, dtype=np.float32)
            for time_index, key in enumerate(month_hour):
                value = station_mh.get(key, np.nan)
                if pd.isna(value):
                    value = station_median
                if pd.isna(value):
                    value = global_month_hour[variable_index].get(key, np.nan)
                if pd.isna(value):
                    value = global_medians[variable_index]
                expected[time_index] = value
            climatology[:, station_index, variable_index] = expected
    return climatology


def impute_values(
    hours: pd.DatetimeIndex,
    observed: np.ndarray,
    climatology: np.ndarray,
    distances_km: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    completed = observed.copy()
    source = np.where(np.isfinite(observed), 0, -1).astype(np.int8)
    n_time, n_station, n_variable = observed.shape
    last_anomaly = np.full((n_station, n_variable), np.nan, dtype=np.float32)
    last_time = np.full((n_station, n_variable), -10_000, dtype=np.int32)

    for time_index in range(n_time):
        if time_index and hours[time_index] - hours[time_index - 1] != pd.Timedelta(hours=1):
            last_anomaly[:] = np.nan
            last_time[:] = -10_000

        for variable_index in range(n_variable):
            current = observed[time_index, :, variable_index]
            expected = climatology[time_index, :, variable_index]
            observed_indices = np.flatnonzero(np.isfinite(current))
            for station_index in np.flatnonzero(~np.isfinite(current)):
                candidate_distances = distances_km[station_index, observed_indices]
                order = np.argsort(candidate_distances)
                order = order[candidate_distances[order] <= IDW_MAX_DISTANCE_KM]
                order = order[:IDW_NEIGHBORS]

                if len(order) >= MIN_IDW_NEIGHBORS:
                    distances = np.maximum(candidate_distances[order], 0.1)
                    weights = 1.0 / np.power(distances, IDW_POWER)
                    # Direct IDW is a convex combination of same-hour values,
                    # so it cannot exceed the contributing observations' range.
                    estimate = float(
                        np.average(current[observed_indices[order]], weights=weights)
                    )
                    completed[time_index, station_index, variable_index] = estimate
                    source[time_index, station_index, variable_index] = 1
                elif (
                    np.isfinite(last_anomaly[station_index, variable_index])
                    and time_index - last_time[station_index, variable_index]
                    <= HISTORY_HOURS
                ):
                    completed[time_index, station_index, variable_index] = (
                        expected[station_index]
                        + last_anomaly[station_index, variable_index]
                    )
                    source[time_index, station_index, variable_index] = 2
                else:
                    completed[time_index, station_index, variable_index] = expected[
                        station_index
                    ]
                    source[time_index, station_index, variable_index] = 3

            final_anomaly = (
                completed[time_index, :, variable_index] - expected
            )
            valid_final = np.isfinite(final_anomaly)
            last_anomaly[valid_final, variable_index] = final_anomaly[valid_final]
            last_time[valid_final, variable_index] = time_index

    for variable_index, variable in enumerate(VARIABLES):
        lower, upper = IMPUTATION_BOUNDS[variable]
        completed[:, :, variable_index] = np.clip(
            completed[:, :, variable_index], lower, upper
        )
    # Allow a small sensor-consistency tolerance but prevent an imputed dew
    # point from materially exceeding the corresponding air temperature.
    air_index = VARIABLES.index("air_temp_c_mean")
    dewpoint_index = VARIABLES.index("dewpoint_c_mean")
    completed[:, :, dewpoint_index] = np.minimum(
        completed[:, :, dewpoint_index], completed[:, :, air_index] + 2.0
    )

    if not np.isfinite(completed).all():
        raise RuntimeError("Imputation finished with non-finite values remaining.")
    return completed, source


def eligible_hours_and_samples(
    hours: pd.DatetimeIndex,
    observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    station_complete = np.isfinite(observed).all(axis=2)
    complete_count = station_complete.sum(axis=1)
    required_count = math.ceil(MIN_HOURLY_STATION_COVERAGE * observed.shape[1])
    hour_eligible = complete_count >= required_count

    eligible_series = pd.Series(hour_eligible, index=hours)
    full_calendar = eligible_series.reindex(
        pd.date_range(hours.min(), hours.max(), freq="h"), fill_value=False
    )
    window_length = HISTORY_HOURS + 1
    endpoint_calendar = full_calendar.rolling(
        window_length, min_periods=window_length
    ).sum().eq(window_length)
    endpoint_hours = endpoint_calendar.index[endpoint_calendar]
    hour_lookup = pd.Series(np.arange(len(hours)), index=hours)
    endpoint_indices = hour_lookup.loc[endpoint_hours].to_numpy(dtype=int)
    return hour_eligible, complete_count, endpoint_indices


def export_station_hour_table(
    hours,
    station_ids,
    latitude,
    longitude,
    completed,
    source,
    hour_eligible,
    complete_count,
):
    qualified_indices = np.flatnonzero(hour_eligible)
    frames = []
    for time_index in qualified_indices:
        utc_hour = pd.Timestamp(hours[time_index]).tz_localize("UTC")
        local_hour = utc_hour.tz_convert(IOWA_TIMEZONE)
        frame = pd.DataFrame(
            {
                EVENT_COLUMN: hours[time_index],
                "hour_iowa_local": local_hour.strftime("%Y-%m-%d %H:%M:%S%z"),
                STATION_COLUMN: station_ids,
                "latitude": latitude,
                "longitude": longitude,
                "observed_complete_station_count": complete_count[time_index],
                "observed_station_coverage_pct": 100.0
                * complete_count[time_index]
                / len(station_ids),
            }
        )
        for variable_index, variable in enumerate(VARIABLES):
            frame[variable] = completed[time_index, :, variable_index]
            frame[f"{variable}_imputation_source"] = [
                SOURCE_NAMES[int(code)]
                for code in source[time_index, :, variable_index]
            ]
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(
        OUTPUT_DIR / "4_qualified_imputed_60station_hourly_events.csv",
        index=False,
    )


def export_samples(
    hours,
    station_ids,
    completed,
    source,
    endpoint_indices,
):
    offsets = np.arange(-HISTORY_HOURS, 1, dtype=np.int8)
    sample_time_indices = endpoint_indices[:, None] + offsets[None, :]
    samples = completed[sample_time_indices].astype(np.float32)
    imputation_mask = (source[sample_time_indices] != 0)
    endpoint_hours = hours[endpoint_indices]
    endpoint_hours_local = endpoint_hours.tz_localize("UTC").tz_convert(IOWA_TIMEZONE)

    np.savez_compressed(
        OUTPUT_DIR / "4_clustering_samples_7h_60stations.npz",
        X=samples,
        imputation_mask=imputation_mask,
        sample_hour_utc=endpoint_hours.astype("datetime64[ns]").to_numpy(),
        # Stored as ISO strings because NumPy datetime64 cannot retain a time zone.
        sample_hour_iowa_local=np.asarray(
            endpoint_hours_local.strftime("%Y-%m-%d %H:%M:%S%z")
        ),
        station_ids=np.asarray(station_ids),
        variables=np.asarray(VARIABLES),
        hour_offsets=offsets,
    )

    sample_index = pd.DataFrame(
        {
            "sample_id": np.arange(len(endpoint_hours)),
            "sample_hour_iowa_local": endpoint_hours_local.strftime(
                "%Y-%m-%d %H:%M:%S%z"
            ),
            "sample_hour_utc": endpoint_hours,
            "sample_year": endpoint_hours_local.year,
            "split": np.where(
                endpoint_hours_local.year == TEST_YEAR, "test", "development"
            ),
            "imputed_cell_count": imputation_mask.sum(axis=(1, 2, 3)),
            "total_cell_count": np.prod(samples.shape[1:]),
        }
    )
    sample_index["imputed_cell_pct"] = (
        100.0 * sample_index["imputed_cell_count"] / sample_index["total_cell_count"]
    )
    sample_index.to_csv(OUTPUT_DIR / "4_clustering_sample_index.csv", index=False)

    feature_order = []
    feature_index = 0
    for hour_offset in offsets:
        for station_id in station_ids:
            for variable in VARIABLES:
                feature_order.append(
                    {
                        "flattened_feature_index": feature_index,
                        "hour_offset": int(hour_offset),
                        STATION_COLUMN: station_id,
                        "variable": variable,
                    }
                )
                feature_index += 1
    pd.DataFrame(feature_order).to_csv(
        OUTPUT_DIR / "4_clustering_feature_order.csv", index=False
    )
    return samples, imputation_mask, sample_index


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data, station_file = load_inputs()
    station_ids = sorted(station_file[STATION_COLUMN].astype(str).unique())
    station_meta = (
        data.groupby(STATION_COLUMN, as_index=False)
        .agg(latitude=("latitude", "median"), longitude=("longitude", "median"))
        .set_index(STATION_COLUMN)
        .reindex(station_ids)
    )
    latitude = station_meta.latitude.to_numpy(dtype=float)
    longitude = station_meta.longitude.to_numpy(dtype=float)
    distances_km = haversine_matrix(latitude, longitude)

    print("Building station-hour arrays...", flush=True)
    hours, observed = build_arrays(data, station_ids)
    hour_eligible, complete_count, endpoint_indices = eligible_hours_and_samples(
        hours, observed
    )
    print(
        f"Qualified hourly events: {hour_eligible.sum():,}/{len(hours):,}; "
        f"eligible seven-hour samples: {len(endpoint_indices):,}",
        flush=True,
    )

    print("Estimating 2016-2017 climatology...", flush=True)
    climatology = training_climatology(hours, observed)
    print("Imputing missing values...", flush=True)
    completed, source = impute_values(
        hours, observed, climatology, distances_km
    )

    print("Exporting completed station-hour events...", flush=True)
    export_station_hour_table(
        hours,
        station_ids,
        latitude,
        longitude,
        completed,
        source,
        hour_eligible,
        complete_count,
    )
    print("Exporting clustering samples...", flush=True)
    samples, imputation_mask, sample_index = export_samples(
        hours, station_ids, completed, source, endpoint_indices
    )

    source_counts = {
        SOURCE_NAMES[code]: int((source[hour_eligible] == code).sum())
        for code in SOURCE_NAMES
    }
    summary = {
        "input_station_count": len(station_ids),
        "variables": VARIABLES,
        "minimum_hourly_station_coverage": MIN_HOURLY_STATION_COVERAGE,
        "minimum_complete_stations_per_hour": math.ceil(
            MIN_HOURLY_STATION_COVERAGE * len(station_ids)
        ),
        "history_hours": HISTORY_HOURS,
        "qualified_hourly_event_count": int(hour_eligible.sum()),
        "eligible_seven_hour_sample_count": int(len(endpoint_indices)),
        "development_sample_count": int(sample_index.split.eq("development").sum()),
        "test_2018_sample_count": int(sample_index.split.eq("test").sum()),
        "sample_tensor_shape": list(samples.shape),
        "sample_feature_count": int(np.prod(samples.shape[1:])),
        "mean_sample_imputed_cell_pct": float(sample_index.imputed_cell_pct.mean()),
        "maximum_sample_imputed_cell_pct": float(sample_index.imputed_cell_pct.max()),
        "qualified_hour_value_source_counts": source_counts,
        "imputation_order": [
            "same-hour direct spatial IDW",
            "most recent completed anomaly within six hours",
            "2016-2017 station-month-hour climatology",
        ],
        "leakage_control": "Climatology is estimated from 2016-2017 only; 2018 is transformation/test only.",
    }
    (OUTPUT_DIR / "4_preparation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("Saved outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
