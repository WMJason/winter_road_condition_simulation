#!/usr/bin/env python3
"""Find RWIS stations shared across hourly events in the modeling table.

The script reports two kinds of overlap:

1. Reporting overlap: a station has a station-hour row in an event.
2. Complete-measurement overlap: that row has non-missing values for every
   variable listed in REQUIRED_MEASUREMENT_COLUMNS.

Edit the configuration below if a different event time field, input table,
or set of required measurements is needed. No command-line arguments are
required.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
IOWA_TIMEZONE = "America/Chicago"
INPUT_DIR = SCRIPT_DIR / "2b_quality_controlled_outputs"
INPUT_FILE_PATTERN = "2b_rwis_station_hourly_modeling_table_qc_*.csv"
OUTPUT_DIR = SCRIPT_DIR / "3_most_overlapped_rwis_outputs"

# Use hour_utc because it is unambiguous across daylight-saving transitions.
EVENT_COLUMN = "hour_utc"
STATION_COLUMN = "rwis_id"
REQUIRED_MEASUREMENT_COLUMNS = [
    "air_temp_c_mean",
    "dewpoint_c_mean",
    "road_surface_temp_c_mean",
]
COVERAGE_THRESHOLDS = (1.00, 0.99, 0.95, 0.90)
# Selection used for the downstream seven-hour weather-pattern samples.
# Choose "rst" to require only RST or "all_weather" to require all variables
# in WEATHER_VARIABLES at every station-hour.
SELECTION_VARIABLE_MODE = "all_weather"
MIN_ANNUAL_HOURLY_COVERAGE = 0.90
HISTORY_HOURS = 6
WEATHER_VARIABLES = {
    # road_surface_temp_c_mean is the mean of the available tfs0--tfs3
    # sensors. Requiring road_surface_sensor_count >= 1 implements the rule
    # that one valid pavement sensor is sufficient for that station-hour.
    "rst": {
        "value_column": "road_surface_temp_c_mean",
        "sensor_count_column": "road_surface_sensor_count",
    },
    "air_temperature": {"value_column": "air_temp_c_mean"},
    "dewpoint": {"value_column": "dewpoint_c_mean"},
    "humidity": {"value_column": "humidity_mean"},
    "wind_speed": {"value_column": "wind_speed_mean"},
}


def find_input_file() -> Path:
    files = sorted(INPUT_DIR.glob(INPUT_FILE_PATTERN))
    if not files:
        raise FileNotFoundError(
            f"No modeling table matching {INPUT_FILE_PATTERN!r} in {INPUT_DIR}"
        )
    if len(files) > 1:
        print("Multiple modeling tables found; using the newest modified file:")
        files = sorted(files, key=lambda path: path.stat().st_mtime)
        for path in files:
            print(" -", path.name)
    return files[-1]


def validate_columns(columns: list[str]) -> None:
    required = {EVENT_COLUMN, STATION_COLUMN, *REQUIRED_MEASUREMENT_COLUMNS}
    missing = sorted(required.difference(columns))
    if missing:
        raise ValueError(f"Input table is missing required columns: {missing}")


def coverage_label(threshold: float) -> str:
    return f"at_least_{int(round(threshold * 100))}_pct"


def main() -> None:
    input_path = find_input_file()
    print("Reading:", input_path)

    columns = pd.read_csv(input_path, nrows=0).columns.tolist()
    validate_columns(columns)
    usecols = [
        EVENT_COLUMN,
        STATION_COLUMN,
        "latitude",
        "longitude",
        "rwis_obs_count",
        *REQUIRED_MEASUREMENT_COLUMNS,
        "road_surface_sensor_count",
        "humidity_mean",
        "wind_speed_mean",
    ]
    usecols = list(dict.fromkeys(usecols))
    data = pd.read_csv(input_path, usecols=usecols)
    data[EVENT_COLUMN] = pd.to_datetime(data[EVENT_COLUMN], errors="coerce")
    data = data.dropna(subset=[EVENT_COLUMN, STATION_COLUMN]).copy()

    # Defensive aggregation prevents accidental duplicate station-hour rows
    # from inflating event counts.
    data["has_reporting_row"] = True
    data["has_complete_measurements"] = data[
        REQUIRED_MEASUREMENT_COLUMNS
    ].notna().all(axis=1)
    station_events = (
        data.groupby([STATION_COLUMN, EVENT_COLUMN], as_index=False)
        .agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            rwis_obs_count=("rwis_obs_count", "sum"),
            has_reporting_row=("has_reporting_row", "max"),
            has_complete_measurements=("has_complete_measurements", "max"),
        )
    )

    all_events = pd.Index(sorted(data[EVENT_COLUMN].unique()))
    total_events = len(all_events)
    if total_events == 0:
        raise ValueError("The input table contains no valid hourly events.")

    station_location = (
        station_events.groupby(STATION_COLUMN, as_index=False)
        .agg(latitude=("latitude", "median"), longitude=("longitude", "median"))
    )
    reporting = (
        station_events.groupby(STATION_COLUMN, as_index=False)
        .agg(
            reporting_event_count=(EVENT_COLUMN, "nunique"),
            total_raw_observation_count=("rwis_obs_count", "sum"),
        )
    )
    complete = (
        station_events[station_events["has_complete_measurements"]]
        .groupby(STATION_COLUMN, as_index=False)
        .agg(complete_event_count=(EVENT_COLUMN, "nunique"))
    )

    coverage = station_location.merge(reporting, on=STATION_COLUMN, how="left")
    coverage = coverage.merge(complete, on=STATION_COLUMN, how="left")
    coverage["complete_event_count"] = coverage["complete_event_count"].fillna(0).astype(int)
    coverage["reporting_coverage_pct"] = (
        100.0 * coverage["reporting_event_count"] / total_events
    )
    coverage["complete_measurement_coverage_pct"] = (
        100.0 * coverage["complete_event_count"] / total_events
    )

    for threshold in COVERAGE_THRESHOLDS:
        suffix = coverage_label(threshold)
        coverage[f"reporting_{suffix}"] = (
            coverage["reporting_event_count"] >= threshold * total_events
        )
        coverage[f"complete_{suffix}"] = (
            coverage["complete_event_count"] >= threshold * total_events
        )

    coverage = coverage.sort_values(
        ["complete_event_count", "reporting_event_count", STATION_COLUMN],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    coverage.insert(0, "coverage_rank", range(1, len(coverage) + 1))

    strict_reporting = coverage[
        coverage["reporting_event_count"].eq(total_events)
    ].copy()
    strict_complete = coverage[
        coverage["complete_event_count"].eq(total_events)
    ].copy()

    event_summary = (
        station_events.groupby(EVENT_COLUMN, as_index=False)
        .agg(
            reporting_station_count=(STATION_COLUMN, "nunique"),
            complete_station_count=("has_complete_measurements", "sum"),
        )
        .sort_values(EVENT_COLUMN)
    )
    event_summary.insert(
        1,
        "hour_iowa_local",
        pd.to_datetime(event_summary[EVENT_COLUMN], utc=True)
        .dt.tz_convert(IOWA_TIMEZONE)
        .dt.strftime("%Y-%m-%d %H:%M:%S%z"),
    )

    threshold_rows = []
    for threshold in COVERAGE_THRESHOLDS:
        threshold_rows.append(
            {
                "minimum_coverage_pct": threshold * 100,
                "minimum_event_count": math.ceil(threshold * total_events),
                "reporting_station_count": int(
                    (coverage["reporting_event_count"] >= threshold * total_events).sum()
                ),
                "complete_measurement_station_count": int(
                    (coverage["complete_event_count"] >= threshold * total_events).sum()
                ),
            }
        )
    threshold_summary = pd.DataFrame(threshold_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(OUTPUT_DIR / "3_rwis_station_hourly_coverage_ranked.csv", index=False)
    strict_reporting.to_csv(
        OUTPUT_DIR / "3_rwis_stations_reporting_in_every_hour.csv", index=False
    )
    strict_complete.to_csv(
        OUTPUT_DIR / "3_rwis_stations_complete_in_every_hour.csv", index=False
    )
    event_summary.to_csv(
        OUTPUT_DIR / "3_hourly_event_station_counts.csv", index=False
    )
    threshold_summary.to_csv(
        OUTPUT_DIR / "3_rwis_overlap_threshold_summary.csv", index=False
    )

    # Export direct station lists for each individual weather variable. These
    # lists avoid requiring all weather variables to be present simultaneously.
    utc_year = data[EVENT_COLUMN].dt.year
    analysis_years = sorted(year for year in utc_year.unique() if 2016 <= year <= 2018)
    variable_summaries = []
    combined_variable_coverage = station_location.copy()
    for variable_name, specification in WEATHER_VARIABLES.items():
        value_column = specification["value_column"]
        valid = data[value_column].notna()
        sensor_count_column = specification.get("sensor_count_column")
        if sensor_count_column:
            valid &= data[sensor_count_column].ge(1)

        variable_data = data.loc[valid].copy()
        variable_data["utc_year"] = variable_data[EVENT_COLUMN].dt.year
        variable_data = variable_data[variable_data["utc_year"].isin(analysis_years)]
        variable_data = variable_data.drop_duplicates([STATION_COLUMN, EVENT_COLUMN])

        per_station = (
            variable_data.groupby(STATION_COLUMN, as_index=False)
            .agg(
                valid_hour_count=(EVENT_COLUMN, "nunique"),
                years_with_valid_observation=("utc_year", "nunique"),
                first_valid_hour=(EVENT_COLUMN, "min"),
                last_valid_hour=(EVENT_COLUMN, "max"),
            )
            .merge(station_location, on=STATION_COLUMN, how="left")
        )
        per_station["valid_hour_coverage_pct"] = (
            100.0 * per_station["valid_hour_count"] / total_events
        )
        per_station["reports_in_every_analysis_year"] = (
            per_station["years_with_valid_observation"].eq(len(analysis_years))
        )
        for threshold in COVERAGE_THRESHOLDS:
            per_station[coverage_label(threshold)] = (
                per_station["valid_hour_count"] >= threshold * total_events
            )
        per_station = per_station.sort_values(
            ["valid_hour_count", STATION_COLUMN], ascending=[False, True]
        ).reset_index(drop=True)
        per_station.insert(0, "coverage_rank", range(1, len(per_station) + 1))
        per_station.to_csv(
            OUTPUT_DIR / f"3_rwis_stations_reporting_{variable_name}.csv",
            index=False,
        )

        every_year = per_station[per_station["reports_in_every_analysis_year"]].copy()
        every_year.to_csv(
            OUTPUT_DIR / f"3_rwis_stations_reporting_{variable_name}_in_every_year.csv",
            index=False,
        )

        renamed = per_station[
            [STATION_COLUMN, "valid_hour_count", "valid_hour_coverage_pct", "reports_in_every_analysis_year"]
        ].rename(
            columns={
                "valid_hour_count": f"{variable_name}_valid_hour_count",
                "valid_hour_coverage_pct": f"{variable_name}_coverage_pct",
                "reports_in_every_analysis_year": f"{variable_name}_reports_every_year",
            }
        )
        combined_variable_coverage = combined_variable_coverage.merge(
            renamed, on=STATION_COLUMN, how="left"
        )
        variable_summaries.append(
            {
                "weather_variable": variable_name,
                "stations_with_any_valid_observation": int(len(per_station)),
                "stations_reporting_in_every_analysis_year": int(len(every_year)),
                **{
                    f"stations_at_least_{int(threshold * 100)}_pct_hours": int(
                        per_station[coverage_label(threshold)].sum()
                    )
                    for threshold in COVERAGE_THRESHOLDS
                },
            }
        )

    combined_variable_coverage.to_csv(
        OUTPUT_DIR / "3_rwis_weather_variable_station_coverage_combined.csv",
        index=False,
    )
    every_year_columns = [
        f"{variable_name}_reports_every_year" for variable_name in WEATHER_VARIABLES
    ]
    all_weather_every_year = combined_variable_coverage[
        combined_variable_coverage[every_year_columns].eq(True).all(axis=1)
    ].copy()
    all_weather_every_year.to_csv(
        OUTPUT_DIR / "3_rwis_stations_reporting_all_weather_variables_every_year.csv",
        index=False,
    )
    variable_summary = pd.DataFrame(variable_summaries)
    variable_summary.to_csv(
        OUTPUT_DIR / "3_rwis_weather_variable_station_counts.csv", index=False
    )

    # Select a stable station network and enumerate complete t,...,t-6 samples.
    if SELECTION_VARIABLE_MODE == "rst":
        selection_valid = (
            data["road_surface_temp_c_mean"].notna()
            & data["road_surface_sensor_count"].ge(1)
        )
    elif SELECTION_VARIABLE_MODE == "all_weather":
        selection_valid = data[
            [spec["value_column"] for spec in WEATHER_VARIABLES.values()]
        ].notna().all(axis=1) & data["road_surface_sensor_count"].ge(1)
    else:
        raise ValueError(
            "SELECTION_VARIABLE_MODE must be 'rst' or 'all_weather', got "
            f"{SELECTION_VARIABLE_MODE!r}"
        )

    availability = (
        data.loc[selection_valid, [EVENT_COLUMN, STATION_COLUMN]]
        .drop_duplicates()
        .assign(available=True)
        .pivot(index=EVENT_COLUMN, columns=STATION_COLUMN, values="available")
        .reindex(all_events)
        .notna()
    )
    annual_coverage = pd.DataFrame(
        {
            year: availability[availability.index.year == year].mean()
            for year in analysis_years
        }
    )
    annual_coverage.columns = [f"coverage_{year}" for year in annual_coverage]
    annual_coverage["minimum_annual_coverage"] = annual_coverage.min(axis=1)
    selected_ids = annual_coverage.index[
        annual_coverage["minimum_annual_coverage"]
        >= MIN_ANNUAL_HOURLY_COVERAGE
    ]
    selected_stations = (
        annual_coverage.loc[selected_ids]
        .reset_index()
        .merge(station_location, on=STATION_COLUMN, how="left")
        .sort_values(["minimum_annual_coverage", STATION_COLUMN], ascending=[False, True])
    )
    selected_stations.to_csv(
        OUTPUT_DIR / "3_selected_consistent_rwis_stations.csv", index=False
    )

    complete_at_hour = availability.reindex(columns=selected_ids).all(axis=1)
    complete_calendar = complete_at_hour.reindex(
        pd.date_range(all_events.min(), all_events.max(), freq="h"),
        fill_value=False,
    )
    window_length = HISTORY_HOURS + 1
    eligible_window = complete_calendar.rolling(
        window_length, min_periods=window_length
    ).sum().eq(window_length)
    eligible_hours = pd.DataFrame(
        {"sample_hour_utc": eligible_window.index[eligible_window]}
    )
    local_hours = pd.to_datetime(eligible_hours["sample_hour_utc"], utc=True).dt.tz_convert(
        IOWA_TIMEZONE
    )
    eligible_hours.insert(
        0, "sample_hour_iowa_local", local_hours.dt.strftime("%Y-%m-%d %H:%M:%S%z")
    )
    eligible_hours["sample_year"] = local_hours.dt.year
    eligible_hours["split"] = eligible_hours["sample_year"].map(
        lambda year: "test" if year == 2018 else "development"
    )
    eligible_hours.to_csv(
        OUTPUT_DIR / "3_eligible_complete_7hour_sample_endpoints.csv", index=False
    )

    selection_summary = {
        "selection_variable_mode": SELECTION_VARIABLE_MODE,
        "minimum_annual_hourly_coverage": MIN_ANNUAL_HOURLY_COVERAGE,
        "history_hours": HISTORY_HOURS,
        "selected_station_count": int(len(selected_stations)),
        "development_sample_count": int(eligible_hours["split"].eq("development").sum()),
        "test_2018_sample_count": int(eligible_hours["split"].eq("test").sum()),
    }
    (OUTPUT_DIR / "3_consistent_station_selection_summary.json").write_text(
        json.dumps(selection_summary, indent=2), encoding="utf-8"
    )

    summary = {
        "input_file": str(input_path.resolve()),
        "event_column": EVENT_COLUMN,
        "first_event": str(all_events.min()),
        "last_event": str(all_events.max()),
        "total_hourly_events": total_events,
        "total_stations": int(coverage[STATION_COLUMN].nunique()),
        "required_measurement_columns": REQUIRED_MEASUREMENT_COLUMNS,
        "stations_reporting_in_every_event": int(len(strict_reporting)),
        "stations_complete_in_every_event": int(len(strict_complete)),
        "highest_reporting_coverage_pct": float(coverage["reporting_coverage_pct"].max()),
        "highest_complete_measurement_coverage_pct": float(
            coverage["complete_measurement_coverage_pct"].max()
        ),
    }
    (OUTPUT_DIR / "3_rwis_overlap_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nHourly events:", total_events)
    print("Stations:", len(coverage))
    print("Stations reporting in every event:", len(strict_reporting))
    print("Stations complete in every event:", len(strict_complete))
    print("\nOverlap by threshold:")
    print(threshold_summary.to_string(index=False))
    print("\nTop 10 stations by complete-measurement coverage:")
    print(
        coverage[
            [
                "coverage_rank",
                STATION_COLUMN,
                "reporting_event_count",
                "reporting_coverage_pct",
                "complete_event_count",
                "complete_measurement_coverage_pct",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print("\nStation counts by individual weather variable:")
    print(variable_summary.to_string(index=False))
    print(
        "\nStations reporting every weather variable in each analysis year:",
        len(all_weather_every_year),
    )
    print(", ".join(all_weather_every_year[STATION_COLUMN].tolist()))
    print("\nDownstream seven-hour sample selection:")
    print(json.dumps(selection_summary, indent=2))
    print("\nSaved outputs to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
