import argparse
import sqlite3
from pathlib import Path

import pandas as pd


IOWA_TZ = "US/Central"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = SCRIPT_DIR.parent / "ROAD_DATA_2015-2019_IOWA.db"
DEFAULT_START_DATE = "2016-01-01"
DEFAULT_END_DATE = "2018-12-31"


def fahrenheit_to_celsius(series: pd.Series) -> pd.Series:
    return (series - 32.0) * 5.0 / 9.0


def read_rwis_observations(
    db_path: Path,
    start_ts: str,
    end_ts: str,
) -> pd.DataFrame:
    sql = """
    SELECT
        o.id,
        o.rwis_id,
        o.timestamp,
        o.air_temp_f,
        o.dewpoint_f,
        o.humidity,
        o.wind_speed,
        o.tfs0,
        o.tfs1,
        o.tfs2,
        o.tfs3,
        s.latitude,
        s.longitude
    FROM rwis_observation o
    JOIN rwis_station s USING (rwis_id)
    WHERE o.timestamp >= ? AND o.timestamp < ?
    ORDER BY o.rwis_id, o.timestamp
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=(start_ts, end_ts))


def add_time_fields(rwis: pd.DataFrame) -> pd.DataFrame:
    rwis = rwis.copy()

    # The existing project convention treats RWIS timestamps as UTC first, then
    # derives Iowa local time. Keep both timelines for unambiguous modeling.
    rwis["timestamp_utc"] = pd.to_datetime(rwis["timestamp"]).dt.tz_localize("UTC")
    rwis["hour_utc"] = rwis["timestamp_utc"].dt.floor("h")
    rwis["hour_local"] = rwis["hour_utc"].dt.tz_convert(IOWA_TZ)

    rwis["year"] = rwis["hour_local"].dt.year
    rwis["month"] = rwis["hour_local"].dt.month
    rwis["hour"] = rwis["hour_local"].dt.hour
    return rwis


def build_station_hourly_table(rwis: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "rwis_id",
        "latitude",
        "longitude",
        "hour_utc",
        "hour_local",
        "year",
        "month",
        "hour",
    ]

    station_hourly = (
        rwis.groupby(group_cols, dropna=False)
        .agg(
            rwis_obs_count=("id", "count"),
            air_temp_f_mean=("air_temp_f", "mean"),
            air_temp_f_min=("air_temp_f", "min"),
            air_temp_f_max=("air_temp_f", "max"),
            dewpoint_f_mean=("dewpoint_f", "mean"),
            dewpoint_f_min=("dewpoint_f", "min"),
            dewpoint_f_max=("dewpoint_f", "max"),
            humidity_mean=("humidity", "mean"),
            wind_speed_mean=("wind_speed", "mean"),
            tfs0_f_mean=("tfs0", "mean"),
            tfs1_f_mean=("tfs1", "mean"),
            tfs2_f_mean=("tfs2", "mean"),
            tfs3_f_mean=("tfs3", "mean"),
        )
        .reset_index()
    )

    tfs_cols = ["tfs0_f_mean", "tfs1_f_mean", "tfs2_f_mean", "tfs3_f_mean"]
    station_hourly["road_surface_temp_f_mean"] = station_hourly[tfs_cols].mean(
        axis=1,
        skipna=True,
    )
    station_hourly["road_surface_sensor_count"] = station_hourly[tfs_cols].notna().sum(axis=1)

    temperature_columns_f = [
        "air_temp_f_mean",
        "air_temp_f_min",
        "air_temp_f_max",
        "dewpoint_f_mean",
        "dewpoint_f_min",
        "dewpoint_f_max",
        "tfs0_f_mean",
        "tfs1_f_mean",
        "tfs2_f_mean",
        "tfs3_f_mean",
        "road_surface_temp_f_mean",
    ]
    for col in temperature_columns_f:
        station_hourly[col.replace("_f_", "_c_")] = fahrenheit_to_celsius(
            station_hourly[col]
        )

    station_hourly["hour_utc"] = station_hourly["hour_utc"].dt.tz_localize(None)
    station_hourly["hour_local"] = station_hourly["hour_local"].dt.tz_localize(None)

    ordered_columns = [
        "hour_utc",
        "hour_local",
        "year",
        "month",
        "hour",
        "rwis_id",
        "latitude",
        "longitude",
        "rwis_obs_count",
        "air_temp_f_mean",
        "air_temp_c_mean",
        "air_temp_f_min",
        "air_temp_c_min",
        "air_temp_f_max",
        "air_temp_c_max",
        "dewpoint_f_mean",
        "dewpoint_c_mean",
        "dewpoint_f_min",
        "dewpoint_c_min",
        "dewpoint_f_max",
        "dewpoint_c_max",
        "humidity_mean",
        "wind_speed_mean",
        "tfs0_f_mean",
        "tfs0_c_mean",
        "tfs1_f_mean",
        "tfs1_c_mean",
        "tfs2_f_mean",
        "tfs2_c_mean",
        "tfs3_f_mean",
        "tfs3_c_mean",
        "road_surface_temp_f_mean",
        "road_surface_temp_c_mean",
        "road_surface_sensor_count",
    ]
    return station_hourly[ordered_columns].sort_values(["hour_utc", "rwis_id"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a station-hour RWIS modeling table with UTC/local time and F/C temperatures."
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        type=Path,
        help=(
            "Path to the Iowa road database. Defaults to the database in the "
            "parent analysis directory."
        ),
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument(
        "--output-dir", default=SCRIPT_DIR / "2a_modeling_outputs", type=Path
    )
    args = parser.parse_args()

    db_path = args.db.expanduser().resolve()
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}\n"
            "Provide its location with --db /full/path/to/database.db"
        )

    # User-supplied dates refer to Iowa calendar days. Convert those boundaries
    # to UTC before querying because the database timestamps are stored in UTC.
    start_local = pd.Timestamp(args.start_date, tz=IOWA_TZ)
    end_local = pd.Timestamp(args.end_date, tz=IOWA_TZ) + pd.Timedelta(days=1)
    start_ts = start_local.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")
    end_ts = end_local.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")

    print("Loading RWIS observations...")
    rwis = read_rwis_observations(db_path, start_ts, end_ts)
    print("Rows loaded:", len(rwis))

    print("Aggregating to station-hour rows...")
    station_hourly = build_station_hourly_table(add_time_fields(rwis))

    args.output_dir.mkdir(exist_ok=True)
    date_tag = f"{args.start_date}_to_{args.end_date}"
    output_path = args.output_dir / f"2_rwis_station_hourly_modeling_table_{date_tag}.csv"
    station_hourly.to_csv(output_path, index=False)

    print("RWIS station-hour modeling table created")
    print("Rows:", len(station_hourly))
    print("Stations:", station_hourly["rwis_id"].nunique())
    print(
        "Date range Iowa local time:",
        station_hourly["hour_local"].min(),
        "to",
        station_hourly["hour_local"].max(),
    )
    print("Saved to:", output_path.resolve())


if __name__ == "__main__":
    main()
