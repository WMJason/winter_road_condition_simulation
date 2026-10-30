# Input Data and Full-Run Requirements

## Not bundled

- The original SQLite archive and full station-hour observation tables.
- Individual completed seven-hour station histories and fitted model pickle/joblib objects, which can embed observations.
- Road GraphML, full support-point/elevation arrays, DEM tiles, and individual SGS temperature fields.
- Roadway imagery, image URLs, annotations, and classifier weights.

Public release of these inputs requires separate source/permission review. Compact aggregate results and numerical parameter exports do not replace the missing histories for a full rerun.

## SQLite schema used by Step 2a

The supplied archive must contain:

```sql
CREATE TABLE rwis_station (
    rwis_id TEXT,
    latitude REAL,
    longitude REAL
);
CREATE TABLE rwis_observation (
    id INTEGER,
    rwis_id TEXT,
    timestamp TEXT,
    air_temp_f REAL,
    dewpoint_f REAL,
    humidity REAL,
    wind_speed REAL,
    tfs0 REAL,
    tfs1 REAL,
    tfs2 REAL,
    tfs3 REAL
);
```

This describes columns read by the code, not the archive's complete original schema. Temperatures in this source schema are Fahrenheit, timestamps follow the project's UTC convention, humidity is percent, and wind speed is knots. Pavement fields are separate sensors. No database credentials are needed by the reader.

The default aggregation covers 2016--2018. Later scripts retain their matching filenames and cool-season window-selection rules. Changing the aggregation dates alone does not update downstream historical filenames or modelling assumptions.

## Files that later stages need

All paths below are relative to `workflow/analysis-iowa/`.

| Stage | Required upstream outputs / external inputs |
|---|---|
| QC and station selection | `RW_module/2a_modeling_outputs/` hourly table, then `2b_quality_controlled_outputs/` |
| Completed histories | QC table and Step 3's selected station list |
| K-means | Step 4 V2 NPZ/index plus the included fixed-camera test-hour metadata |
| Model fitting | Step 5 outputs, Step 4 histories, selected station list, and `decision_making/0c_elevation_outputs/0c_rwis_stations_with_elevation.csv` |
| Evaluation | Step 7 fitted joblib models, Step 5 assignments, and Step 4 histories |
| Parent selection | Step 7 models and their fitting-event audits, Step 5 grouping pipeline, and Step 4 histories |
| SGS | Step 9 HDF5 histories and model metadata; `decision_making/shared_cache/0b_road_support_points.csv.gz`; `decision_making/0c_elevation_outputs/0c_road_support_points_with_elevation.csv.gz` |

The compact diagnostics under `results/iowa/` are intentionally separate from these active output paths. They are not sufficient to run model fitting or SGS by themselves.

## Road support and elevation preparation

The optional `decision_making/0a_download_iowa_roads.py` downloads roads; `0b_prepare_road_support_points.py` builds support cells. `0c_add_elevation.py` reads `0c_elevation_config.json` and prepares station/support elevations. Its reported station inventory, `analysis-iowa/all_rwis_locations.csv`, is not included. Supply the original inventory or explicitly revise the elevation configuration according to available metadata.

These services may be slow or unavailable. A current download may have different link geometry and identifiers from the saved Iowa experiment. To reproduce that experiment, use its original graph and support catalogue rather than redownloading and assuming they match.

## Before a public GitHub release

Confirm the code license, coauthor approval, final citation, and redistribution status of derived results. Do not upload the entire original research directory. Review `provenance/export_manifest.csv`, keep large-input patterns excluded by `.gitignore`, and run `python tools/audit_release.py`.
