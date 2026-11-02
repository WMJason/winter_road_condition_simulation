# The 700 Simulated Reference Events

## Scope

Seven weather groups each contribute ten historical parent events. Ten conditional RST realizations were generated for each parent, giving 700 issue-time reference maps at 122,329 road support points. Matching drivability and maintenance-need categories were generated from empirical temperature/illumination relationships with spatial dependence. The two visual applications were simulated separately. They are not observed networkwide conditions and are not guaranteed to describe a physically consistent joint surface state at every point.

The public arrays are copied from the completed research outputs. No fields are rerun, rounded, spatially subsampled, or replaced by an illustrative example. This release contains the current issue-time fields only. The seven-hour RWIS history used for conditioning is not a seven-hour full-network simulation and is not included here.

## Files

- `data/weather_events/scenario_manifest.csv`: one row per realization, including parent identifier, group, UTC and Iowa-local issue time, zero-based realization index, and archive/member locations.
- `data/weather_events/weather_events_part_*.zip`: compressed arrays and metadata for all 70 parents. Each parent NPZ contains ten realizations.
- `data/weather_events/weather_support.zip`: one shared `support_points.npz`, in the exact order used by every field.
- `data/weather_events/source_checksums.csv`: hashes of each original RST/visual HDF5 file and exported parent NPZ. These are provenance records, not downloadable paths to the original private workspace.
- `data/weather_events/dataset_summary.json`: dimensions, exact file sizes, and SHA-256 checksums of all downloadable archives.

Archives remain ordinary ZIP files in GitHub; uploading a ZIP does not unpack it into repository files. The Python reader opens archives directly. Downloading the complete repository ZIP preserves all relative paths.

## Array Dictionary

Within each parent NPZ:

| Array | Shape | Meaning |
|---|---|---|
| `hour_offsets` | `(1,)` | Contains zero: issue time |
| `trend_c` | `(1, 122329)` | Saved regression trend in degrees Celsius |
| `residual_c` | `(10, 1, 122329)` | Saved conditional residual realizations in degrees Celsius |
| `drivability` | `(10, 122329)` | 0 normal; 1 caution; 2 not advised |
| `maintenance_need` | `(10, 122329)` | 0 monitor only; 1 treatment needed; 2 plowing needed |
| `low_natural_illumination` | `(122329,)` | 1 low natural illumination; 0 higher natural illumination |
| `solar_altitude_degrees` | `(122329,)` | Solar-altitude proxy used for natural illumination |

Reconstruct the reference RST using `trend_c[0] + residual_c[realization_index, 0]`. This is the original saved reconstruction, not a new prediction. Realization index 0 corresponds to the identifier suffix `SGS01`. Reference categories contain no unknown values. They are distinct from the correct, incorrect, or unknown reports generated in decision experiments.

The shared support catalogue contains longitude and latitude in degrees; elevation in metres; represented `length_km`; physical-link and support-cell identifiers; and segment order. String identifiers are stored as fixed-width byte arrays and can be decoded with `.astype(str)`. Each point represents road length, not an independent road link. Use the represented lengths for network proportions and length-weighted summaries. Longitude and latitude are not kilometre coordinates. The catalogue does not contain the full road geometry, directed connectivity, or station observations needed to reproduce routing from scratch.

## Read Without Unpacking

```python
from tools.read_weather_events import read_npz, read_scenario, DEFAULT_DATA

event = read_scenario('C00_R01_S5447__SGS01')
support = read_npz(DEFAULT_DATA / 'weather_support.zip', 'support_points.npz')
rst = event['rst_c']
drivability = event['drivability']
length = support['length_km']
not_advised_km = length[drivability == 2].sum()
print(not_advised_km)
```

Only NumPy and the Python standard library are required to read these archives. The exporter additionally requires h5py and access to the original outputs. Run `python tools/read_weather_events.py --verify` to check every archive hash, all parent dimensions, and category codes after downloading.

## Interpretation and Rights

Realizations from the same parent share conditioning observations and fitted models. They must not be treated as 700 independent historical events. Near-centroid parent selection represents typical fitted groups rather than the frequency of all Iowa winter conditions or rare extremes. These fields support analysis under the study's fitted assumptions; they are not live road advice, verified crash risk, or actual treatment requirements.

The support locations and identifiers derive from OpenStreetMap: copyright OpenStreetMap contributors, https://www.openstreetmap.org/copyright. OpenStreetMap database terms apply to derived network data; this notice does not replace review of applicable attribution/share-alike requirements. Elevations are derived from the study's USGS DEM processing. No raw images, raw historical station records, DEM tiles, or image-redistribution permissions are supplied. The authors have not selected a code or simulated-data license; public downloadability is not an unrestricted reuse license. Confirm code/data licensing with coauthors before publication of this update.
