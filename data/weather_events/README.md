# Public Reference Fields

This directory contains all 700 issue-time simulated RST, drivability, and maintenance-need maps: 70 parent events with ten realizations each, seven weather groups, and 122,329 support points per map. These are simulated reference conditions, not independently observed full-network weather events.

Read [the dataset guide](../../docs/WEATHER_EVENTS.md) for category definitions, array dimensions, reconstruction, limitations, and attribution. Read [the website upload guide](../../docs/UPLOAD_WITH_BROWSER.md) to publish this update without GitHub Desktop.

All 19 ZIP archives must be present: `weather_support.zip` and `weather_events_part_01.zip` through `weather_events_part_18.zip`. The archives total 353.184 MiB; the largest is 20.293 MiB. Do not unpack them before uploading. `dataset_summary.json` lists the exact sizes and hashes; `scenario_manifest.csv` maps all 700 scenario identifiers to their arrays.

After downloading and extracting the repository ZIP, run from the repository root:

```sh
python -m pip install numpy
python tools/read_weather_events.py --verify
```

Code and data licenses remain pending author/coauthor approval. See the repository's `LICENSE_NOTICE.md`.
