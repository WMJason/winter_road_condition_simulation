# RWIS Weather Grouping, STRK, and Conditional RST Simulation

A compact research package using road weather information system (RWIS) observations for hourly data preparation, weather-event grouping, road surface temperature (RST) estimation with spatiotemporal regression kriging (STRK), and conditional networkwide simulation. Iowa is the case study; the workflow can be adapted to another region after changing its inputs and regional assumptions.

This release stops at RST simulation. Image classification, visual-condition simulation, routing, and maintenance prioritization are not included.

## What the pipeline does

1. Aggregate RWIS records into station-hour observations and screen unreasonable values.
2. Select consistently reporting stations and complete missing observations.
3. Form seven-hour histories with shape **7 hours x 60 stations x 5 variables**.
4. Exclude protected evaluation periods, standardize the fitting histories, apply PCA, and group histories using K-means.
5. Fit a regression trend and spatiotemporal residual dependence model for each group. Estimate issue-time RST using regression kriging.
6. Select near-centroid parent histories and generate conditional RST fields using sequential Gaussian simulation (SGS).

The Iowa run produced **seven groups, 70 parent events, and 700 realizations**. These are simulated reference fields, not observed temperatures throughout the road network.

## Contents

| Directory | Contents |
|---|---|
| `workflow/analysis-iowa/RW_module/` | Current scripts and configurations, retaining their original step numbers |
| `workflow/analysis-iowa/decision_making/` | Optional road-support and elevation preparation scripts |
| `workflow/analysis-iowa/visual_module/` | Test-event timing metadata only; no imagery or category labels |
| `results/iowa/` | Existing compact diagnostics, coefficient tables, evaluation summaries, figures, and SGS statistics |
| `models/` | Numerical K-means/PCA parameters and group-specific regression/covariance parameters; no pickle files |
| `examples/` | A runnable synthetic demonstration, separate from the Iowa findings |
| `docs/` | Methods, input requirements, output definitions, and reproduction limits |
| `provenance/` | Source-to-export checksums and release accounting |

## Quick Start: No Original Data Required

Use Python 3.10 or newer. The included numerical dependencies were tested locally with Python 3.10.7 on macOS. Windows and Linux use the same Python commands, but have not been directly executed in this release preparation.

On Windows, use a short local working path, such as `C:\work\winter-rwis-simulation`, rather than a deeply nested OneDrive folder. Large production HDF5 outputs can otherwise encounter path-length or synchronization locks. The package does not download missing inputs automatically.

```sh
python -m venv .venv
```

Activate the environment using `.venv/bin/activate` on macOS/Linux or `.venv\Scripts\activate` on Windows, then run:

```sh
python -m pip install -r requirements-tested.txt
python examples/synthetic_demo.py
python -m unittest discover -s tests -v
python tools/audit_release.py
```

The demo creates artificial station observations, aggregates repeated records, fills missing values, applies PCA/K-means, fits group-specific regression and residual semivariograms, performs an example station-history holdout nowcast, and simulates three conditional fields per selected synthetic parent. It saves a small figure and statistical tables under `examples/generated/`.

A checked example figure and summary are included under `examples/reference_outputs/`. They are clearly separated from the Iowa results.

**The demo is not the Iowa experiment.** It uses fewer histories, reduced pair sampling, and empirical-fit RMSE to select covariance families rather than the complete event-fold selection procedure. Its generated numerical results must not be reported as Iowa findings.

## Reproduce the Iowa Workflow

The full RWIS SQLite archive, individual station histories, road graph/support catalogue, elevation arrays, and full SGS fields are deliberately excluded. See [Data Requirements](docs/DATA_REQUIREMENTS.md) before running a full stage.

Run commands from the repository root:

```sh
python run_pipeline.py --stage aggregate --db /path/to/ROAD_DATA_2015-2019_IOWA.db
python run_pipeline.py --stage qc
python run_pipeline.py --stage stations
python run_pipeline.py --stage prepare
python run_pipeline.py --stage cluster
python run_pipeline.py --stage visualize
python run_pipeline.py --stage fit
python run_pipeline.py --stage evaluate
python run_pipeline.py --stage select
python run_pipeline.py --stage sgs
```

The runner uses the current Python interpreter and paths relative to this repository. `prepare` rebuilds the completed tensor from the QC table rather than requiring the old Step 4 output archive. Saved examples in `results/iowa/` are **not** silently copied into runnable output folders. Full stages write new outputs next to their scripts.

Two older-named files remain because current stages import their helper functions: `4_prepare_imputed_weather_regime_samples.py` and `7_fit_cluster_specific_regression_variograms.py`. They are dependencies, not alternative recommended entry points. Use the runner's stages rather than guessing from version numbers.

`--dry-run` prints commands without running them; it does not assert that data inputs are available. Stage-specific arguments can be passed through, for example:

```sh
python run_pipeline.py --stage sgs --max-events 1 --realizations 2 --workers 1 --max-support-points 256
```

This produces a partial test, not the complete experiment. Do not use an overwrite option until you have checked the target directory. Full SGS uses multiprocessing: launch it from a terminal rather than an interactive notebook.

Optional road downloading/elevation preparation needs `requirements-geospatial.txt`, network access, and the original source inventories. Downloaded roads may differ from the study snapshot. A newly downloaded network is not an exact reproduction of the original support-point identifiers.

## Reading the Existing Results

- `results/iowa/5_v2_multivariable_weather_regimes/`: group-count diagnostics, PCA variance, reconstructed centroids, and group statistics.
- `results/iowa/7_v2_elevation_standardized_nowcast_models/`: regression coefficients, scaling, VIF, empirical semivariograms, model comparisons, and cross-validation metrics.
- `results/iowa/8_v2_elevation_nowcast_evaluation/`: protected-event evaluation summaries.
- `results/iowa/parent_events/`: the 70 selected parent identifiers and timing information.
- `results/iowa/sgs/`: all 700 realization statistics, recorded seeds, completion accounting, and an example network figure. The full temperature arrays are omitted.

See [Methods](docs/METHODS.md) and [Results Dictionary](docs/RESULTS_DICTIONARY.md). Exported paths containing `${RESEARCH_ROOT}` document original provenance; they are not live input configurations.

## Reproducibility and Limitations

Exact Iowa reruns require the matching original input snapshots. Seeds alone do not ensure identical results across numerical-library versions and operating systems. Completed RWIS values are treated as fixed conditioning data. Parameter uncertainty is not simulated. SGS uses finite local neighbourhoods and therefore approximates dense conditional Gaussian simulation. Some fitted dependence parameters reach optimization bounds; the comparison tables retain that diagnostic.

Missing-value completion preceded the protected-event exclusions. Those exclusions protect later clustering/model development; they do not make all preprocessing independent of the protected data. These limitations are part of the study design, not omitted release dependencies.

## License and Citation

**License pending author/coauthor approval.** No open-source license is granted by this draft package, and no rights to redistribute the original third-party data or imagery are claimed. Review `LICENSE_NOTICE.md` before publishing.

The final manuscript title, authorship, repository URL, and DOI should be added when confirmed. No DOI or final publication citation has been invented.
