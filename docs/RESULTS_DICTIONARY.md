# Compact Results and Parameter Files

`results/iowa/` preserves reported diagnostics from the saved research run. It is not the output of the synthetic example. CSV files retain numerical precision for later plotting; PDF/PNG figures are illustrations of those saved results.

## Preparation and grouping

- QC summaries count rejected values and affected rows. Missing values are not automatically deleted station-hours.
- Step 4 variable summaries describe completed inputs. Completion uncertainty is not estimated by these tables.
- Step 5 K diagnostics record inertia, silhouette, Calinski--Harabasz, Davies--Bouldin, seed stability, and group shares. The selected K follows the saved elbow/size rule, not maximization of every indicator.
- PCA variance and feature-scaling tables document the fitted transformations. C-order flattening maps `[hour_offset, station, variable]` into 2,100 coordinates.
- Step 6 tables describe group fingerprints, issue-hour distributions, and temporal profiles.

## Regression and STRK

- `7_regression_coefficients.csv`: coefficients in the fitted standardized design, with uncertainty/test statistics.
- `7_regression_coefficients_original_units.csv`: corresponding coefficients in the original feature units. Do not multiply original-unit coefficients by standardized features.
- `7_v2_regression_scaling.json`: training means/scales.
- `7_regression_vif.csv`: variance inflation diagnostics for the selected design.
- `7_v2_variogram_comparison.csv`: the competing covariance families and fitting/validation diagnostics. Spatial and temporal ranges use the source function's scale convention, not an assumed practical-range conversion.
- `7_v2_empirical_spatiotemporal_variogram.csv`: binned residual semivariances, mean distances, temporal lags, and pair counts.
- CV and evaluation metrics include MAE, RMSE, bias, and R-squared where defined. See each CSV's column headers for counts and grouping. The saved Step 7 and Step 8 holdout designs are different; do not pool them as independent repetitions of one validation.

## Parent events and SGS

- Parent-event tables record cluster, parent/sample IDs, dates, selection distances, and temporal spacing. Their ten nearest-centroid events per group are not frequency-weighted random samples of all Iowa weather.
- `sgs/10_sgs_manifest.csv`: one row per parent, including cluster, realization/support counts, conditioning count, runtime, and archive size. Its HDF5 paths are provenance references to omitted large files.
- `sgs/all_realization_statistics.csv`: consolidated statistics from each parent's original `10_realization_statistics.csv`. Read headers for field/residual means, spread, and diagnostics; these are map summaries, not station-validated model errors.
- `sgs/all_realization_seeds.csv`: original path/draw seeds, with parent identifiers added when absent. Seeds need the same model, observations, support order, and numerical environment to be reproducible.
- `sgs/10_sgs_run_summary.json`: completion and configuration accounting. Recorded source paths are sanitized provenance, not runnable configurations.

## Portable model parameters

`models/iowa_strk_parameters.json` contains original-unit regression coefficients, selected features, feature scaling, coordinate origin, and selected covariance parameters for each group. It does not contain the observed station histories needed to condition a nowcast or SGS realization. It is not a replacement for Step 7's fitted joblib objects in the full workflow.

`models/iowa_kmeans_parameters.npz` contains scaler means/scales, PCA components/centering/variance, and K-means centres. The associated schema JSON records station/variable/hour order. The archive can be opened with `numpy.load(..., allow_pickle=False)`. Training samples are not included. These parameters enable numerical inspection without loading executable pickle objects.

## Statistical interpretation

Seven groups do not imply seven physically distinct storm mechanisms. Multiple SGS realizations share a parent and fitted model. Neither the 700 realizations nor map support points should be treated as 700 independent observed storms or 122,329 independent validation stations. New predictions should be validated against genuinely withheld observations.
