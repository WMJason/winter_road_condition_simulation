# Release Verification

The following checks passed locally on macOS with Python 3.10.7 and the included tested numerical dependency versions:

1. STRK and SGS covariance functions agree for exponential, Gaussian, and spherical families.
2. SGS exactly honors coincident hard conditions, repeats with fixed seeds, and produces variation across independent draw seeds.
3. Regression coefficients transformed back to original units preserve predictions.
4. The minimal protected-event metadata reconstructs all nine recorded test hours and 607 matched image-record timings without distributing images or condition labels.
5. Exported numerical PCA/K-means parameters reproduce the ordered group identities of the saved reconstructed centroids.

The synthetic demonstration ran through hourly aggregation, missing-value completion, grouping, numerical model fitting, an example holdout correction, and conditional simulation. It produced nine synthetic realizations. The maximum conditioning discrepancy was approximately 8.9e-16 degrees C.

The ZIP was then extracted into a new temporary directory, and both the five tests and the demonstration were rerun from that independent copy. The synthetic completion report matched the included reference report exactly in that local environment. This checks relocation of the public example; it does not establish numerical identity across different operating systems.

The compact release audit checks Python syntax, machine-specific path removal, excluded large-input formats, export checksums, parameter-array shapes, exact integer representation of seeds, and the 70-parent/700-realization completion counts.

The full Iowa workflow was not rerun when packaging. The optional road-download/DEM services were not contacted. Windows/Linux execution is configured in GitHub Actions but was not directly tested during this local preparation. The code license remains pending author/coauthor approval.
