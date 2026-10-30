# Methods and Study Settings

## 1. Hourly observations and quality control

Step 2a joins `rwis_observation` and `rwis_station` in a supplied SQLite archive and aggregates repeated measurements into station-hour means. The RST target is the mean of the available pavement-sensor means. The five grouping variables are RST, air temperature, dewpoint, relative humidity, and wind speed. Temperatures are represented in degrees Celsius, humidity in percent, and archived wind speed in knots; wind speed is not converted by these scripts.

Step 2b uses broad variable limits plus median/MAD spatial and temporal consistency checks. Values outside broad limits are set to missing. Additional values are rejected only when both spatial and temporal criteria are satisfied; individual values, not entire station-hour rows, are removed. A separate dewpoint/air-temperature consistency rule is implemented. This is not IQR filtering. The source code records the thresholds, and the release retains the rejection summary.

## 2. Station selection and fixed completed histories

Step 3 selects consistently reporting stations according to the saved annual coverage rule. The Iowa workflow expects 60 stations and seven-hour windows. Before completion, every hour in a candidate history must have complete five-variable vectors at at least 48 stations.

Step 4 fills missing values using same-hour inverse-squared-distance weighting, then recent station departures from expected values, then station/month/hour expected values. The spatial stage uses at most five neighbours within 200 km and needs at least three. Recent departures extend at most six hours. Expected values are developed from 2016--2017. Imputation source flags are retained. Step 4 V2 keeps all five variables in `[sample, hour_offset, station, variable]` order; it does not standardize them.

The scripts contain the actual calendar/indexing implementation, including UTC indexing for the completed table. The later regression calendar features explicitly use Iowa local time. Imputation is performed before the later protected-event exclusions and is not refitted after them.

## 3. Grouping histories

Step 5 excludes histories overlapping fixed-camera test periods, accounting for the six-hour history and a 24-hour buffer. It also applies maximum completion percentages and a seven-hour minimum gap between fitting endpoints. The Iowa archive has 14,379 candidate histories, of which 2,043 are used to fit grouping transformations.

The 2,100 entries in each history are standardized using the fitting histories only. PCA is also fitted on that subset, retains at least 95% of variance, and is not whitened. The recorded Iowa fit retained 45 components (approximately 95.03%). K-means considers K=2 through 20, multiple random initializations, and minimum group-size requirements. The saved elbow rule selects seven groups. These are representative multivariable histories, not seven independently established storm types.

The protected-event timing table is a minimal reconstruction from the saved audit. Duplicate rows retain the original image-record multiplicities, but contain no image paths, URLs, imagery, or category annotations.

## 4. Group-specific STRK

RST is the numerical target. The other four grouping variables are not regression covariates. Candidate regressors include projected horizontal coordinates in kilometres, their quadratic/interaction terms, elevation in metres, and cyclic local-calendar terms. Regression covariates are standardized within the fitting data. The code exports both fitted-design coefficients and coefficients transformed back to original units.

Bidirectional selection uses conditional coefficient tests at the configured 0.05 level, with event-clustered standard errors in the current Iowa configuration. Fold-level transformations are fitted within training folds. Regression residuals are used to fit separable spatiotemporal exponential, Gaussian, and spherical covariance/semivariogram candidates. Candidate comparison uses event-held-out evaluation and exclusion of the target station's entire history from residual kriging. Tables retain both fit and validation diagnostics; the best empirical variogram fit is not necessarily the best predictor.

RST nowcasts add the regression trend to a residual simple-kriging correction. Step 8 evaluates saved models on the protected event subset, with observed-target/neighbor restrictions as recorded in its configuration. This differs from treating the completed station data as fixed conditioning values in SGS.

## 5. Parent events and SGS

Step 9 selects ten observed histories nearest each fitted cluster centre in the retained grouping representation, giving 70 parents. This emphasizes typical histories rather than rare extremes or historical event frequencies. The selected dates, identifiers, spacing summaries, and model parameters are supplied.

Step 10 uses the matching group regression and residual covariance. It subtracts the regression trend from the seven-hour completed RWIS RST histories, sequentially simulates residuals at network support points, and adds the location/time-specific regression trend back. The current conditioning policy includes all 420 completed station-hour RST values per parent, not only originally measured values.

The default produces an issue-time map, not a seven-hour road-map simulation. Each parent has ten realizations with the same conditioning observations and fitted parameters but different random innovations. Finite local simple-kriging neighbourhoods make this a path-dependent approximation to dense conditional Gaussian simulation. The code reuses path geometry across realizations for efficiency while drawing independent innovations. Exactly coincident conditioning locations/times are honored explicitly. Road supports merely near a station do not have to equal the station observation.

The Iowa run uses 122,329 support points, 70 successful parents, and 700 realizations. Support cells are no longer than 500 m. Regression and covariance parameter uncertainty, completed-input uncertainty, and alternative covariance structures are not propagated by those ten draws.

## Adaptation versus exact reproduction

The code retains Iowa-specific station counts, historical years, calendar timezone, geographical checks, and filenames. A different region requires deliberate changes; it is not selected by changing one filename. Exact Iowa reconstruction requires the corresponding data and network snapshots. The public synthetic example shows the main numerical operations, not the full study's fitting/selection design.
