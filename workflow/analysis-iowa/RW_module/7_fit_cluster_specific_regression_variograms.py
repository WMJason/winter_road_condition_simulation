#!/usr/bin/env python3
"""Fit one linear-regression STRK model for each RST weather regime.

Development data are the 2016--2017 observations.  For every cluster this
script:

1. selects a parsimonious OLS trend model by bidirectional stepwise testing;
2. requires every retained non-intercept coefficient to have p < 0.05;
3. constructs an empirical spatiotemporal semivariogram from OLS residuals;
4. fits exponential, Gaussian, and spherical candidate structures; and
5. evaluates nowcasts and 1--6 h forecasts by leave-one-station-out STRK; and
6. selects the structure with the lowest pooled STRK RMSE over all horizons.

The leave-one-station-out exercise holds the fitted trend fixed and excludes
the target station from residual kriging.  The independent 2018 evaluation is
performed by 8_evaluate_cluster_specific_strk_2018.py.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import cm
from scipy.optimize import least_squares
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


ROOT = Path(__file__).resolve().parent
OBSERVATION_FILE = ROOT / "2b_quality_controlled_outputs" / (
    "2b_rwis_station_hourly_modeling_table_qc_2016-01-01_to_2018-12-31.csv"
)
ASSIGNMENT_FILE = ROOT / "5_rst_kmeans_weather_regimes" / (
    "5_rst_cluster_assignments_all_years.csv"
)
STATION_FILE = ROOT / "3_most_overlapped_rwis_outputs" / (
    "3_rwis_stations_reporting_all_weather_variables_every_year.csv"
)
OUTPUT_DIR = ROOT / "7_cluster_specific_strk_models"

TARGET = "road_surface_temp_c_mean"
UTC_COLUMN = "hour_utc"
LOCAL_COLUMN = "hour_local"
STATION_COLUMN = "rwis_id"
DEVELOPMENT_YEARS = (2016, 2017)
SIGNIFICANCE_LEVEL = 0.05

# Candidate trend terms. Coordinates are local equirectangular kilometres.
CANDIDATE_FEATURES = [
    "x_km", "y_km", "x_km_sq", "y_km_sq", "x_y_interaction",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "dayofyear_sin", "dayofyear_cos",
]

# Empirical semivariogram and CV controls. Set either maximum to None to use
# every eligible pair/target; the defaults keep development practical.
MAX_PAIR_SAMPLE = 300_000
# Issue-time event sampling for candidate-semivariogram cross-validation. The
# percentage is applied first and then bounded by the minimum and maximum.
# Use None for the percentage to request all events before applying the bounds;
# use None for the maximum to remove the upper bound.
CV_EVENT_SAMPLE_PERCENT: float | None = 10 #10%
MIN_CV_EVENTS_PER_CLUSTER = 10
MAX_CV_EVENTS_PER_CLUSTER: int | None = 1000
HORIZONS = tuple(range(0, 7))
MAX_SPATIAL_LAG_KM = 500.0
MAX_TEMPORAL_LAG_HOURS = 6
SPATIAL_BIN_KM = 25.0
MIN_PAIRS_PER_BIN = 30
LOOKBACK_HOURS = 6
NEIGHBOR_COUNT = 80
KRIGING_RIDGE = 1e-8
INTERVAL_Z = 1.96
RANDOM_STATE = 2027

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
})


def add_features(frame: pd.DataFrame, projection: dict) -> pd.DataFrame:
    frame = frame.copy()
    frame[UTC_COLUMN] = pd.to_datetime(frame[UTC_COLUMN], errors="coerce")
    local = pd.to_datetime(frame[LOCAL_COLUMN], errors="coerce")
    frame[LOCAL_COLUMN] = local
    frame["local_year"] = local.dt.year
    frame["local_hour"] = local.dt.hour
    frame["local_month"] = local.dt.month
    frame["local_dayofyear"] = local.dt.dayofyear
    frame["hour_sin"] = np.sin(2 * np.pi * frame["local_hour"] / 24.0)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["local_hour"] / 24.0)
    frame["month_sin"] = np.sin(2 * np.pi * frame["local_month"] / 12.0)
    frame["month_cos"] = np.cos(2 * np.pi * frame["local_month"] / 12.0)
    frame["dayofyear_sin"] = np.sin(2 * np.pi * frame["local_dayofyear"] / 366.0)
    frame["dayofyear_cos"] = np.cos(2 * np.pi * frame["local_dayofyear"] / 366.0)
    r = 6371.0088
    lat0 = float(projection["origin_latitude"])
    lon0 = float(projection["origin_longitude"])
    frame["x_km"] = r * np.deg2rad(frame["longitude"] - lon0) * math.cos(math.radians(lat0))
    frame["y_km"] = r * np.deg2rad(frame["latitude"] - lat0)
    frame["x_km_sq"] = frame["x_km"] ** 2
    frame["y_km_sq"] = frame["y_km"] ** 2
    frame["x_y_interaction"] = frame["x_km"] * frame["y_km"]
    return frame


def fit_ols(data: pd.DataFrame, features: list[str]):
    x = sm.add_constant(data[features].astype(float), has_constant="add")
    return sm.OLS(data[TARGET].astype(float), x, missing="drop").fit()


def stepwise_ols(data: pd.DataFrame) -> tuple[object, list[str], pd.DataFrame]:
    """Bidirectional selection using conditional two-sided coefficient tests."""
    selected: list[str] = []
    remaining = CANDIDATE_FEATURES.copy()
    history: list[dict] = []
    iteration = 0

    while True:
        iteration += 1
        changed = False
        forward = []
        for feature in remaining:
            model = fit_ols(data, selected + [feature])
            forward.append((float(model.pvalues.get(feature, np.inf)), feature))
        if forward:
            best_p, best_feature = min(forward)
            if np.isfinite(best_p) and best_p < SIGNIFICANCE_LEVEL:
                selected.append(best_feature)
                remaining.remove(best_feature)
                history.append({"iteration": iteration, "action": "add", "feature": best_feature, "p_value": best_p})
                changed = True

        # Backward elimination is repeated because removal can alter other tests.
        while selected:
            model = fit_ols(data, selected)
            pvalues = model.pvalues.drop("const", errors="ignore")
            worst_feature = str(pvalues.idxmax())
            worst_p = float(pvalues.max())
            if np.isfinite(worst_p) and worst_p >= SIGNIFICANCE_LEVEL:
                selected.remove(worst_feature)
                if worst_feature not in remaining:
                    remaining.append(worst_feature)
                history.append({"iteration": iteration, "action": "remove", "feature": worst_feature, "p_value": worst_p})
                changed = True
            else:
                break
        if not changed:
            break

    final_model = fit_ols(data, selected)
    retained_p = final_model.pvalues.drop("const", errors="ignore")
    if len(retained_p) and not bool((retained_p < SIGNIFICANCE_LEVEL).all()):
        raise RuntimeError("Stepwise selection retained a term with p >= 0.05")
    return final_model, selected, pd.DataFrame(history)


def regression_exports(model, features: list[str], data: pd.DataFrame, cluster_dir: Path):
    ci = model.conf_int(alpha=SIGNIFICANCE_LEVEL)
    coefficient_table = pd.DataFrame({
        "term": model.params.index,
        "coefficient": model.params.values,
        "standard_error": model.bse.values,
        "t_statistic": model.tvalues.values,
        "p_value": model.pvalues.values,
        "ci_95_lower": ci.iloc[:, 0].values,
        "ci_95_upper": ci.iloc[:, 1].values,
    })
    coefficient_table.to_csv(cluster_dir / "7_regression_coefficients.csv", index=False)

    if features:
        x = sm.add_constant(data[features].astype(float), has_constant="add")
        vif = pd.DataFrame({
            "term": x.columns,
            "vif": [variance_inflation_factor(x.to_numpy(), i) for i in range(x.shape[1])],
        })
    else:
        vif = pd.DataFrame({"term": ["const"], "vif": [np.nan]})
    vif.to_csv(cluster_dir / "7_regression_vif.csv", index=False)

    predictions = np.asarray(model.predict(sm.add_constant(data[features], has_constant="add")))
    actual = data[TARGET].to_numpy()
    metrics = {
        "n": int(len(data)), "feature_count": int(len(features)),
        "r_squared": float(model.rsquared), "adjusted_r_squared": float(model.rsquared_adj),
        "aic": float(model.aic), "bic": float(model.bic),
        "mae_c": float(mean_absolute_error(actual, predictions)),
        "rmse_c": float(mean_squared_error(actual, predictions) ** 0.5),
        "bias_c": float(np.mean(predictions - actual)),
        "residual_standard_error_c": float(np.sqrt(model.ssr / model.df_resid)),
        "condition_number": float(model.condition_number),
        "all_nonintercept_terms_p_lt_0_05": bool(
            (model.pvalues.drop("const", errors="ignore") < SIGNIFICANCE_LEVEL).all()
        ),
    }
    (cluster_dir / "7_regression_summary.txt").write_text(model.summary().as_text(), encoding="utf-8")
    return metrics


def sample_pairs(data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Draw a lag-stratified random pair sample without enumerating all pairs."""
    columns = [STATION_COLUMN, "x_km", "y_km", "residual"]
    by_time = {
        pd.Timestamp(time): part[columns].to_numpy()
        for time, part in data.groupby(UTC_COLUMN, sort=False)
    }
    rows: list[tuple[float, int, float]] = []
    lag_count = MAX_TEMPORAL_LAG_HOURS + 1
    per_lag_budget = (
        50_000 if MAX_PAIR_SAMPLE is None
        else max(1, int(math.ceil(MAX_PAIR_SAMPLE / lag_count)))
    )

    for lag in range(lag_count):
        eligible_times = np.array([
            time for time in by_time
            if time + pd.Timedelta(hours=lag) in by_time
        ], dtype="datetime64[ns]")
        if not len(eligible_times):
            continue
        accepted = 0
        attempts = 0
        max_attempts = per_lag_budget * 30
        while accepted < per_lag_budget and attempts < max_attempts:
            attempts += 1
            time = pd.Timestamp(rng.choice(eligible_times))
            left = by_time[time]
            right = by_time[time + pd.Timedelta(hours=lag)]
            if not len(left) or not len(right):
                continue
            i = int(rng.integers(len(left)))
            j = int(rng.integers(len(right)))
            if lag == 0 and i == j:
                continue
            if left[i, 0] == right[j, 0]:
                continue
            distance = float(np.hypot(
                float(left[i, 1]) - float(right[j, 1]),
                float(left[i, 2]) - float(right[j, 2]),
            ))
            if distance > MAX_SPATIAL_LAG_KM:
                continue
            semivariance = 0.5 * (
                float(left[i, 3]) - float(right[j, 3])
            ) ** 2
            rows.append((distance, lag, semivariance))
            accepted += 1
        print(
            f"  sampled {accepted:,} semivariogram pairs at temporal lag {lag} h",
            flush=True,
        )

    if not rows:
        raise RuntimeError("No eligible residual pairs for empirical semivariogram")
    pairs = pd.DataFrame(
        rows, columns=["distance_km", "lag_hours", "semivariance"]
    )
    if MAX_PAIR_SAMPLE is not None and len(pairs) > MAX_PAIR_SAMPLE:
        pairs = pairs.sample(MAX_PAIR_SAMPLE, random_state=RANDOM_STATE)
    return pairs.reset_index(drop=True)


def empirical_variogram(pairs: pd.DataFrame) -> pd.DataFrame:
    pairs = pairs.copy()
    pairs["space_bin"] = np.floor(pairs["distance_km"] / SPATIAL_BIN_KM).astype(int)
    empirical = pairs.groupby(["space_bin", "lag_hours"], as_index=False).agg(
        distance_km=("distance_km", "mean"),
        semivariance=("semivariance", "median"),
        pair_count=("semivariance", "size"),
    )
    return empirical[empirical["pair_count"] >= MIN_PAIRS_PER_BIN].copy()


def structure(family: str, d: np.ndarray, t: np.ndarray, params: np.ndarray) -> np.ndarray:
    nugget, partial_sill, space_range, time_range = params
    if family == "exponential":
        correlation = np.exp(-d / space_range - t / time_range)
    elif family == "gaussian":
        correlation = np.exp(-(d / space_range) ** 2 - (t / time_range) ** 2)
    elif family == "spherical":
        radius = d / space_range + t / time_range
        correlation = np.where(radius < 1.0, 1.0 - 1.5 * radius + 0.5 * radius ** 3, 0.0)
    else:
        raise ValueError(family)
    return nugget + partial_sill * (1.0 - correlation)


def fit_variogram_candidates(empirical: pd.DataFrame, residual_variance: float) -> list[dict]:
    d = empirical["distance_km"].to_numpy()
    t = empirical["lag_hours"].to_numpy()
    y = empirical["semivariance"].to_numpy()
    weights = np.sqrt(empirical["pair_count"].to_numpy())
    initial = np.array([max(residual_variance * 0.05, 1e-4), max(residual_variance * 0.95, 1e-4), 150.0, 3.0])
    lower = np.array([0.0, 1e-8, 1.0, 0.10])
    upper = np.array([residual_variance * 5 + 1.0, residual_variance * 10 + 1.0, 2000.0, 48.0])
    candidates = []
    for family in ("exponential", "gaussian", "spherical"):
        result = least_squares(
            lambda p: (structure(family, d, t, p) - y) * weights,
            initial, bounds=(lower, upper), max_nfev=20_000,
        )
        fitted = structure(family, d, t, result.x)
        candidates.append({
            "model_family": family, "nugget": float(result.x[0]),
            "partial_sill": float(result.x[1]), "space_range_km": float(result.x[2]),
            "time_range_hours": float(result.x[3]),
            "weighted_fit_rmse": float(np.sqrt(np.average((fitted - y) ** 2, weights=weights))),
        })
    return candidates


def fitted_variogram_surface(
    space_lag: np.ndarray, time_lag: np.ndarray, params: dict
) -> np.ndarray:
    parameter_vector = np.array([
        params["nugget"], params["partial_sill"],
        params["space_range_km"], params["time_range_hours"],
    ])
    return structure(
        params["model_family"], space_lag, time_lag, parameter_vector
    )


def plot_empirical_variogram(
    empirical: pd.DataFrame, cluster: int, output_path: Path
) -> None:
    plot_data = empirical.copy()
    plot_data["space_lag_center_km"] = (
        plot_data["space_bin"] + 0.5
    ) * SPATIAL_BIN_KM
    pivot = plot_data.pivot_table(
        index="lag_hours", columns="space_lag_center_km",
        values="semivariance",
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    im = ax.imshow(
        pivot, origin="lower", aspect="auto", cmap="viridis",
        extent=[
            float(pivot.columns.min()), float(pivot.columns.max()),
            float(pivot.index.min()), float(pivot.index.max()),
        ],
    )
    ax.set_xlabel("Spatial lag (km)")
    ax.set_ylabel("Temporal lag (hours)")
    ax.set_title(f"Cluster {cluster}: empirical spatiotemporal semivariogram")
    fig.colorbar(im, ax=ax, label=r"Semivariance ($^\circ$C$^2$)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_selected_variogram_3d(
    empirical: pd.DataFrame, params: dict, cluster: int, output_path: Path
) -> None:
    space_grid = np.linspace(0.0, float(empirical["distance_km"].max()), 50)
    time_grid = np.linspace(0.0, float(empirical["lag_hours"].max()), 35)
    space_mesh, time_mesh = np.meshgrid(space_grid, time_grid)
    gamma_mesh = fitted_variogram_surface(space_mesh, time_mesh, params)

    fig = plt.figure(figsize=(9.2, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        space_mesh, time_mesh, gamma_mesh, cmap=cm.viridis,
        alpha=0.68, linewidth=0, antialiased=True,
    )
    ax.scatter(
        empirical["distance_km"], empirical["lag_hours"],
        empirical["semivariance"], c="black",
        s=np.clip(empirical["pair_count"] / 35.0, 9, 36),
        alpha=0.60, label="Empirical bins",
    )
    fig.colorbar(
        surface, ax=ax, shrink=0.64, pad=0.09,
        label=r"Fitted semivariance ($^\circ$C$^2$)",
    )
    ax.set_xlabel("Spatial lag (km)")
    ax.set_ylabel("Temporal lag (hours)")
    ax.set_zlabel(r"Semivariance ($^\circ$C$^2$)")
    ax.set_title(
        f"Cluster {cluster}: selected {params['model_family'].capitalize()} "
        "spatiotemporal semivariogram"
    )
    ax.view_init(elev=25, azim=-135)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_selected_variogram_contours(
    empirical: pd.DataFrame, params: dict, cluster: int, output_path: Path
) -> None:
    space_grid = np.linspace(0.0, float(empirical["distance_km"].max()), 140)
    time_grid = np.linspace(0.0, float(empirical["lag_hours"].max()), 90)
    space_mesh, time_mesh = np.meshgrid(space_grid, time_grid)
    gamma_mesh = fitted_variogram_surface(space_mesh, time_mesh, params)

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    contour = ax.contourf(
        space_mesh, time_mesh, gamma_mesh, levels=20, cmap="viridis"
    )
    points = ax.scatter(
        empirical["distance_km"], empirical["lag_hours"],
        c=empirical["semivariance"], cmap="magma",
        s=np.clip(empirical["pair_count"] / 40.0, 9, 38),
        edgecolors="white", linewidths=0.25, alpha=0.85,
    )
    ax.set_xlabel("Spatial lag (km)")
    ax.set_ylabel("Temporal lag (hours)")
    ax.set_title(
        f"Cluster {cluster}: selected {params['model_family'].capitalize()} "
        "semivariogram"
    )
    fig.colorbar(contour, ax=ax, label=r"Fitted semivariance ($^\circ$C$^2$)")
    # The empirical points use a separate color scale; this compact key makes
    # that distinction explicit without adding a second full colorbar.
    points.set_label("Empirical bins (color = empirical semivariance)")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def covariance(distance_km: np.ndarray, lag_hours: np.ndarray, params: dict) -> np.ndarray:
    family = params["model_family"]
    if family == "exponential":
        corr = np.exp(-distance_km / params["space_range_km"] - lag_hours / params["time_range_hours"])
    elif family == "gaussian":
        corr = np.exp(-(distance_km / params["space_range_km"]) ** 2 - (lag_hours / params["time_range_hours"]) ** 2)
    else:
        radius = distance_km / params["space_range_km"] + lag_hours / params["time_range_hours"]
        corr = np.where(radius < 1.0, 1.0 - 1.5 * radius + 0.5 * radius ** 3, 0.0)
    return params["partial_sill"] * corr + params["nugget"] * ((distance_km == 0) & (lag_hours == 0))


def local_kriging(target: pd.Series, available: pd.DataFrame, params: dict):
    if available.empty:
        return 0.0, params["nugget"] + params["partial_sill"], 0
    spatial = np.hypot(available["x_km"] - target["x_km"], available["y_km"] - target["y_km"])
    temporal = np.abs((available[UTC_COLUMN] - target[UTC_COLUMN]).dt.total_seconds() / 3600.0)
    order = np.lexsort((temporal.to_numpy(), spatial.to_numpy()))[:NEIGHBOR_COUNT]
    selected = available.iloc[order]
    xy = selected[["x_km", "y_km"]].to_numpy()
    hours = selected[UTC_COLUMN].astype("int64").to_numpy() / 3.6e12
    dmat = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    tmat = np.abs(hours[:, None] - hours[None, :])
    matrix = covariance(dmat, tmat, params) + np.eye(len(selected)) * KRIGING_RIDGE
    dvec = np.hypot(selected["x_km"] - target["x_km"], selected["y_km"] - target["y_km"]).to_numpy()
    tvec = np.abs((selected[UTC_COLUMN] - target[UTC_COLUMN]).dt.total_seconds().to_numpy() / 3600.0)
    vector = covariance(dvec, tvec, params)
    try:
        weights = np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(matrix, vector, rcond=None)[0]
    prediction = float(weights @ selected["residual"].to_numpy())
    variance = float(max(params["nugget"] + params["partial_sill"] - vector @ weights, 1e-9))
    return prediction, variance, len(selected)


def metrics(actual: np.ndarray, predicted: np.ndarray, lower=None, upper=None) -> dict:
    answer = {
        "n": int(len(actual)), "mae_c": float(mean_absolute_error(actual, predicted)),
        "rmse_c": float(mean_squared_error(actual, predicted) ** 0.5),
        "r_squared": float(r2_score(actual, predicted)) if len(actual) > 1 else np.nan,
        "bias_c": float(np.mean(predicted - actual)),
    }
    if lower is not None:
        answer["interval_95_coverage"] = float(np.mean((actual >= lower) & (actual <= upper)))
        answer["interval_95_mean_width_c"] = float(np.mean(upper - lower))
    return answer


def station_cv(
    observations: pd.DataFrame,
    issue_times: pd.DatetimeIndex,
    regression,
    features: list[str],
    params: dict,
    cluster: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Evaluate an issue-time regime over h=0,...,6 with target station held out."""
    issue_times = pd.DatetimeIndex(sorted(pd.unique(issue_times)))
    observations_by_time = {
        pd.Timestamp(time): part
        for time, part in observations.groupby(UTC_COLUMN, sort=False)
    }
    records = []
    for event_number, issue_time in enumerate(issue_times, start=1):
        if event_number % 10 == 0:
            print(
                f"  {params['model_family']}: event {event_number:,}/{len(issue_times):,}",
                flush=True,
            )
        window_parts = [
            observations_by_time[hour]
            for hour in (
                issue_time - pd.to_timedelta(
                    np.arange(LOOKBACK_HOURS, -1, -1), unit="h"
                )
            )
            if hour in observations_by_time
        ]
        available = (
            pd.concat(window_parts, ignore_index=True)
            if window_parts else pd.DataFrame()
        )
        if available.empty:
            continue
        available["trend_prediction"] = regression.predict(
            sm.add_constant(available[features], has_constant="add")
        )
        available["residual"] = available[TARGET] - available["trend_prediction"]

        for horizon in HORIZONS:
            target_time = issue_time + pd.Timedelta(hours=horizon)
            targets = observations_by_time.get(target_time)
            if targets is None:
                continue
            for _, target in targets.iterrows():
                neighbors_data = available[
                    available[STATION_COLUMN].ne(target[STATION_COLUMN])
                ]
                residual_prediction, variance, neighbors = local_kriging(
                    target, neighbors_data, params
                )
                target_design = sm.add_constant(
                    pd.DataFrame([target])[features], has_constant="add"
                )
                trend = float(regression.predict(target_design).iloc[0])
                prediction = trend + residual_prediction
                se = math.sqrt(variance)
                records.append({
                    "cluster": cluster, "model_family": params["model_family"],
                    "issue_time_utc": issue_time, "target_hour_utc": target_time,
                    "horizon_h": horizon, "rwis_id": target[STATION_COLUMN],
                    "actual_c": target[TARGET], "regression_prediction_c": trend,
                    "strk_prediction_c": prediction,
                    "lower_95_c": prediction - INTERVAL_Z * se,
                    "upper_95_c": prediction + INTERVAL_Z * se,
                    "neighbor_count": neighbors,
                })
    predictions = pd.DataFrame(records)
    if predictions.empty:
        raise RuntimeError(f"No CV predictions were generated for cluster {cluster}")
    actual = predictions["actual_c"].to_numpy()
    result = metrics(
        actual, predictions["strk_prediction_c"].to_numpy(),
        predictions["lower_95_c"].to_numpy(), predictions["upper_95_c"].to_numpy(),
    )
    regression_metric = metrics(
        actual, predictions["regression_prediction_c"].to_numpy()
    )
    result.update({
        f"regression_{key}": value
        for key, value in regression_metric.items() if key != "n"
    })
    result["mean_neighbor_count"] = float(predictions["neighbor_count"].mean())
    horizon_rows = []
    for horizon, group in predictions.groupby("horizon_h"):
        horizon_metric = metrics(
            group["actual_c"].to_numpy(), group["strk_prediction_c"].to_numpy(),
            group["lower_95_c"].to_numpy(), group["upper_95_c"].to_numpy(),
        )
        regression_metric = metrics(
            group["actual_c"].to_numpy(),
            group["regression_prediction_c"].to_numpy(),
        )
        horizon_metric.update({
            f"regression_{key}": value
            for key, value in regression_metric.items() if key != "n"
        })
        horizon_rows.append({
            "cluster": cluster, "model_family": params["model_family"],
            "horizon_h": int(horizon), **horizon_metric,
        })
    return result, pd.DataFrame(horizon_rows), predictions


def select_cv_issue_times(
    issue_times: pd.Series, cluster: int
) -> tuple[pd.DatetimeIndex, dict]:
    """Select one reproducible event subset shared by all variogram candidates."""
    available = pd.DatetimeIndex(sorted(pd.unique(pd.to_datetime(issue_times))))
    available_count = len(available)
    if not available_count:
        raise RuntimeError(f"Cluster {cluster} has no issue-time samples")

    if CV_EVENT_SAMPLE_PERCENT is None:
        requested = available_count
    else:
        if not 0 < CV_EVENT_SAMPLE_PERCENT <= 100:
            raise ValueError("CV_EVENT_SAMPLE_PERCENT must be in (0, 100]")
        requested = int(math.ceil(
            available_count * CV_EVENT_SAMPLE_PERCENT / 100.0
        ))

    if MIN_CV_EVENTS_PER_CLUSTER < 1:
        raise ValueError("MIN_CV_EVENTS_PER_CLUSTER must be at least 1")
    requested = max(requested, MIN_CV_EVENTS_PER_CLUSTER)
    if MAX_CV_EVENTS_PER_CLUSTER is not None:
        if MAX_CV_EVENTS_PER_CLUSTER < MIN_CV_EVENTS_PER_CLUSTER:
            raise ValueError(
                "MAX_CV_EVENTS_PER_CLUSTER cannot be smaller than "
                "MIN_CV_EVENTS_PER_CLUSTER"
            )
        requested = min(requested, MAX_CV_EVENTS_PER_CLUSTER)
    selected_count = min(requested, available_count)

    if selected_count == available_count:
        selected = available
    else:
        rng = np.random.default_rng(RANDOM_STATE + cluster)
        selected = pd.DatetimeIndex(np.sort(
            rng.choice(available.to_numpy(), selected_count, replace=False)
        ))

    details = {
        "cluster": cluster,
        "available_event_count": available_count,
        "sample_percent_requested": CV_EVENT_SAMPLE_PERCENT,
        "minimum_event_count": MIN_CV_EVENTS_PER_CLUSTER,
        "maximum_event_count": MAX_CV_EVENTS_PER_CLUSTER,
        "selected_event_count": selected_count,
        "selected_event_pct": 100.0 * selected_count / available_count,
        "random_state": RANDOM_STATE + cluster,
    }
    return selected, details


def load_development_data():
    observations = pd.read_csv(OBSERVATION_FILE)
    stations = pd.read_csv(STATION_FILE)
    selected = set(stations[STATION_COLUMN].astype(str))
    observations = observations[observations[STATION_COLUMN].astype(str).isin(selected)].copy()
    projection = {
        "origin_latitude": float(stations["latitude"].median()),
        "origin_longitude": float(stations["longitude"].median()),
        "units": "kilometres",
        "method": "local_equirectangular",
    }
    observations = add_features(observations, projection)
    assignments = pd.read_csv(ASSIGNMENT_FILE)
    assignments["sample_hour_utc"] = pd.to_datetime(assignments["sample_hour_utc"], errors="coerce")
    assignments = assignments[assignments["sample_year"].isin(DEVELOPMENT_YEARS)].copy()
    required = [TARGET, "latitude", "longitude"] + CANDIDATE_FEATURES
    observations = observations[
        observations["local_year"].isin(DEVELOPMENT_YEARS)
    ].dropna(subset=required).copy()
    assignments = assignments.dropna(subset=["sample_hour_utc", "cluster"])
    assignments["cluster"] = assignments["cluster"].astype(int)
    return observations, assignments, projection


def cluster_window_data(
    observations: pd.DataFrame, issue_times: pd.Series
) -> pd.DataFrame:
    """Unique station-hours appearing in a cluster's seven-hour samples."""
    included_hours = set()
    for issue_time in pd.to_datetime(issue_times):
        included_hours.update(
            issue_time - pd.to_timedelta(np.arange(LOOKBACK_HOURS + 1), unit="h")
        )
    return observations[observations[UTC_COLUMN].isin(included_hours)].copy()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data, assignments, projection = load_development_data()
    rng = np.random.default_rng(RANDOM_STATE)
    model_rows, cv_rows, all_predictions, sampling_rows = [], [], [], []

    print(f"Development observations: {len(data):,}")
    print(f"Clusters: {sorted(assignments['cluster'].unique())}")
    for cluster, cluster_assignments in assignments.groupby("cluster", sort=True):
        cluster = int(cluster)
        cluster_dir = OUTPUT_DIR / f"cluster_{cluster:02d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        issue_times = cluster_assignments["sample_hour_utc"]
        cv_issue_times, sampling_details = select_cv_issue_times(
            issue_times, cluster
        )
        sampling_rows.append(sampling_details)
        pd.DataFrame({
            "cluster": cluster,
            "selected_issue_time_utc": cv_issue_times,
        }).to_csv(
            cluster_dir / "7_selected_cv_issue_times.csv", index=False
        )
        cluster_data = cluster_window_data(data, issue_times)
        cluster_data = cluster_data.sort_values([UTC_COLUMN, STATION_COLUMN]).copy()
        print(
            f"\nCluster {cluster}: {len(issue_times):,} samples and "
            f"{len(cluster_data):,} unique seven-hour-window observations; "
            f"{len(cv_issue_times):,} samples selected for cross-validation",
            flush=True,
        )

        regression, features, history = stepwise_ols(cluster_data)
        history.to_csv(cluster_dir / "7_stepwise_selection_history.csv", index=False)
        regression_metrics = regression_exports(regression, features, cluster_data, cluster_dir)
        cluster_data["trend_prediction"] = regression.predict(
            sm.add_constant(cluster_data[features], has_constant="add")
        )
        cluster_data["residual"] = cluster_data[TARGET] - cluster_data["trend_prediction"]
        print(f"  Selected trend terms: {features or ['intercept only']}")

        pairs = sample_pairs(cluster_data, rng)
        empirical = empirical_variogram(pairs)
        empirical.to_csv(cluster_dir / "7_empirical_spatiotemporal_variogram.csv", index=False)
        candidates = fit_variogram_candidates(empirical, float(cluster_data["residual"].var()))

        candidate_rows = []
        candidate_predictions = []
        candidate_horizon_rows = []
        for candidate in candidates:
            print(f"  Evaluating {candidate['model_family']} variogram", flush=True)
            cv_metric, horizon_metrics, predictions = station_cv(
                data, cv_issue_times, regression, features,
                candidate, cluster,
            )
            candidate_rows.append({"cluster": cluster, **candidate, **cv_metric})
            candidate_horizon_rows.append(horizon_metrics)
            candidate_predictions.append(predictions)
        comparison = pd.DataFrame(candidate_rows).sort_values(["rmse_c", "mae_c"])
        comparison.to_csv(cluster_dir / "7_variogram_loso_comparison.csv", index=False)
        candidate_horizon_comparison = pd.concat(
            candidate_horizon_rows, ignore_index=True
        )
        candidate_horizon_comparison.to_csv(
            cluster_dir / "7_variogram_loso_comparison_by_horizon.csv", index=False
        )
        best_family = str(comparison.iloc[0]["model_family"])
        selected_variogram = next(c for c in candidates if c["model_family"] == best_family)
        best_predictions = next(p for p in candidate_predictions if p["model_family"].iloc[0] == best_family)
        best_predictions.to_csv(cluster_dir / "7_selected_variogram_loso_predictions.csv", index=False)
        selected_horizon_metrics = candidate_horizon_comparison[
            candidate_horizon_comparison["model_family"].eq(best_family)
        ].copy()
        selected_horizon_metrics.insert(
            0, "selection_rule", "lowest pooled h=0,...,6 STRK RMSE"
        )
        selected_horizon_metrics.to_csv(
            cluster_dir / "7_selected_variogram_loso_metrics_by_horizon.csv",
            index=False,
        )
        plot_empirical_variogram(
            empirical, cluster,
            cluster_dir / "7_empirical_spatiotemporal_variogram.png",
        )
        plot_selected_variogram_3d(
            empirical, selected_variogram, cluster,
            cluster_dir / "7_selected_spatiotemporal_variogram_3d.png",
        )
        plot_selected_variogram_contours(
            empirical, selected_variogram, cluster,
            cluster_dir / "7_selected_spatiotemporal_variogram_contours.png",
        )
        all_predictions.append(best_predictions)

        package = {
            "cluster": cluster, "target": TARGET, "development_years": DEVELOPMENT_YEARS,
            "projection": projection, "regression_model": regression,
            "selected_features": features, "candidate_features": CANDIDATE_FEATURES,
            "significance_level": SIGNIFICANCE_LEVEL, "variogram": selected_variogram,
            "lookback_hours": LOOKBACK_HOURS, "neighbor_count": NEIGHBOR_COUNT,
            "cv_design": (
                "issue-time event evaluation at h=0,...,6 with fixed trend and "
                "target station excluded from residual kriging"
            ),
            "variogram_selection_rule": "lowest pooled h=0,...,6 STRK RMSE",
        }
        joblib.dump(package, cluster_dir / "7_cluster_strk_model.joblib")
        selected_cv = comparison.iloc[0].to_dict()
        model_rows.append({"cluster": cluster, "selected_features": ";".join(features), **regression_metrics, **selected_variogram})
        cv_rows.append(selected_cv)

    pd.DataFrame(model_rows).to_csv(OUTPUT_DIR / "7_cluster_regression_and_variogram_summary.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(OUTPUT_DIR / "7_cluster_loso_strk_metrics.csv", index=False)
    pd.DataFrame(sampling_rows).to_csv(
        OUTPUT_DIR / "7_cv_event_sampling_summary.csv", index=False
    )
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(OUTPUT_DIR / "7_all_selected_variogram_loso_predictions.csv", index=False)
    horizon_rows = []
    for horizon, group in predictions.groupby("horizon_h"):
        strk_metric = metrics(
            group["actual_c"].to_numpy(), group["strk_prediction_c"].to_numpy(),
            group["lower_95_c"].to_numpy(), group["upper_95_c"].to_numpy(),
        )
        regression_metric = metrics(
            group["actual_c"].to_numpy(),
            group["regression_prediction_c"].to_numpy(),
        )
        horizon_rows.append({
            "horizon_h": int(horizon), **strk_metric,
            **{
                f"regression_{key}": value
                for key, value in regression_metric.items() if key != "n"
            },
        })
    pd.DataFrame(horizon_rows).to_csv(
        OUTPUT_DIR / "7_selected_variograms_loso_metrics_by_horizon.csv",
        index=False,
    )
    cluster_horizon_rows = []
    for (cluster, horizon), group in predictions.groupby(
        ["cluster", "horizon_h"]
    ):
        strk_metric = metrics(
            group["actual_c"].to_numpy(), group["strk_prediction_c"].to_numpy(),
            group["lower_95_c"].to_numpy(), group["upper_95_c"].to_numpy(),
        )
        regression_metric = metrics(
            group["actual_c"].to_numpy(),
            group["regression_prediction_c"].to_numpy(),
        )
        cluster_horizon_rows.append({
            "cluster": int(cluster), "horizon_h": int(horizon), **strk_metric,
            **{
                f"regression_{key}": value
                for key, value in regression_metric.items() if key != "n"
            },
        })
    pd.DataFrame(cluster_horizon_rows).to_csv(
        OUTPUT_DIR
        / "7_selected_variograms_loso_metrics_by_cluster_and_horizon.csv",
        index=False,
    )
    overall_rows = []
    for name, column in (("regression_only", "regression_prediction_c"), ("strk", "strk_prediction_c")):
        values = metrics(predictions["actual_c"].to_numpy(), predictions[column].to_numpy())
        values.update({"model": name})
        if name == "strk":
            interval = metrics(
                predictions["actual_c"].to_numpy(), predictions[column].to_numpy(),
                predictions["lower_95_c"].to_numpy(), predictions["upper_95_c"].to_numpy(),
            )
            values.update({k: v for k, v in interval.items() if k.startswith("interval_")})
        overall_rows.append(values)
    pd.DataFrame(overall_rows).to_csv(OUTPUT_DIR / "7_overall_development_loso_metrics.csv", index=False)
    (OUTPUT_DIR / "7_run_configuration.json").write_text(json.dumps({
        "development_years": DEVELOPMENT_YEARS, "significance_level": SIGNIFICANCE_LEVEL,
        "candidate_features": CANDIDATE_FEATURES, "max_pair_sample": MAX_PAIR_SAMPLE,
        "cv_event_sample_percent": CV_EVENT_SAMPLE_PERCENT,
        "min_cv_events_per_cluster": MIN_CV_EVENTS_PER_CLUSTER,
        "max_cv_events_per_cluster": MAX_CV_EVENTS_PER_CLUSTER,
        "evaluation_horizons": list(HORIZONS),
        "max_spatial_lag_km": MAX_SPATIAL_LAG_KM,
        "max_temporal_lag_hours": MAX_TEMPORAL_LAG_HOURS,
        "neighbor_count": NEIGHBOR_COUNT, "lookback_hours": LOOKBACK_HOURS,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved cluster-specific regression STRK models to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
