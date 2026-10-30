#!/usr/bin/env python3
"""Evaluate saved Step 7 V2 models on protected Step 5 V2 events, nowcasts only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def load_step7():
    spec = importlib.util.spec_from_file_location("step7v2_evaluation_helpers",
        ROOT / "7_v2_fit_cluster_specific_regression_variograms_with_elevation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def absolute(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boolean(series):
    values = series.astype(str).str.lower().map({"true": True, "false": False})
    if values.isna().any():
        raise ValueError(f"Invalid boolean column: {series.name}")
    return values


def select_events(assignments, scope):
    data = assignments.copy()
    for column in ["used_for_kmeans_fit", "overlaps_protected_period", "is_cv_test_event"]:
        data[column] = boolean(data[column])
    if scope == "protected":
        selected = data[data.overlaps_protected_period].copy()
    elif scope == "cv_test":
        selected = data[data.is_cv_test_event].copy()
    else:
        raise ValueError("event_scope must be protected or cv_test")
    if selected.used_for_kmeans_fit.any():
        raise ValueError("Evaluation events intersect Step 5 fitting events.")
    if selected.sample_id.duplicated().any():
        raise ValueError("Duplicate evaluation events.")
    return selected.sort_values(["sample_hour_utc", "sample_id"]).reset_index(drop=True)


def evaluate_event(step7, arrays, event, stations, package, helpers):
    window = step7.event_frame(arrays, event, stations, package["projection"], helpers)
    trend = step7.predict_trend(package["regression_model"], package["selected_features"], window)
    residual = window[step7.TARGET].to_numpy() - trend
    neighbor_config = {"cv_observed_neighbors_only": True,
                       "neighbor_count": int(package["neighbor_count"])}
    targets = window.hour_offset.eq(0) & ~window.rst_is_imputed
    rows = []
    for position in np.flatnonzero(targets):
        target = window.iloc[position]
        geometry = step7.kriging_geometry(window, position, neighbor_config)
        correction, variance = step7.residual_nowcast(residual, geometry, package["variogram"],
                                                    package["config"]["kriging_ridge"])
        rows.append({"sample_id": target.sample_id, "cluster": int(event.iloc[0].cluster),
            "issue_time_utc": target.issue_time_utc, "issue_time_iowa_local":
                pd.Timestamp(target.issue_time_utc).tz_localize("UTC").tz_convert("America/Chicago").isoformat(),
            "target_hour_utc": target.hour_utc, "horizon_h": 0, "rwis_id": target.rwis_id,
            "year": int(pd.Timestamp(target.hour_local).year),
            "is_cv_test_event": bool(event.iloc[0].is_cv_test_event),
            "actual_c": float(target[step7.TARGET]), "target_is_imputed": False,
            "regression_prediction_c": float(trend[position]),
            "strk_prediction_c": float(trend[position]+correction),
            "neighbor_count": len(geometry["indices"]),
            "trend_only_fallback": len(geometry["indices"]) == 0,
            "residual_kriging_variance_c2": variance,
            "model_family": package["variogram"]["model_family"]})
    return rows


def metric_tables(step7, predictions, output):
    specifications = {"overall": [], "by_cluster": ["cluster"],
        "by_event": ["sample_id", "cluster", "issue_time_utc"],
        "by_station": ["rwis_id"], "by_year": ["year"]}
    for name, keys in specifications.items():
        rows = []
        for scope, subset in [("all_evaluated_events", predictions),
                              ("exact_cv_test_hours", predictions[predictions.is_cv_test_event])]:
            if subset.empty:
                continue
            groups = subset.groupby(keys, sort=True) if keys else [((), subset)]
            for values, group in groups:
                if not isinstance(values, tuple):
                    values = (values,)
                for method, column in [("regression", "regression_prediction_c"),
                                       ("strk", "strk_prediction_c")]:
                    rows.append({**dict(zip(keys, values)), "evaluation_scope": scope,
                        "method": method, "horizon_h": 0,
                        "event_count": group.sample_id.nunique(),
                        **step7.metrics(group.actual_c, group[column])})
        pd.DataFrame(rows).to_csv(output / f"8_v2_metrics_{name}.csv", index=False)


def plot_results(step7, data, output, dpi):
    plt = step7.plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 11, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    lo = min(data.actual_c.min(), data.strk_prediction_c.min())
    hi = max(data.actual_c.max(), data.strk_prediction_c.max())
    density = axes[0].hexbin(data.actual_c, data.strk_prediction_c, gridsize=45, mincnt=1, cmap="viridis")
    axes[0].plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--")
    axes[0].set(xlabel="Observed RST (°C)", ylabel="STRK nowcast (°C)")
    fig.colorbar(density, ax=axes[0], label="Station-hour count")
    clusters = sorted(data.cluster.unique())
    positions = np.arange(len(clusters))
    for method, column, shift, color in [("Regression", "regression_prediction_c", -.18, "#777777"),
                                          ("STRK", "strk_prediction_c", .18, "#16756a")]:
        errors = [step7.metrics(g.actual_c, g[column])["rmse_c"]
                  for cluster in clusters for g in [data[data.cluster.eq(cluster)]]]
        axes[1].bar(positions+shift, errors, width=.34, color=color, label=method)
    axes[1].set(xticks=positions, xticklabels=clusters, xlabel="Assigned cluster", ylabel="Nowcast RMSE (°C)")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"8_v2_nowcast_performance.{extension}", dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "8_v2_evaluation_config.json"))
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--output-directory")
    parser.add_argument("--event-scope", choices=["protected", "cv_test"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.event_scope:
        config["event_scope"] = args.event_scope
    if args.max_events is not None and args.max_events < 1:
        parser.error("--max-events must be positive")
    step7 = load_step7()
    fit_config = json.loads(absolute(config["step7_config"]).read_text())
    for key in ["cluster_directory", "station_file", "output_directory"]:
        fit_config[key] = str(absolute(fit_config[key]))
    arrays, fitting_events, stations, _ = step7.load_inputs(fit_config)
    cluster_directory = Path(fit_config["cluster_directory"])
    assignments = pd.read_csv(cluster_directory / "5_v2_cluster_assignments_all_samples.csv.gz")
    events = select_events(assignments, config["event_scope"])
    source = json.loads((cluster_directory / "5_v2_effective_config.json").read_text())
    source_index = pd.read_csv(source["input_index"])
    events["source_position"] = pd.Index(source_index.sample_id).get_indexer(events.sample_id)
    if (events.source_position < 0).any():
        raise ValueError("An evaluation event is missing from the source archive.")
    source_times = pd.to_datetime(arrays["sample_hour_utc"][events.source_position], utc=True)
    if not np.array_equal(source_times.to_numpy(), pd.to_datetime(events.sample_hour_utc, utc=True).to_numpy()):
        raise ValueError("Evaluation/source timestamps disagree.")
    if args.max_events:
        events = events.head(args.max_events).copy()
    if events.empty:
        raise ValueError("No eligible evaluation events.")
    output = absolute(args.output_directory or config["output_directory"])
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is not empty: {output}; use a new folder or --overwrite")
    packages, provenance = {}, []
    for cluster in sorted(events.cluster.unique()):
        directory = Path(fit_config["output_directory"]) / f"cluster_{int(cluster):02d}"
        path = directory / "7_v2_cluster_strk_model.joblib"
        package = joblib.load(path)
        if (not package.get("nowcasting_only") or package.get("horizons") != [0]
                or package.get("covariance_form") != "separable_product"
                or package["target"] != step7.TARGET or int(package["cluster"]) != int(cluster)):
            raise ValueError(f"Incompatible Step 7 V2 model: {path}")
        audit = pd.read_csv(directory / "7_v2_fitting_event_audit.csv")
        if not set(audit.sample_id).issubset(set(fitting_events.sample_id)):
            raise ValueError("Model audit contains events outside Step 5 fitting subset.")
        if set(audit.sample_id).intersection(set(events.sample_id)):
            raise ValueError("A model was trained on evaluation events.")
        for term in package["selected_features"]:
            if term not in step7.FEATURES:
                raise ValueError(f"Unexpected regression covariate: {term}")
        packages[int(cluster)] = package
        provenance.append({"cluster": int(cluster), "model_path": str(path),
                           "sha256": file_hash(path), "run_signature": package["run_signature"]})
    output.mkdir(parents=True, exist_ok=True)
    events.to_csv(output / "8_v2_evaluated_event_audit.csv", index=False)
    cv_audit = pd.read_csv(cluster_directory / "5_v2_cv_test_event_audit.csv")
    evaluated_times = pd.to_datetime(events.sample_hour_utc, utc=True)
    cv_audit["evaluated_in_this_run"] = pd.to_datetime(cv_audit.issue_utc, utc=True).isin(evaluated_times)
    cv_audit.to_csv(output / "8_v2_cv_test_event_coverage.csv", index=False)
    rows = []
    helpers = step7.original_helpers()
    with step7.threadpool_limits(limits=int(config["thread_limit"])):
        for position in range(len(events)):
            event = events.iloc[[position]]
            print(f"[{position+1}/{len(events)}] {event.iloc[0].sample_id}, cluster {event.iloc[0].cluster}: nowcast", flush=True)
            rows.extend(evaluate_event(step7, arrays, event, stations,
                                       packages[int(event.iloc[0].cluster)], helpers))
    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise ValueError("No originally observed RST targets are available.")
    predictions.to_csv(output / "8_v2_heldout_nowcast_predictions.csv.gz", index=False)
    metric_tables(step7, predictions, output)
    if config["make_plots"]:
        plot_results(step7, predictions, output, int(config["plot_dpi"]))
    summary = {"config": config, "max_events": args.max_events, "nowcasting_only": True,
        "horizons": [0], "selected_event_count": len(events),
        "events_with_observed_targets": int(predictions.sample_id.nunique()),
        "evaluated_station_hours": len(predictions), "model_provenance": provenance,
        "script_sha256": file_hash(Path(__file__)),
        "overall_strk_metrics": step7.metrics(predictions.actual_c, predictions.strk_prediction_c),
        "evaluation_design": "Fixed fitted models; target station whole history excluded; observed targets/neighbors only.",
        "limitations": ["Conditional on Step 5 saved assignments, which use completed five-variable histories including target RST.",
            "Step 4 imputation precedes the holdout split; this is not an end-to-end preprocessing holdout.",
            "Protected hourly histories overlap; station-hour metrics are descriptive, not independent replications.",
            "Residual kriging variance excludes regression and model-selection uncertainty."]}
    (output / "8_v2_evaluation_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(f"Finished. Results: {output}", flush=True)


if __name__ == "__main__":
    main()
