#!/usr/bin/env python3
"""Cluster five-variable (7,60,5) weather histories with CV-event holdout.

Despite the retained 'rst' filename, all five variables are clustered. Original
Step 4/5 outputs are not modified. See 5_V2_MULTIVARIABLE_KMEANS_GUIDE.md.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "step5v2_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, pairwise_distances, silhouette_score)
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
AXES = ["sample", "hour_offset", "station", "variable"]


def load_fixed_inputs(config):
    """Read Step 4 V2 unchanged; exclusions apply only to subsequent fitting."""
    with np.load(config["input_archive"], allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in ["X", "imputation_mask", "station_ids",
                  "variables", "hour_offsets", "axis_names", "sample_hour_utc"]}
    index = pd.read_csv(config["input_index"])
    x = arrays["X"]
    if x.ndim != 4 or x.shape[1:] != (7, 60, 5) or not len(x):
        raise ValueError(f"Expected nonempty [sample,7,60,5], got {x.shape}.")
    if arrays["imputation_mask"].shape != x.shape or arrays["imputation_mask"].dtype != np.bool_ or not np.isfinite(x).all():
        raise ValueError("Invalid imputation mask or nonfinite Step 4 V2 input.")
    if arrays["axis_names"].astype(str).tolist() != AXES or not np.array_equal(arrays["hour_offsets"], np.arange(-6, 1)):
        raise ValueError("Unexpected Step 4 V2 axes or hour offsets.")
    if arrays["variables"].astype(str).tolist() != ["road_surface_temp_c_mean", "air_temp_c_mean", "dewpoint_c_mean", "humidity_mean", "wind_speed_mean"]:
        raise ValueError("Unexpected Step 4 V2 variable order.")
    if len(arrays["station_ids"]) != 60 or len(set(arrays["station_ids"].astype(str))) != 60:
        raise ValueError("Expected 60 unique Step 4 V2 stations.")
    hours = pd.DatetimeIndex(pd.to_datetime(arrays["sample_hour_utc"], utc=True))
    if len(index) != len(x) or index.sample_id.duplicated().any() or not hours.equals(pd.DatetimeIndex(pd.to_datetime(index.sample_hour_utc, utc=True))):
        raise ValueError("Step 4 V2 archive and index disagree.")
    if hours.has_duplicates or not hours.is_monotonic_increasing:
        raise ValueError("Step 4 V2 sample timestamps must be unique and chronological.")
    return arrays, index


def merge_intervals(starts, ends):
    intervals = []
    for start, end in sorted(zip(pd.to_datetime(starts, utc=True), pd.to_datetime(ends, utc=True))):
        if intervals and start <= intervals[-1][1] + pd.Timedelta(hours=1):
            intervals[-1][1] = max(intervals[-1][1], end)
        else:
            intervals.append([start, end])
    return intervals


def intersects(times, intervals, history_hours=0):
    ends = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    starts = ends - pd.Timedelta(hours=history_hours)
    result = np.zeros(len(ends), dtype=bool)
    for left, right in intervals:
        result |= (ends >= left) & (starts <= right)
    return result


def test_events(path, history, buffer):
    labels = pd.read_csv(path)
    required = {"source", "split", "match_hour_local"}
    if not required.issubset(labels):
        raise ValueError(f"Visual matches lack columns: {required-set(labels)}")
    selected = labels[labels.source.eq("rwis_fixed_camera") & labels["split"].str.lower().eq("test")].copy()
    if selected.empty or selected.match_hour_local.isna().any():
        raise ValueError("No valid fixed-camera test events, or missing test event hours.")
    local = pd.to_datetime(selected.match_hour_local, errors="raise")
    utc = local.dt.tz_localize("America/Chicago", ambiguous="raise", nonexistent="raise").dt.tz_convert("UTC")
    if "weather_hour_utc" in selected:
        matched_utc = pd.to_datetime(selected.weather_hour_utc, utc=True, errors="coerce")
        if not (matched_utc.isna() | matched_utc.eq(utc)).all():
            raise ValueError("Visual local times and matched weather UTC times disagree.")
    selected["issue_utc"] = utc
    events = selected.groupby("issue_utc", as_index=False).size().rename(columns={"size": "test_image_records"})
    events["issue_iowa_local"] = events.issue_utc.dt.tz_convert("America/Chicago").astype(str)
    events["history_start_utc"] = events.issue_utc - pd.Timedelta(hours=history)
    events["protected_start_utc"] = events.history_start_utc - pd.Timedelta(hours=buffer)
    events["protected_end_utc"] = events.issue_utc + pd.Timedelta(hours=buffer)
    return events, merge_intervals(events.protected_start_utc, events.protected_end_utc)


def nonoverlap_mask(hours, eligible, minimum_gap):
    selected = np.zeros(len(hours), dtype=bool)
    last = None
    for i in np.flatnonzero(eligible):
        if last is None or hours[i]-last >= pd.Timedelta(hours=minimum_gap):
            selected[i] = True
            last = hours[i]
    return selected


def preparation_fingerprint(config):
    paths = [Path(config[key]) for key in ["input_archive", "input_index", "visual_matches"]]
    signatures = {str(p): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in paths}
    signatures["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    keys = ["history_hours", "test_buffer_hours",
            "maximum_sample_imputed_cell_pct", "maximum_variable_imputed_cell_pct", "minimum_fit_endpoint_gap_hours"]
    signatures["settings"] = {key: config[key] for key in keys}
    return hashlib.sha256(json.dumps(signatures, sort_keys=True).encode()).hexdigest()


def prepare(config, output, force=False):
    history, buffer = int(config["history_hours"]), int(config["test_buffer_hours"])
    if history != 6 or buffer < 24 or config["minimum_fit_endpoint_gap_hours"] < history+1:
        raise ValueError("Require six history hours, at least a 24-hour test buffer, and >=7-hour fitting endpoint spacing.")
    events, intervals = test_events(Path(config["visual_matches"]), history, buffer)
    fingerprint = preparation_fingerprint(config)
    arrays, source_index = load_fixed_inputs(config)
    cache_meta = output / "5_v2_preparation_metadata.json"
    if not force and (output / "5_v2_sample_exclusion_audit.csv.gz").exists() and cache_meta.exists() and json.loads(cache_meta.read_text()).get("fingerprint") == fingerprint:
        index = pd.read_csv(output / "5_v2_sample_exclusion_audit.csv.gz")
        print("Reading fixed Step 4 V2 data; reusing matching exclusion audit.", flush=True)
        return arrays, index, json.loads(cache_meta.read_text())
    print("Reading fixed Step 4 V2 samples without repeating imputation...", flush=True)
    x, mask = arrays["X"], arrays["imputation_mask"]
    utc_hours = pd.DatetimeIndex(pd.to_datetime(arrays["sample_hour_utc"], utc=True))
    local_hours = utc_hours.tz_convert("America/Chicago")
    overlaps = intersects(utc_hours, intervals, history)
    is_test = utc_hours.isin(events.issue_utc)
    imputed_pct = 100*mask.mean(axis=(1, 2, 3))
    per_variable_pct = 100*mask.mean(axis=(1, 2))
    quality = (imputed_pct <= config["maximum_sample_imputed_cell_pct"]+1e-10) & (
        per_variable_pct.max(axis=1) <= config["maximum_variable_imputed_cell_pct"]+1e-10)
    pool = ~overlaps & quality
    fit = nonoverlap_mask(utc_hours, pool, config["minimum_fit_endpoint_gap_hours"])
    if not fit.any():
        raise ValueError("No non-overlapping fitting samples remain.")
    assert not intersects(utc_hours[fit], intervals, history).any()
    reasons = np.select([is_test, overlaps, ~quality, fit],
                        ["cv_test_event", "protected_history_or_buffer", "imputation_quality", "fit_nonoverlapping"],
                        default="fit_pool_overlapping_not_selected")
    index = source_index.rename(columns={"split": "step4_split"}).copy()
    fields = {"sample_hour_utc": utc_hours,
        "sample_hour_iowa_local": local_hours.strftime("%Y-%m-%d %H:%M:%S%z"),
        "sample_year": local_hours.year, "is_cv_test_event": is_test,
        "overlaps_protected_period": overlaps, "included_by_quality_rule": quality,
        "eligible_fit_pool": pool, "used_for_kmeans_fit": fit, "exclusion_or_use_reason": reasons,
        "imputed_cell_pct": imputed_pct}
    for name, values in fields.items():
        index[name] = values
    for i, variable in enumerate(arrays["variables"].astype(str)):
        index[f"{variable}_imputed_cell_pct"] = per_variable_pct[:, i]
    index.to_csv(output / "5_v2_sample_exclusion_audit.csv.gz", index=False)
    events["has_eligible_seven_hour_sample"] = events.issue_utc.isin(utc_hours)
    events.to_csv(output / "5_v2_cv_test_event_audit.csv", index=False)
    pd.DataFrame(intervals, columns=["protected_start_utc", "protected_end_utc"]).to_csv(output / "5_v2_protected_periods.csv", index=False)
    metadata = {"fingerprint": fingerprint, "sample_count": len(x), "tensor_shape": list(x.shape),
        "cv_test_event_count": len(events), "cv_test_event_sample_count": int(is_test.sum()),
        "eligible_fit_pool_count": int(pool.sum()),
        "samples_overlapping_protected_period": int(overlaps.sum()), "nonoverlapping_fit_count": int(fit.sum()),
        "quality_excluded_count": int((~quality).sum()), "history_hours": history, "buffer_hours": buffer,
        "input_mode": "Fixed completed Step 4 V2 dataset; observed and imputed values unchanged.",
        "input_archive": config["input_archive"], "input_index": config["input_index"],
        "imputation_refitted": False,
        "evaluation_scope": "Events held out from clustering/scaling/PCA AFTER fixed Step 4 imputation; not an end-to-end preprocessing holdout or future-year evaluation.",
        "nonoverlap_note": "Fitting histories share no hourly cells; meteorological independence is not guaranteed."}
    cache_meta.write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2), flush=True)
    return arrays, index, metadata


def features(x, fit, config):
    scaler = StandardScaler().fit(x.reshape(len(x), -1)[fit])
    standardized = scaler.transform(x.reshape(len(x), -1)).astype(np.float32)
    pca = None
    transformed = standardized
    if config["use_pca"]:
        pca = PCA(n_components=float(config["pca_explained_variance"]), svd_solver="full", whiten=bool(config["pca_whiten"]))
        pca.fit(standardized[fit])
        transformed = pca.transform(standardized).astype(np.float32)
    return standardized, transformed, scaler, pca


def choose_k(diagnostics, config):
    eligible = diagnostics[diagnostics.passes_minimum_cluster_share].copy()
    if eligible.empty:
        raise ValueError("No candidate passes minimum cluster share; inspect diagnostics before changing the rule.")
    x, y = eligible.k.to_numpy(float), eligible.inertia.to_numpy(float)
    if len(eligible) < 3:
        elbow = int(eligible.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0].k)
    else:
        xn, yn = (x-x.min())/np.ptp(x), (y-y.min())/max(np.ptp(y), np.finfo(float).eps)
        elbow = int(x[np.argmax(np.abs((xn-xn[0])+(yn-yn[0])))])
    if config["selected_k"] is not None:
        selected = int(config["selected_k"])
        if selected not in eligible.k.to_list():
            raise ValueError("Manual K is absent or fails the minimum cluster-share rule.")
    elif config["selection_method"] == "elbow":
        selected = elbow
    elif config["selection_method"] == "silhouette":
        selected = int(eligible.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0].k)
    else:
        raise ValueError("selection_method must be elbow or silhouette.")
    return selected, elbow


def fit_candidates(standardized, transformed, fit, config, output):
    training = transformed[fit]
    count = min(len(training), int(config["max_silhouette_samples"]))
    rng = np.random.default_rng(config["diagnostic_sampling_seed"])
    subset = np.sort(rng.choice(len(training), count, replace=False))
    metric_distances = pairwise_distances(training[subset])
    original_distances = pairwise_distances(standardized[fit][subset])
    np.fill_diagonal(metric_distances, 0)
    np.fill_diagonal(original_distances, 0)
    rows, models = [], {}
    for k in config["k_candidates"]:
        if not 2 <= int(k) < len(training):
            raise ValueError(f"Invalid k={k} for {len(training)} fitting samples.")
        seed_models = [KMeans(n_clusters=int(k), n_init=int(config["n_init"]), max_iter=int(config["max_iter"]),
                             random_state=int(seed), algorithm="lloyd").fit(training)
                       for seed in config["random_seeds"]]
        best = min(seed_models, key=lambda item: item.inertia_)
        models[int(k)] = best
        stability = [adjusted_rand_score(a.labels_, b.labels_) for a, b in itertools.combinations(seed_models, 2)]
        counts = np.bincount(best.labels_, minlength=int(k))
        metric_labels = best.labels_[subset]
        valid_silhouette = 1 < len(np.unique(metric_labels)) < len(subset)
        row = {"k": int(k), "inertia": float(best.inertia_),
               "silhouette": float(silhouette_score(metric_distances, metric_labels, metric="precomputed")) if valid_silhouette else np.nan,
               "silhouette_standardized_original_space": float(silhouette_score(original_distances, metric_labels, metric="precomputed")) if valid_silhouette else np.nan,
               "calinski_harabasz": float(calinski_harabasz_score(training, best.labels_)),
               "davies_bouldin": float(davies_bouldin_score(training, best.labels_)),
               "mean_seed_stability_ari": float(np.mean(stability)) if stability else np.nan,
               "minimum_seed_stability_ari": float(np.min(stability)) if stability else np.nan,
               "minimum_cluster_count": int(counts.min()), "minimum_cluster_share": float(counts.min()/len(training)),
               "maximum_cluster_share": float(counts.max()/len(training)),
               "passes_minimum_cluster_share": bool(counts.min()/len(training) >= config["minimum_cluster_share"]),
               "silhouette_sample_count": count}
        rows.append(row)
        pd.DataFrame(rows).to_csv(output / "5_v2_kmeans_k_diagnostics.csv", index=False)
        print(f"K={k}: silhouette={row['silhouette']:.3f}, minimum share={row['minimum_cluster_share']:.3f}", flush=True)
    return pd.DataFrame(rows), models


def export_statistics(arrays, assignments, centers, output):
    x, variables = arrays["X"], arrays["variables"].astype(str)
    sample_stats = assignments[["sample_id", "sample_hour_utc", "sample_year", "cluster", "used_for_kmeans_fit", "is_cv_test_event"]].copy()
    for j, variable in enumerate(variables):
        values = x[:, :, :, j]
        cells = values.reshape(len(x), -1)
        for name, numbers in [("mean", cells.mean(axis=1)), ("sd", cells.std(axis=1)),
                              ("minimum", cells.min(axis=1)), ("maximum", cells.max(axis=1)),
                              ("endpoint_mean", values[:, -1].mean(axis=1)),
                              ("mean_6h_change", values[:, -1].mean(axis=1)-values[:, 0].mean(axis=1))]:
            sample_stats[f"{variable}__{name}"] = numbers
    sample_stats.to_csv(output / "5_v2_sample_descriptive_statistics.csv.gz", index=False)
    profiles, pooled = [], []
    labels = assignments.cluster.to_numpy()
    fitting = assignments.used_for_kmeans_fit.to_numpy(bool)
    tests = assignments.is_cv_test_event.to_numpy(bool)
    for cluster in range(len(centers)):
        members = x[(labels==cluster) & fitting]
        mean, sd = members.mean(axis=0), members.std(axis=0)
        for h, s, v in np.ndindex(x.shape[1:]):
            profiles.append({"cluster": cluster, "hour_offset": int(arrays["hour_offsets"][h]),
                "rwis_id": str(arrays["station_ids"][s]), "variable": str(variables[v]),
                "reconstructed_centroid": float(centers[cluster, h, s, v]),
                "fitting_member_mean": float(mean[h, s, v]), "fitting_member_sd": float(sd[h, s, v]),
                "fitting_member_count": len(members)})
        for role, group in [("fit_nonoverlapping", fitting), ("cv_test_events", tests), ("all_assigned", np.ones(len(x), bool))]:
            block = x[(labels==cluster) & group]
            if not len(block):
                continue
            for v, variable in enumerate(variables):
                values = block[:, :, :, v].ravel()
                q = np.quantile(values, [.05, .25, .5, .75, .95])
                pooled.append({"cluster": cluster, "scope": role, "variable": variable, "sample_count": len(block),
                    "mean": float(values.mean()), "sd": float(values.std()), "minimum": float(values.min()),
                    "q05": q[0], "q25": q[1], "median": q[2], "q75": q[3], "q95": q[4], "maximum": float(values.max()),
                    "endpoint_mean": float(block[:, -1, :, v].mean()),
                    "mean_6h_change": float((block[:, -1, :, v]-block[:, 0, :, v]).mean())})
    pd.DataFrame(profiles).to_csv(output / "5_v2_centroid_and_member_profiles.csv.gz", index=False)
    pd.DataFrame(pooled).to_csv(output / "5_v2_cluster_descriptive_statistics.csv", index=False)


def plots(centers, arrays, diagnostics, selected_k, pca, output, config):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 10, "pdf.fonttype": 42})
    names = ["Road surface temperature", "Air temperature", "Dewpoint", "Relative humidity", "Wind speed"]
    for cluster, center in enumerate(centers):
        fig, axes = plt.subplots(1, 5, figsize=(16, 7), constrained_layout=True)
        for j, ax in enumerate(axes):
            lo, hi = np.min(centers[:, :, :, j]), np.max(centers[:, :, :, j])
            artist = ax.imshow(center[:, :, j].T, aspect="auto", cmap="coolwarm" if j<3 else "viridis", vmin=lo, vmax=hi)
            ax.set_xticks(range(7), arrays["hour_offsets"])
            ax.set_yticks(range(0, 60, 5), arrays["station_ids"][::5], fontsize=7)
            ax.set(title=names[j], xlabel="Hour relative to issue time")
            fig.colorbar(artist, ax=ax, shrink=.6, label="degrees C" if j<3 else ("%" if j==3 else "source units"))
        axes[0].set_ylabel("RWIS station")
        fig.suptitle(f"Cluster {cluster}: inverse-transformed centroid")
        fig.savefig(output / f"5_v2_cluster_{cluster:02d}_centroid.png", dpi=config["plot_dpi"])
        fig.savefig(output / f"5_v2_cluster_{cluster:02d}_centroid.pdf")
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for ax, metric in zip(axes, ["inertia", "silhouette", "mean_seed_stability_ari"]):
        ax.plot(diagnostics.k, diagnostics[metric], "o-", color="#315c7d")
        ax.axvline(selected_k, ls="--", color="#ac443a")
        ax.set(xlabel="Number of clusters", ylabel=metric.replace("_", " "))
    fig.savefig(output / "5_v2_k_diagnostics.png", dpi=config["plot_dpi"])
    fig.savefig(output / "5_v2_k_diagnostics.pdf")
    plt.close(fig)
    if pca is not None:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.plot(np.arange(1, pca.n_components_+1), np.cumsum(pca.explained_variance_ratio_), color="#315c7d")
        ax.set(xlabel="Retained principal components", ylabel="Cumulative explained variance")
        fig.savefig(output / "5_v2_pca_retained_variance.png", dpi=config["plot_dpi"])
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "5_v2_kmeans_config.json")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--k-values", nargs="+", type=int)
    parser.add_argument("--random-seeds", nargs="+", type=int)
    parser.add_argument("--n-init", type=int)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--no-pca", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-reprepare", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key, value in [("k_candidates", args.k_values), ("random_seeds", args.random_seeds), ("n_init", args.n_init), ("max_iter", args.max_iter)]:
        if value is not None:
            config[key] = value
    if args.no_pca:
        config["use_pca"] = False
    for key in ["visual_matches", "input_archive", "input_index"]:
        config[key] = str((ROOT / config[key]).resolve())
    output = args.output_directory or ROOT / config["output_directory"]
    if (output / "5_v2_kmeans_pipeline.joblib").exists() and not args.overwrite and not args.prepare_only:
        raise FileExistsError("Completed model exists; use --overwrite or a separate output directory.")
    if len(config["random_seeds"]) < 2 or len(set(config["k_candidates"])) != len(config["k_candidates"]):
        raise ValueError("Use at least two seeds and unique K candidates.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "5_v2_effective_config.json").write_text(json.dumps(config, indent=2)+"\n")
    with threadpool_limits(limits=int(config["thread_limit"])):
        arrays, index, preparation = prepare(config, output, args.force_reprepare)
        if args.prepare_only:
            return
        fit = index.used_for_kmeans_fit.to_numpy(bool)
        print(f"Fitting scaler/PCA on {fit.sum():,} non-overlapping samples only...", flush=True)
        standardized, transformed, scaler, pca = features(arrays["X"], fit, config)
        diagnostics, candidates = fit_candidates(standardized, transformed, fit, config, output)
        selected, elbow = choose_k(diagnostics, config)
        model = candidates[selected]
        original_centers = scaler.inverse_transform(pca.inverse_transform(model.cluster_centers_) if pca is not None else model.cluster_centers_)
        order = np.argsort(original_centers.reshape(selected, 7, 60, 5)[:, -1, :, 0].mean(axis=1))
        inverse_order = np.argsort(order)
        model.cluster_centers_ = model.cluster_centers_[order]
        model.labels_ = inverse_order[model.labels_]
        centers = original_centers[order].reshape(selected, 7, 60, 5)
        distances = model.transform(transformed)
        assignments = index.copy()
        assignments["cluster"] = model.predict(transformed)
        assignments["distance_to_assigned_centroid"] = distances[np.arange(len(index)), assignments.cluster]
        assignments["centroid_distance_margin"] = np.partition(distances, 1, axis=1)[:, 1]-assignments.distance_to_assigned_centroid
        assignments.to_csv(output / "5_v2_cluster_assignments_all_samples.csv.gz", index=False)
        diagnostics["selected"] = diagnostics.k.eq(selected)
        diagnostics["estimated_elbow"] = diagnostics.k.eq(elbow)
        diagnostics.to_csv(output / "5_v2_kmeans_k_diagnostics.csv", index=False)
        joblib.dump({"scaler": scaler, "pca": pca, "kmeans": model,
            "station_ids": arrays["station_ids"], "variables": arrays["variables"], "hour_offsets": arrays["hour_offsets"],
            "sample_shape": [7, 60, 5], "axis_names": AXES, "selected_k": selected,
            "config": config, "preparation_fingerprint": preparation["fingerprint"]}, output / "5_v2_kmeans_pipeline.joblib", compress=3)
        np.savez_compressed(output / "5_v2_centroids.npz", centroids_original_units=centers,
                            variables=arrays["variables"], station_ids=arrays["station_ids"], hour_offsets=arrays["hour_offsets"])
        export_statistics(arrays, assignments, centers, output)
        feature_order = pd.DataFrame([{"flattened_feature_index": i, "hour_offset": int(arrays["hour_offsets"][h]),
            "rwis_id": str(arrays["station_ids"][s]), "variable": str(arrays["variables"][v]),
            "fitting_mean": scaler.mean_[i], "fitting_scale": scaler.scale_[i]}
            for i, (h, s, v) in enumerate(np.ndindex(7, 60, 5))])
        feature_order.to_csv(output / "5_v2_feature_scaling_parameters.csv", index=False)
        if pca is not None:
            pd.DataFrame({"component": np.arange(1, pca.n_components_+1), "explained_variance_ratio": pca.explained_variance_ratio_,
                          "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_)}).to_csv(output / "5_v2_pca_variance.csv", index=False)
        plots(centers, arrays, diagnostics, selected, pca, output, config)
        summary = {**preparation, "selected_k": selected, "estimated_elbow_k": elbow,
            "pca_used": pca is not None, "retained_components": int(transformed.shape[1]),
            "pca_retained_variance": float(pca.explained_variance_ratio_.sum()) if pca is not None else None,
            "fit_sample_count_by_year": index.loc[fit].groupby("sample_year").size().to_dict(),
            "heldout_events_used_to_fit_scaler_pca_or_clusters": False,
            "earlier_step4_imputation_respects_this_holdout": False,
            "cluster_labels": "Ordered by increasing reconstructed issue-time mean RST; all five variables define clustering.",
            "warning": "CV image test splits share weather events with CV training images. This protects the numerical clustering pipeline, not the original classifier's event independence."}
        (output / "5_v2_kmeans_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
