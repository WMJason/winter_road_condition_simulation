#!/usr/bin/env python3
"""Show what distinguishes Step 5 V2's multivariable weather regimes.

Read-only visualization of fitted clusters, member distributions and nearest
examples. No clustering, imputation, scaling, or model selection is repeated.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "step6v2_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
NAMES = ["RST", "Air temperature", "Dewpoint", "Relative humidity", "Wind speed"]
UNITS = ["degrees C", "degrees C", "degrees C", "%", "source units"]
AXES = ["sample", "hour_offset", "station", "variable"]


def resolve(path):
    return (ROOT / path).resolve()


def load_results(config):
    directory = resolve(config["cluster_directory"])
    effective = json.loads((directory / "5_v2_effective_config.json").read_text())
    with np.load(resolve(effective["input_archive"]), allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in ["X", "imputation_mask", "sample_hour_utc", "station_ids", "variables", "hour_offsets", "axis_names"]}
    source_index = pd.read_csv(resolve(effective["input_index"]))
    assigned = pd.read_csv(directory / "5_v2_cluster_assignments_all_samples.csv.gz")
    if source_index.sample_id.duplicated().any() or assigned.sample_id.duplicated().any():
        raise ValueError("Sample IDs must be unique.")
    if len(source_index) != len(arrays["X"]):
        raise ValueError("Step 4 archive/index length mismatch.")
    positions = pd.Index(source_index.sample_id).get_indexer(assigned.sample_id)
    if (positions < 0).any():
        raise ValueError("Assignments contain sample IDs absent from the fixed Step 4 dataset.")
    if not np.array_equal(pd.to_datetime(assigned.sample_hour_utc, utc=True).to_numpy(),
                          pd.to_datetime(arrays["sample_hour_utc"][positions], utc=True).to_numpy()):
        raise ValueError("Assignment timestamps do not match the source archive.")
    if not np.array_equal(positions, np.arange(len(source_index))):
        arrays["X"] = arrays["X"][positions]
        arrays["imputation_mask"] = arrays["imputation_mask"][positions]
    with np.load(directory / "5_v2_centroids.npz", allow_pickle=False) as handle:
        centers = handle["centroids_original_units"]
        for key in ["station_ids", "variables", "hour_offsets"]:
            if not np.array_equal(handle[key], arrays[key]):
                raise ValueError(f"Centroid/source {key} mismatch.")
    if arrays["X"].shape[1:] != (7, 60, 5) or centers.shape[1:] != (7, 60, 5) or arrays["axis_names"].tolist() != AXES:
        raise ValueError("Expected samples and centroids in [hour,station,variable] order.")
    if not np.isfinite(arrays["X"]).all() or not np.isfinite(centers).all():
        raise ValueError("Nonfinite sample or centroid values.")
    if not np.isfinite(assigned.distance_to_assigned_centroid).all() or (assigned.distance_to_assigned_centroid < 0).any():
        raise ValueError("Invalid saved centroid distances.")
    stations = pd.read_csv(resolve(config["station_file"])).set_index("rwis_id").reindex(arrays["station_ids"].astype(str))
    if not np.isfinite(stations[["latitude", "longitude"]].to_numpy()).all():
        raise ValueError("Missing station coordinates.")
    fit = assigned.used_for_kmeans_fit.to_numpy(bool)
    if not fit.any() or assigned.loc[fit, "overlaps_protected_period"].any():
        raise ValueError("Invalid fitting sample flags.")
    if set(assigned.loc[fit, "cluster"]) != set(range(len(centers))):
        raise ValueError("Expected every fitted cluster to have fitting members.")
    return arrays, assigned, centers, stations, directory


def rank_members(assignments, cluster, scope):
    keep = assignments.cluster.eq(cluster)
    if scope == "fitting":
        keep &= assignments.used_for_kmeans_fit
    elif scope == "fit_pool":
        keep &= assignments.eligible_fit_pool
    elif scope != "all_assigned":
        raise ValueError("representative_scope must be fitting, fit_pool, or all_assigned.")
    return assignments.loc[keep].sort_values(["distance_to_assigned_centroid", "sample_id"], kind="stable")


def display_fingerprints(x, assignments, cluster_count):
    """Summarize endpoint level and six-hour change; not clustering coordinates."""
    endpoint = x[:, -1].mean(axis=1)
    change = endpoint-x[:, 0].mean(axis=1)
    descriptors = np.c_[endpoint, change]
    fit = assignments.used_for_kmeans_fit.to_numpy(bool)
    mean, sd = descriptors[fit].mean(axis=0), descriptors[fit].std(axis=0)
    sd = np.where(sd == 0, 1, sd)
    raw = np.array([descriptors[fit & assignments.cluster.eq(c).to_numpy()].mean(axis=0) for c in range(cluster_count)])
    return raw, (raw-mean)/sd, mean, sd


def save(fig, output, name, config):
    fig.savefig(output / f"{name}.png", dpi=int(config["plot_dpi"]), bbox_inches="tight")
    if config["save_pdf"]:
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def overview(x, assigned, centers, offsets, colors, output, config):
    k = len(centers)
    raw, relative, ref_mean, ref_sd = display_fingerprints(x, assigned, k)
    descriptor_names = [f"{name}: issue-time mean" for name in NAMES]+[f"{name}: 6-h change" for name in NAMES]
    records = []
    fit = assigned.used_for_kmeans_fit.to_numpy(bool)
    for c in range(k):
        for j, name in enumerate(descriptor_names):
            records.append({"cluster": c, "descriptor": name, "fitting_member_mean_original_units": raw[c, j],
                "relative_standardized_mean": relative[c, j], "reference_fitting_mean": ref_mean[j], "reference_fitting_sd": ref_sd[j]})
    pd.DataFrame(records).to_csv(output / "6_v2_regime_fingerprints.csv", index=False)
    fig, ax = plt.subplots(figsize=(13, max(4, .65*k+2)), constrained_layout=True)
    limit = config["fingerprint_color_limit"]
    artist = ax.imshow(relative, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    short_names = ["RST", "Air T", "Dewpoint", "Humidity", "Wind"]*2
    short_units = ["deg C", "deg C", "deg C", "%", "source units"]+["deg C", "deg C", "deg C", "pct points", "source units"]
    ax.set_xticks(range(10), [f"{name}\n({unit})" for name, unit in zip(short_names, short_units)], fontsize=8)
    groups = ax.secondary_xaxis("top")
    groups.set_xticks([2,7], ["Issue-time network mean", "Six-hour change in network mean"])
    groups.tick_params(length=0, pad=6, labelsize=10)
    counts = assigned.loc[fit].cluster.value_counts()
    ax.set_yticks(range(k), [f"Cluster {c}  (n={counts[c]})" for c in range(k)])
    for c, j in np.ndindex(raw.shape):
        value = 0.0 if abs(raw[c,j]) < .05 else raw[c,j]
        ax.text(j, c, f"{value:.1f}", ha="center", va="center", fontsize=9,
                color="white" if abs(relative[c,j])>1.5 else "black")
    ax.axvline(4.5, color="white", lw=3)
    ax.set_title("Weather regime characteristics", pad=33)
    fig.colorbar(artist, ax=ax, shrink=.75, extend="both", label="Difference from fitting-sample mean (SD units)")
    save(fig, output, "6_v2_01_regime_fingerprints", config)

    trajectories = x.mean(axis=2)  # Network-mean trajectory for each completed sample.
    fig, axes = plt.subplots(1, 5, figsize=(15, 4.2), constrained_layout=True)
    rows = []
    for c in range(k):
        block = trajectories[fit & assigned.cluster.eq(c).to_numpy()]
        q25, median, q75 = np.quantile(block, [.25, .5, .75], axis=0)
        for j, ax in enumerate(axes):
            ax.plot(offsets, median[:, j], color=colors[c], label=f"Cluster {c}", lw=1.7)
            ax.fill_between(offsets, q25[:, j], q75[:, j], color=colors[c], alpha=.10)
            for h, offset in enumerate(offsets):
                rows.append({"cluster": c, "variable": NAMES[j], "hour_offset": int(offset),
                    "fitting_count": len(block), "q25_network_mean": q25[h,j], "median_network_mean": median[h,j], "q75_network_mean": q75[h,j]})
    for j, ax in enumerate(axes):
        ax.set(title=NAMES[j], xlabel="Hour relative to issue time", ylabel=UNITS[j], xticks=[-6, -3, 0])
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=min(k, 7), frameon=False, fontsize=8)
    fig.suptitle("Recent evolution: fitting-member medians and interquartile ranges")
    save(fig, output, "6_v2_02_temporal_regime_profiles", config)
    pd.DataFrame(rows).to_csv(output / "6_v2_temporal_profile_statistics.csv", index=False)

    fig, axes = plt.subplots(1, 5, figsize=(15, 4), constrained_layout=True)
    endpoint = trajectories[:, -1]
    distributions = []
    for j, ax in enumerate(axes):
        values = [endpoint[fit & assigned.cluster.eq(c).to_numpy(), j] for c in range(k)]
        artists = ax.boxplot(values, patch_artist=True, showfliers=False, widths=.65)
        for c, box in enumerate(artists["boxes"]):
            box.set_facecolor(colors[c]); box.set_alpha(.65)
            q = np.quantile(values[c], [.05, .25, .5, .75, .95])
            distributions.append({"cluster": c, "variable": NAMES[j], "sample_count": len(values[c]),
                "mean_network_endpoint": np.mean(values[c]), "q05": q[0], "q25": q[1], "median": q[2], "q75": q[3], "q95": q[4]})
        ax.set(title=NAMES[j], xlabel="Cluster", ylabel=UNITS[j], xticks=np.arange(1,k+1), xticklabels=np.arange(k))
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Within-regime variation: network-mean issue-time conditions")
    save(fig, output, "6_v2_03_member_distributions", config)
    pd.DataFrame(distributions).to_csv(output / "6_v2_endpoint_distribution_statistics.csv", index=False)


def centroid_atlas(centers, arrays, station_order, limits, output, config):
    k = len(centers)
    fig, axes = plt.subplots(k, 5, figsize=(13, 1.9*k+1), squeeze=False, constrained_layout=True)
    for c, j in np.ndindex(k, 5):
        ax = axes[c,j]
        artist = ax.imshow(centers[c, :, :, j].T[station_order], aspect="auto", cmap="coolwarm" if j<3 else "viridis", vmin=limits[j][0], vmax=limits[j][1])
        ax.set_xticks([0,3,6], [-6,-3,0], fontsize=7)
        if j==0:
            ax.set_yticks([0,20,40,59], arrays["station_ids"][station_order][[0,20,40,59]], fontsize=6)
        else:
            ax.set_yticks([])
        if j==0:
            ax.set_ylabel(f"Cluster {c}\nRWIS station", fontsize=9)
        if c==0:
            ax.set_title(f"{NAMES[j]} ({UNITS[j]})", fontsize=9)
        if c==k-1:
            ax.set_xlabel("Hour relative to issue time", fontsize=8)
        if c==k-1:
            fig.colorbar(artist, ax=axes[:,j].tolist(), orientation="horizontal", shrink=.8, pad=.015, extend="both")
    fig.suptitle("Spatiotemporal centroid atlas: shared scales within each variable")
    save(fig, output, "6_v2_04_spatiotemporal_centroid_atlas", config)


def spatial_maps(centers, stations, output, config):
    k = len(centers)
    levels = centers[:, -1, :, 0]
    changes = levels-centers[:, 0, :, 0]
    lo, hi = levels.min(), levels.max()
    change_limit = max(abs(changes).max(), .1)
    fig, axes = plt.subplots(2, k, figsize=(max(8, 2.55*k), 4.8), squeeze=False, constrained_layout=True)
    for row in range(2):
        for c in range(k):
            ax = axes[row,c]
            values = levels[c] if row==0 else changes[c]
            artist = ax.scatter(stations.longitude, stations.latitude, c=values, s=17,
                cmap="coolwarm", edgecolor=".3", linewidth=.25, vmin=lo if row==0 else -change_limit, vmax=hi if row==0 else change_limit)
            ax.set_aspect(1/np.cos(np.deg2rad(stations.latitude.mean())))
            ax.tick_params(labelsize=7)
            if row==0:
                ax.set_title(f"Cluster {c}")
            if c==0:
                ax.set_ylabel(("Issue-time RST" if row==0 else "Six-hour RST change")+"\nLatitude")
            if row==1:
                ax.set_xlabel("Longitude", fontsize=8)
        fig.colorbar(artist, ax=axes[row].tolist(), shrink=.8, label="degrees C")
    fig.suptitle("Spatial RST patterns at RWIS locations (no spatial interpolation)")
    save(fig, output, "6_v2_05_spatial_rst_regimes", config)


def projection(arrays, assigned, centers, directory, colors, output, config):
    pipeline = joblib.load(directory / "5_v2_kmeans_pipeline.joblib")
    pca = pipeline["pca"]
    if pca is None or pca.n_components_ < 2:
        return "Skipped: the fitted model has no two-component PCA representation."
    fit = assigned.used_for_kmeans_fit.to_numpy(bool)
    z = pipeline["scaler"].transform(arrays["X"][fit].reshape(fit.sum(), -1))
    scores = pca.transform(z)
    predicted = pipeline["kmeans"].predict(scores)
    if not np.array_equal(predicted, assigned.loc[fit, "cluster"].to_numpy()):
        raise ValueError("Saved pipeline does not reproduce fitting assignments; results and inputs may be stale.")
    frame = assigned.loc[fit, ["sample_id", "cluster"]].copy()
    frame["PC1"], frame["PC2"] = scores[:,0], scores[:,1]
    frame.to_csv(output / "6_v2_pca_projection.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for c in range(len(centers)):
        member = predicted==c
        ax.scatter(scores[member,0], scores[member,1], s=10, alpha=.4, color=colors[c], label=f"Cluster {c}", rasterized=True)
        point = pipeline["kmeans"].cluster_centers_[c, :2]
        ax.scatter(*point, marker="X", s=95, color=colors[c], edgecolor="black", linewidth=.7)
    ax.set(xlabel=f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% variance)",
           ylabel=f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% variance)", title="Fitting events in the first two principal components")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, output, "6_v2_06_pca_projection", config)
    return "Two-dimensional projection of the saved PCA; clustering uses every retained component."


def representative_atlas(cluster, centers, arrays, assigned, order, limits, output, config):
    ranked = rank_members(assigned, cluster, config["representative_scope"])
    nearest = ranked.head(config["representatives_per_cluster"])
    positions = nearest.index.to_numpy()
    tiles = [centers[cluster]]+[arrays["X"][i] for i in positions]
    fig, axes = plt.subplots(len(tiles), 5, figsize=(13, 2*len(tiles)+1), squeeze=False, constrained_layout=True)
    for r, tile in enumerate(tiles):
        for j in range(5):
            ax = axes[r,j]
            artist = ax.imshow(tile[:, :, j].T[order], aspect="auto", cmap="coolwarm" if j<3 else "viridis", vmin=limits[j][0], vmax=limits[j][1])
            ax.set_xticks([0,3,6], [-6,-3,0], fontsize=7)
            if j==0:
                ax.set_yticks([0,30,59], arrays["station_ids"][order][[0,30,59]], fontsize=6)
            else:
                ax.set_yticks([])
            if r==0:
                ax.set_title(f"{NAMES[j]} ({UNITS[j]})", fontsize=9)
            if j==0:
                if r==0:
                    label="Reconstructed\ncentroid"
                else:
                    event=nearest.iloc[r-1]
                    label=f"Rank {r} | ID {event.sample_id}\n{pd.to_datetime(event.sample_hour_utc, utc=True).tz_convert('America/Chicago'):%Y-%m-%d %H:%M}\nd={event.distance_to_assigned_centroid:.2f}"
                ax.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center", labelpad=38)
            if r==len(tiles)-1:
                ax.set_xlabel("Hour relative to issue time", fontsize=8)
                fig.colorbar(artist, ax=axes[:,j].tolist(), orientation="horizontal", shrink=.8, pad=.025, extend="both")
    fig.suptitle(f"Cluster {cluster}: centroid and nearest {config['representative_scope']} events")
    save(fig, output, f"6_v2_cluster_{cluster:02d}_nearest_event_atlas", config)
    if config["show_3d"]:
        fig = plt.figure(figsize=(18, 4.5), constrained_layout=True)
        t, s = np.meshgrid(arrays["hour_offsets"], np.arange(1,61))
        for j in range(5):
            ax = fig.add_subplot(1,5,j+1,projection="3d")
            for i in ranked.head(config["three_d_samples_per_cluster"]).index:
                ax.plot_wireframe(t,s,arrays["X"][i,:,:,j].T[order],rstride=4,cstride=1,color="#426b86",alpha=.12,linewidth=.45)
            ax.plot_surface(t,s,centers[cluster,:,:,j].T[order],cmap="coolwarm" if j<3 else "viridis",alpha=1,
                            norm=Normalize(*limits[j]),linewidth=0)
            ax.view_init(elev=25, azim=-135)
            ax.set(title=f"{NAMES[j]}\n({UNITS[j]})", xlabel="Hour", ylabel="Station row", xticks=[-6,-3,0], yticks=[1,30,60])
            ax.tick_params(labelsize=7)
        fig.suptitle(f"Cluster {cluster}: opaque centroid and transparent nearest-event wireframes")
        save(fig, output, f"6_v2_cluster_{cluster:02d}_3d", config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "6_v2_visualization_config.json")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--clusters", nargs="+", type=int, help="Limit supporting representative atlases; overview still shows all clusters.")
    parser.add_argument("--representatives", type=int)
    parser.add_argument("--show-3d", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.representatives is not None:
        config["representatives_per_cluster"] = args.representatives
    if args.show_3d:
        config["show_3d"] = True
    if config["representatives_per_cluster"] < 1:
        raise ValueError("At least one representative is required.")
    plt.rcParams.update({"font.family":"serif", "font.serif":[config["font_family"],"DejaVu Serif"],
                         "font.size":9, "pdf.fonttype":42, "axes.titleweight":"normal"})
    output = args.output_directory or resolve(config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    arrays, assigned, centers, stations, directory = load_results(config)
    colors = [plt.get_cmap("tab10")(i%10) for i in range(len(centers))]
    if config["station_order"] == "north_to_south":
        order = np.lexsort((stations.longitude.to_numpy(), -stations.latitude.to_numpy()))
    elif config["station_order"] == "saved":
        order = np.arange(60)
    else:
        raise ValueError("station_order must be north_to_south or saved.")
    fit = assigned.used_for_kmeans_fit.to_numpy(bool)
    limits = [np.quantile(arrays["X"][fit,:,:,j], config["shared_color_quantiles"]) for j in range(5)]
    limits = [(float(lo), float(hi if hi>lo else lo+1)) for lo,hi in limits]
    overview(arrays["X"], assigned, centers, arrays["hour_offsets"], colors, output, config)
    centroid_atlas(centers, arrays, order, limits, output, config)
    spatial_maps(centers, stations, output, config)
    projection_note = projection(arrays, assigned, centers, directory, colors, output, config)
    selected_clusters = args.clusters if args.clusters is not None else list(range(len(centers)))
    if not set(selected_clusters).issubset(range(len(centers))):
        raise ValueError("Requested cluster does not exist.")
    representative_rows=[]
    for c in selected_clusters:
        print(f"Plotting cluster {c}...", flush=True)
        ranked = rank_members(assigned, c, config["representative_scope"])
        count = max(config["representatives_per_cluster"], config["three_d_samples_per_cluster"] if config["show_3d"] else 0)
        selected = ranked.head(count).copy()
        selected["rank_within_scope"] = np.arange(1,len(selected)+1)
        selected["representative_scope"] = config["representative_scope"]
        representative_rows.append(selected)
        representative_atlas(c, centers, arrays, assigned, order, limits, output, config)
    pd.concat(representative_rows,ignore_index=True).to_csv(output / "6_v2_representative_sample_ranking.csv",index=False)
    assigned.groupby(["cluster","used_for_kmeans_fit","is_cv_test_event"]).size().rename("sample_count").reset_index().to_csv(output / "6_v2_cluster_counts_by_role.csv",index=False)
    pd.DataFrame({"display_row":np.arange(1,61),"rwis_id":arrays["station_ids"][order],
                  "latitude":stations.latitude.to_numpy()[order],"longitude":stations.longitude.to_numpy()[order]}).to_csv(output / "6_v2_station_display_order.csv",index=False)
    metadata={"configuration":config,"cluster_count":len(centers),"fitting_sample_count":int(fit.sum()),
        "projection_note":projection_note,"shared_color_limits":limits,
        "interpretation":"Descriptive displays, not proof of disjoint meteorological populations; within-cluster overlap is retained.",
        "fingerprint_note":"Display-only standardized summary descriptors; numbers are original-unit fitting means, not the 2,100 clustering coordinates.",
        "centroid_note":"Centroids are inverse-PCA/inverse-scaler reconstructions. Temporal summaries and boxes use actual fitting members.",
        "sampling_note":"Nearest examples illustrate cluster cores, not the full member spread; distribution panels show heterogeneity.",
        "spatial_note":"Maps show station values only; heatmap rows/surfaces do not imply interpolation between stations.",
        "units_note":"Wind units are retained as source units; no unsupported unit conversion is applied."}
    (output / "6_v2_visualization_summary.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print("Saved figures and plotting statistics to:",output,flush=True)


if __name__=="__main__":
    main()
