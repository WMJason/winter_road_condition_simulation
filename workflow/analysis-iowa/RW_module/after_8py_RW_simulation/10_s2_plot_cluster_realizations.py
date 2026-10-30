#!/usr/bin/env python3
"""Plot saved SGS fields for the first Step 9 event in each cluster; never simulate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "step10_sgs_mpl"))
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def plot_first_events(config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    events = pd.read_csv(Path(config["events_directory"]) / "9_selected_weather_events.csv")
    # Step 9 manifest order defines 'first', independently of worker completion order.
    first = events.drop_duplicates("cluster", keep="first").sort_values("cluster")
    root = Path(config["output_directory"])
    catalog = root / "10_road_support_catalog.h5"
    if not catalog.exists():
        print("Realization panels: no saved support catalog yet", flush=True)
        return []
    with h5py.File(catalog, "r") as store:
        longitude = store["longitude"][:]
        latitude = store["latitude"][:]
    directory = root / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 9, "pdf.fonttype": 42, "axes.linewidth": 0.6})
    rows = []
    for event in first.to_dict("records"):
        event_id = str(event["parent_event_id"])
        source = root / "events" / event_id / "10_sgs_realizations.h5"
        if not source.exists():
            print(f"Realization panels: {event_id} not simulated yet; skipped", flush=True)
            continue
        with h5py.File(source, "r") as store:
            hours = store["hour_offsets"][:]
            plane = int(np.flatnonzero(hours == 0)[0]) if 0 in hours else len(hours) - 1
            samples = store["residual_c"][:, plane, :] + store["trend_c"][plane, :]
            issue = str(store.attrs.get("issue_time_utc", event.get("sample_hour_utc", "")))
        if samples.shape[1] != len(longitude) or not np.isfinite(samples).all():
            raise ValueError(f"{event_id}: inconsistent support count or nonfinite temperatures")
        observed = pd.read_csv(source.parent / "10_observed_rst_conditioning.csv.gz")
        observed = observed[observed.hour_offset.eq(hours[plane])]
        vmin, vmax = float(samples.min()), float(samples.max())
        if vmax <= vmin:
            vmin, vmax = vmin - 0.5, vmax + 0.5
        norm = Normalize(vmin=vmin, vmax=vmax)
        aspect = 1 / np.cos(np.deg2rad(float(np.mean(latitude))))
        xpad = max(float(np.ptp(longitude)) * 0.025, 0.01)
        ypad = max(float(np.ptp(latitude)) * 0.025, 0.01)
        for start in range(0, len(samples), 10):
            count = min(10, len(samples) - start)
            columns = min(5, count)
            nrows = (count + columns - 1) // columns
            fig, axes = plt.subplots(nrows, columns, figsize=(3.0 * columns, 2.7 * nrows + 0.8),
                                     sharex=True, sharey=True, squeeze=False, layout="constrained")
            try:
                for offset, ax in enumerate(axes.flat):
                    if offset >= count:
                        ax.set_visible(False)
                        continue
                    index = start + offset
                    picture = ax.scatter(longitude, latitude, c=samples[index], cmap="coolwarm", norm=norm,
                                         s=0.7, linewidths=0, rasterized=True)
                    measured = observed[~observed.is_imputed]
                    filled = observed[observed.is_imputed]
                    ax.scatter(measured.longitude, measured.latitude, marker="^", s=12,
                               facecolors="white", edgecolors="black", linewidths=0.5, zorder=3)
                    ax.scatter(filled.longitude, filled.latitude, marker="o", s=12,
                               facecolors="white", edgecolors="black", linewidths=0.5, zorder=3)
                    ax.set_title(f"Realization {index + 1}", fontsize=10, pad=5)
                    ax.set_aspect(aspect)
                    ax.set_xlim(float(longitude.min()) - xpad, float(longitude.max()) + xpad)
                    ax.set_ylim(float(latitude.min()) - ypad, float(latitude.max()) + ypad)
                    ax.xaxis.set_major_locator(MaxNLocator(3))
                    ax.yaxis.set_major_locator(MaxNLocator(3))
                    ax.tick_params(labelsize=8, length=2)
                    ax.spines[["top", "right"]].set_visible(False)
                active = list(axes.flat)[:count]
                fig.colorbar(picture, ax=active, label="Road surface temperature (\N{DEGREE SIGN}C)",
                             shrink=0.83, pad=0.015, fraction=0.025)
                axes[(count - 1) // columns, ((count - 1) % columns) // 2].set_xlabel("Longitude (degrees)", fontsize=10)
                fig.supylabel("Latitude (degrees)", fontsize=10)
                fig.suptitle(f"Cluster {int(event['cluster'])} | {event_id}\n"
                             f"Issue time: {issue} | Hour offset: {int(hours[plane])}", fontsize=12)
                handle = Line2D([], [], marker="^", color="none", markerfacecolor="white",
                                markeredgecolor="black", markersize=5, label="Measured RWIS")
                filled_handle = Line2D([], [], marker="o", color="none", markerfacecolor="white",
                                       markeredgecolor="black", markersize=5, label="Imputed/completed RWIS")
                # Reserve a narrow strip below the map panels for a shared legend.
                fig.legend(handles=[handle, filled_handle], loc="outside lower center", ncol=2, frameon=False, fontsize=9)
                suffix = "" if len(samples) <= 10 else f"_page_{start // 10 + 1:02d}"
                stem = directory / f"10_cluster_{int(event['cluster']):02d}_{event_id}_realizations{suffix}"
                for extension in ("png", "pdf"):
                    path = stem.with_suffix(f".{extension}")
                    fig.savefig(path, dpi=240)
                    rows.append({"cluster": int(event["cluster"]), "event_id": event_id,
                                 "hour_offset": int(hours[plane]), "first_realization": start + 1,
                                 "last_realization": start + count, "n_support_points": len(longitude),
                                 "color_min_c": vmin, "color_max_c": vmax, "figure": str(path)})
            finally:
                plt.close(fig)
        print(f"Realization panels: {event_id}, {len(samples)} saved fields plotted", flush=True)
    pd.DataFrame(rows, columns=["cluster", "event_id", "hour_offset", "first_realization", "last_realization",
                               "n_support_points", "color_min_c", "color_max_c", "figure"]).to_csv(
                                   directory / "10_realization_panel_manifest.csv", index=False)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=str(ROOT / "10_sgs_config.json"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    for key in ("events_directory", "output_directory"):
        path = Path(config[key]).expanduser()
        config[key] = str(path if path.is_absolute() else ROOT / path)
    plot_first_events(config)


if __name__ == "__main__":
    main()
