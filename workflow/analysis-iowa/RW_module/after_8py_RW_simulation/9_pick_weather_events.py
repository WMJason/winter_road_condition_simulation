#!/usr/bin/env python3
"""Export nearest-centroid fitting histories as parent events for later SGS."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RW_ROOT = ROOT.parent


def resolve(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def step7_module():
    spec = importlib.util.spec_from_file_location("step7_parent_event_helpers",
        RW_ROOT/"7_v2_fit_cluster_specific_regression_variograms_with_elevation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def as_bool(values):
    converted = values.astype(str).str.lower().map({"true": True, "false": False})
    if converted.isna().any():
        raise ValueError(f"Invalid boolean column {values.name}")
    return converted


def nearest_events(assignments, audits, count):
    if count < 1:
        raise ValueError("events_per_cluster must be positive")
    data = assignments.copy()
    if data.sample_id.duplicated().any():
        raise ValueError("Duplicate source sample IDs")
    for name in ["used_for_kmeans_fit", "overlaps_protected_period", "is_cv_test_event"]:
        data[name] = as_bool(data[name])
    rows = []
    for cluster, audit_ids in sorted(audits.items()):
        group = data[data.cluster.eq(cluster) & data.sample_id.isin(audit_ids)].copy()
        if set(group.sample_id) != set(audit_ids):
            raise ValueError(f"Cluster {cluster}: model audit and assignments disagree")
        if (not group.used_for_kmeans_fit.all() or group.overlaps_protected_period.any()
                or group.is_cv_test_event.any()):
            raise ValueError("Only unprotected Step 7 fitting events may be selected")
        distance = group.distance_to_assigned_centroid.to_numpy(float)
        if not np.isfinite(distance).all() or (distance < 0).any():
            raise ValueError("Invalid centroid distances")
        if len(group) < count:
            raise ValueError(f"Cluster {cluster}: only {len(group)} fitting events; requested {count}")
        group["sort_time"] = pd.to_datetime(group.sample_hour_utc, utc=True)
        chosen = group.sort_values(["distance_to_assigned_centroid", "sort_time", "sample_id"],
                                   kind="stable").head(count).drop(columns="sort_time")
        chosen["centroid_rank"] = np.arange(1, count+1)
        chosen["cluster_fitting_event_count"] = len(group)
        chosen["parent_event_id"] = [f"C{cluster:02d}_R{rank:02d}_S{sample}"
            for rank, sample in zip(chosen.centroid_rank, chosen.sample_id)]
        rows.append(chosen)
    if not rows:
        raise ValueError("No fitted clusters")
    return pd.concat(rows, ignore_index=True)


def export_frame(step7, arrays, event, stations, package, helpers):
    frame = step7.event_frame(arrays, event, stations, package["projection"], helpers)
    position = int(event.iloc[0].source_position)
    for variable_index, variable in enumerate(arrays["variables"].astype(str)):
        frame[variable] = arrays["X"][position,:,:,variable_index].ravel()
        imputed = arrays["imputation_mask"][position,:,:,variable_index].ravel().astype(bool)
        frame[f"{variable}__is_imputed"] = imputed
        frame[f"{variable}__is_observed"] = ~imputed
    frame["hour_iowa_local_iso"] = pd.to_datetime(frame.hour_utc, utc=True).dt.tz_convert("America/Chicago").map(lambda x:x.isoformat())
    frame["regression_trend_c"] = step7.predict_trend(package["regression_model"], package["selected_features"], frame)
    frame["rst_residual_c"] = frame[step7.TARGET]-frame.regression_trend_c
    frame["rst_conditioning_allowed"] = ~frame.rst_is_imputed
    frame["parent_event_id"] = event.iloc[0].parent_event_id
    frame["cluster"] = int(event.iloc[0].cluster)
    frame["centroid_rank"] = int(event.iloc[0].centroid_rank)
    frame["distance_to_assigned_centroid"] = float(event.iloc[0].distance_to_assigned_centroid)
    frame["selected_regression_terms"] = ";".join(package["selected_features"])
    return frame


def draw_event(step7, frame, event, output, limits, dpi):
    plt = step7.plt
    values = frame[step7.TARGET].to_numpy().reshape(7,60)
    observed = ~frame.rst_is_imputed.to_numpy().reshape(7,60)
    locations = frame.iloc[:60]
    order = np.argsort(-locations.latitude.to_numpy(), kind="stable")
    fig, axes = plt.subplots(2,2,figsize=(11,7), constrained_layout=True)
    image = axes[0,0].imshow(values[:,order], aspect="auto", origin="lower", cmap="coolwarm",
        vmin=limits[0], vmax=limits[1], extent=(-.5,59.5,-6.5,.5))
    axes[0,0].set(xlabel="RWIS stations ordered north to south", ylabel="Hour relative to issue time")
    fig.colorbar(image,ax=axes[0,0],label="Completed RST (°C)")
    issue = frame[frame.hour_offset.eq(0)]
    scatter = axes[0,1].scatter(issue.x_km,issue.y_km,c=issue[step7.TARGET],cmap="coolwarm",
        vmin=limits[0],vmax=limits[1],s=28)
    missing = issue[issue.rst_is_imputed]
    axes[0,1].scatter(missing.x_km,missing.y_km,s=65,facecolors="none",edgecolors="black",
                      linewidths=.8,label="Imputed RST")
    axes[0,1].set(xlabel="East–west coordinate (km)",ylabel="North–south coordinate (km)")
    axes[0,1].set_aspect("equal",adjustable="datalim")
    if len(missing):
        axes[0,1].legend(frameon=False,fontsize=8)
    fig.colorbar(scatter,ax=axes[0,1],label="Issue-time RST (°C)")
    bins = np.linspace(limits[0],limits[1],26)
    axes[1,0].hist(values[observed],bins=bins,color="#19766a",alpha=.8,label="Observed")
    axes[1,0].hist(values[~observed],bins=bins,histtype="step",color="#ab4d3b",linewidth=1.5,label="Imputed")
    axes[1,0].set(xlabel="RST over seven-hour history (°C)",ylabel="Station-hour count")
    axes[1,0].legend(frameon=False)
    for index in range(60):
        axes[1,1].plot(np.arange(-6,1),values[:,index],color="#79898d",alpha=.18,linewidth=.6)
    axes[1,1].plot(np.arange(-6,1),np.median(values,axis=1),color="#19766a",linewidth=2,label="Station median")
    axes[1,1].set(xlabel="Hour relative to issue time",ylabel="Completed RST (°C)",ylim=limits)
    axes[1,1].legend(frameon=False)
    for ax in axes.flat:
        ax.spines[["top","right"]].set_visible(False)
    record = event.iloc[0]
    local = pd.to_datetime(record.sample_hour_utc,utc=True).tz_convert("America/Chicago")
    fig.suptitle(f"Cluster {int(record.cluster)} · rank {int(record.centroid_rank)} · {local:%Y-%m-%d %H:%M %Z}",fontsize=12)
    for extension in ["png","pdf"]:
        fig.savefig(output/f"9_rst_diagnostic.{extension}",dpi=dpi)
    plt.close(fig)


def summary_plots(step7, selected, distributions, output, dpi):
    plt = step7.plt
    fig, axes = plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    clusters = sorted(selected.cluster.unique())
    for cluster in clusters:
        g=selected[selected.cluster.eq(cluster)]
        axes[0].plot(g.centroid_rank,g.distance_to_assigned_centroid,marker="o",markersize=3,label=f"Cluster {cluster}")
    axes[0].set(xlabel="Within-cluster centroid rank",ylabel="Distance in fitted clustering space")
    axes[0].legend(frameon=False,ncol=2,fontsize=8)
    axes[1].boxplot([distributions[int(k)] for k in clusters],
        showfliers=False,medianprops={"color":"#19766a"})
    axes[1].set_xticks(np.arange(1,len(clusters)+1),[str(k) for k in clusters])
    axes[1].set(xlabel="Cluster",ylabel="Completed RST across selected histories (°C)")
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
    for extension in ["png","pdf"]:
        fig.savefig(output/f"9_selected_event_overview.{extension}",dpi=dpi)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(ROOT/"9_event_selection_config.json"))
    parser.add_argument("--events-per-cluster",type=int)
    parser.add_argument("--output-directory")
    parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    count=int(args.events_per_cluster if args.events_per_cluster is not None else config["events_per_cluster"])
    step7=step7_module()
    fit_path=resolve(config["step7_config"])
    fit_config=json.loads(fit_path.read_text())
    for key in ["cluster_directory","station_file","output_directory"]:
        path=Path(fit_config[key])
        fit_config[key]=str(path if path.is_absolute() else RW_ROOT/path)
    arrays, fitting, stations, _=step7.load_inputs(fit_config)
    cluster_dir=Path(fit_config["cluster_directory"])
    assigned=pd.read_csv(cluster_dir/"5_v2_cluster_assignments_all_samples.csv.gz")
    source=json.loads((cluster_dir/"5_v2_effective_config.json").read_text())
    source_index=pd.read_csv(source["input_index"])
    packages,audits,model_info={},{},{}
    for cluster in sorted(fitting.cluster.unique()):
        cluster=int(cluster)
        directory=Path(fit_config["output_directory"])/f"cluster_{cluster:02d}"
        model_path=directory/"7_v2_cluster_strk_model.joblib"
        package=joblib.load(model_path)
        if (int(package["cluster"])!=cluster or package["target"]!=step7.TARGET
                or package.get("horizons")!=[0] or package.get("covariance_form")!="separable_product"):
            raise ValueError(f"Incompatible model: {model_path}")
        if not set(package["selected_features"]).issubset(step7.FEATURES):
            raise ValueError("Model contains unsupported regression covariates")
        audit=pd.read_csv(directory/"7_v2_fitting_event_audit.csv")
        if audit.sample_id.duplicated().any():
            raise ValueError("Duplicate model fitting IDs")
        audits[cluster]=set(audit.sample_id)
        packages[cluster]=package
        model_info[str(cluster)]={"model_file":str(model_path),"model_sha256":sha256(model_path),
            "training_signature":package["run_signature"],"projection":package["projection"],
            "selected_features":package["selected_features"],"candidate_features":package["candidate_features"],
            "regression_coefficients":package.get("regression_coefficients_original_units",package["regression_model"].params.to_dict()),
            "regression_coefficient_units":"original covariate units, not z-scores",
            "fitted_design_regression_coefficients":package["regression_model"].params.to_dict(),
            "regression_scaling":package.get("regression_scaling",{}),
            "variogram":package["variogram"]}
    candidate_features=next(iter(packages.values()))["candidate_features"]
    if any(package["candidate_features"]!=candidate_features for package in packages.values()):
        raise ValueError("Mixed candidate feature definitions across fitted cluster models")
    selected=nearest_events(assigned,audits,count)
    selected["source_position"]=pd.Index(source_index.sample_id).get_indexer(selected.sample_id)
    if (selected.source_position<0).any():
        raise ValueError("Missing source event")
    if not np.array_equal(pd.to_datetime(arrays["sample_hour_utc"][selected.source_position],utc=True).to_numpy(),
                          pd.to_datetime(selected.sample_hour_utc,utc=True).to_numpy()):
        raise ValueError("Source/selection timestamp mismatch")
    output=resolve(args.output_directory or config["output_directory"])
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}; use a new folder or --overwrite")
    output.mkdir(parents=True,exist_ok=True)
    selected["hdf5_event_index"]=np.arange(len(selected))
    selected["event_table"]=[f"events/{name}/9_station_hour_data.csv.gz" for name in selected.parent_event_id]
    selected.to_csv(output/"9_selected_weather_events.csv",index=False)
    stations.rename_axis("rwis_id").reset_index().to_csv(output/"9_rwis_station_metadata.csv",index=False)
    (output/"9_cluster_model_metadata.json").write_text(json.dumps(model_info,indent=2)+"\n")
    histories=arrays["X"][selected.source_position.to_numpy(int)]
    masks=arrays["imputation_mask"][selected.source_position.to_numpy(int)]
    limits=(float(histories[...,0].min())-1,float(histories[...,0].max())+1)
    step7.plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
                               "font.size":10,"pdf.fonttype":42})
    helpers=step7.original_helpers()
    statistics=[]
    with h5py.File(output/"9_selected_event_histories.h5","w") as store:
        store.attrs["schema_version"]=1
        store.attrs["axis_order"]="event,hour_offset,station,variable"
        store.attrs["timezone"]="America/Chicago"
        for key,value in [("rwsc",histories),("is_imputed",masks)]:
            store.create_dataset(key,data=value,compression="gzip",compression_opts=4,shuffle=True)
        text_type=h5py.string_dtype("utf-8")
        for key,value in [("station_ids",arrays["station_ids"].astype(str)),("variables",arrays["variables"].astype(str)),
                          ("parent_event_ids",selected.parent_event_id.to_numpy())]:
            store.create_dataset(key,data=np.asarray(value,dtype=object),dtype=text_type)
        for key,value in [("hour_offsets",arrays["hour_offsets"]),("sample_ids",selected.sample_id.to_numpy()),
                          ("clusters",selected.cluster.to_numpy()),
                          ("issue_time_utc_ns",pd.to_datetime(selected.sample_hour_utc,utc=True).astype("int64").to_numpy()),
                          ("latitude",stations.latitude.to_numpy()),("longitude",stations.longitude.to_numpy())]:
            store.create_dataset(key,data=value)
        if "elevation_m" in stations:
            store.create_dataset("station_elevation_m",data=stations.elevation_m.to_numpy(float))
        covariates=store.create_dataset("regression_candidate_covariates",shape=(len(selected),7,60,len(candidate_features)),
                                       dtype="f8",compression="gzip",shuffle=True)
        store.create_dataset("regression_candidate_names",data=np.array(candidate_features,dtype=object),dtype=text_type)
        trend_data=store.create_dataset("regression_trend_c",shape=(len(selected),7,60),dtype="f8",compression="gzip")
        for i in range(len(selected)):
            event=selected.iloc[[i]]
            record=event.iloc[0]
            print(f"[{i+1}/{len(selected)}] {record.parent_event_id}",flush=True)
            frame=export_frame(step7,arrays,event,stations,packages[int(record.cluster)],helpers)
            directory=output/"events"/record.parent_event_id
            directory.mkdir(parents=True,exist_ok=True)
            frame.to_csv(directory/"9_station_hour_data.csv.gz",index=False)
            covariates[i]=frame[candidate_features].to_numpy().reshape(7,60,-1)
            trend_data[i]=frame.regression_trend_c.to_numpy().reshape(7,60)
            for variable in arrays["variables"].astype(str):
                for scope,subset in [("completed",frame),("observed_only",frame[frame[f"{variable}__is_observed"]])]:
                    values=subset[variable]
                    statistics.append({"parent_event_id":record.parent_event_id,"cluster":int(record.cluster),
                        "sample_id":record.sample_id,"variable":variable,"scope":scope,"n":len(values),
                        "mean":values.mean(),"sd":values.std(),"minimum":values.min(),"q05":values.quantile(.05),
                        "median":values.median(),"q95":values.quantile(.95),"maximum":values.max()})
            if config["make_plots"]:
                draw_event(step7,frame,event,directory,limits,int(config["plot_dpi"]))
    pd.DataFrame(statistics).to_csv(output/"9_event_rwsc_distribution_statistics.csv",index=False)
    gaps=[]
    for cluster,group in selected.groupby("cluster"):
        chronological=group.assign(time=pd.to_datetime(group.sample_hour_utc,utc=True)).sort_values("time")
        delta=chronological.time.diff().dt.total_seconds()/3600
        for index,(_,row) in enumerate(chronological.iterrows()):
            gaps.append({"cluster":int(cluster),"parent_event_id":row.parent_event_id,
                "previous_selected_parent_event_id":None if index==0 else chronological.iloc[index-1].parent_event_id,
                "hours_since_previous_selected_event":delta.iloc[index],
                "less_than_24h_from_previous":False if index==0 else bool(delta.iloc[index]<24)})
    pd.DataFrame(gaps).to_csv(output/"9_selected_event_temporal_spacing.csv",index=False)
    if config["make_plots"]:
        summary_plots(step7,selected,{int(k):histories[selected.cluster.eq(k).to_numpy(),:,:,0].ravel()
            for k in selected.cluster.unique()},output,int(config["plot_dpi"]))
    summary={"events_per_cluster":count,"selected_event_count":len(selected),"clusters":sorted(audits),
        "selection":"Ascending saved full PCA-space centroid distance; UTC timestamp and sample ID break ties; Step 7 fitting events only.",
        "minimum_temporal_separation_enforced":False,"under_24h_adjacent_selected_pairs":sum(g["less_than_24h_from_previous"] for g in gaps),
        "planned_realizations_per_event":int(config["planned_realizations_per_event"]),
        "sgs_executed":False,"source_config":str(fit_path),
        "source_hashes":{str(path):sha256(path) for path in [Path(__file__),fit_path,
            cluster_dir/"5_v2_cluster_assignments_all_samples.csv.gz",Path(source["input_archive"]),Path(source["input_index"])]},
        "covariates":"RST regression uses only geographic and cyclic Iowa-local calendar terms. Four other RWSC variables are exported, not regression predictors.",
        "conditioning_policy":"Use observed RST only by default: is_imputed=false. Completed values are preserved but not labelled observed.",
        "interpretation":"Typical near-centroid histories, not a random sample of all events; distances use all five variables and seven hours.",
        "wind_units":"Source units preserved; no unverified conversion.",
        "step10_warning":"Check range-bound fits and residual Gaussianity before SGS; do not simulate independent hourly fields or use prediction uncertainty as a covariance model."}
    (output/"9_selection_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(f"Finished: {len(selected)} parent histories. Output: {output}",flush=True)


if __name__=="__main__":
    main()
