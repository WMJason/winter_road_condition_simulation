#!/usr/bin/env python3
"""Local conditional SGS using Step 9 histories and existing Iowa road supports.

Defaults to the issue-time road map conditioned on seven completed RWIS hours.
Set simulation_hour_offsets to [-6,-5,-4,-3,-2,-1,0] for a JOINT history.
No road download, imputation, regression fitting or covariance fitting occurs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"step10_sgs_mpl"))
import h5py
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import skew, kurtosis
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
TARGET = "road_surface_temp_c_mean"
STATE = {}


def resolve(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda:stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def seed_for(base, event_id, purpose):
    return int.from_bytes(hashlib.sha256(f"{base}|{event_id}|{purpose}".encode()).digest()[:8], "little")


def load_helpers():
    spec = importlib.util.spec_from_file_location("sgs_step7_helpers",
        ROOT.parent/"7_v2_fit_cluster_specific_regression_variograms_with_elevation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.original_helpers()


def covariance(d, u, p):
    """Exactly the Step 7 V2 product correlation, NOT practical-range scaling."""
    x, y = np.asarray(d)/p["space_range_km"], np.abs(u)/p["time_range_hours"]
    family = p["model_family"]
    if family == "exponential":
        rho = np.exp(-x-y)
    elif family == "gaussian":
        rho = np.exp(-x*x-y*y)
    elif family == "spherical":
        x, y = np.minimum(x,1), np.minimum(y,1)
        rho = np.maximum(1-1.5*x+.5*x**3,0)*np.maximum(1-1.5*y+.5*y**3,0)
    else:
        raise ValueError(f"Unknown covariance family: {family}")
    return p["partial_sill"]*rho


def feature_frame(lat, lon, hours_utc, projection, helpers, elevation=None):
    frame = pd.DataFrame({"latitude":np.asarray(lat), "longitude":np.asarray(lon),
                          "hour_utc":pd.to_datetime(hours_utc, utc=True).tz_localize(None)})
    frame["hour_local"] = pd.to_datetime(frame.hour_utc, utc=True).dt.tz_convert("America/Chicago").dt.tz_localize(None)
    frame=helpers.add_features(frame, projection)
    if elevation is not None:
        frame["elevation_m"]=np.asarray(elevation,float)
    return frame


def trend(frame, model):
    coefficients = model["regression_coefficients"]
    terms = ["const"]+model["selected_features"]
    if not all(term in coefficients for term in terms):
        raise ValueError("Missing regression coefficient; refusing a silent zero-coefficient fallback")
    result = np.full(len(frame), float(coefficients["const"]))
    for term in model["selected_features"]:
        result += frame[term].to_numpy(float)*float(coefficients[term])
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite regression trend")
    return result


def collapse_conditions(xy, hours, values):
    coordinates = np.column_stack([xy,hours])
    unique, first, inverse = np.unique(coordinates,axis=0,return_index=True,return_inverse=True)
    for group in range(len(unique)):
        members = values[inverse==group]
        if np.ptp(members)>1e-5:
            raise ValueError("Conflicting observed residuals at identical coordinates/time; a measurement-error model is needed")
    return unique[:,:2], unique[:,2], values[first]


def load_conditioning(event, model, events_directory, helpers, conditioning_policy="completed"):
    """Read exact Step 9 HDF5 values and retain their measured/imputed provenance."""
    if conditioning_policy not in ("completed", "observed_only"):
        raise ValueError("conditioning_policy must be completed or observed_only")
    with h5py.File(events_directory/"9_selected_event_histories.h5", "r") as store:
        ids = store["parent_event_ids"].asstr()[:].tolist()
        index = ids.index(event["parent_event_id"])
        if int(store["sample_ids"][index]) != int(event["sample_id"]) or int(store["clusters"][index]) != int(event["cluster"]):
            raise ValueError("Step 9 manifest/HDF5 identifiers disagree")
        issue = pd.to_datetime(int(store["issue_time_utc_ns"][index]), utc=True)
        if issue != pd.to_datetime(event["sample_hour_utc"], utc=True):
            raise ValueError("Step 9 manifest/HDF5 issue timestamps disagree")
        offsets = store["hour_offsets"][:].astype(int)
        variables = store["variables"].asstr()[:].tolist()
        if variables[0]!=TARGET or not np.array_equal(offsets,np.arange(-6,1)):
            raise ValueError("Expected RST first and seven history offsets -6,...,0")
        values = store["rwsc"][index,:,:,0].astype(float)
        imputed = store["is_imputed"][index,:,:,0].astype(bool)
        station_ids = store["station_ids"].asstr()[:]
        lat,lon = store["latitude"][:],store["longitude"][:]
        elevation=store["station_elevation_m"][:] if "station_elevation_m" in store else None
        if elevation is None and "elevation_m" in model["selected_features"]:
            raise ValueError("Elevation model requires new Step 9 station elevations; rerun Step 9 after refitting Step 7")
    timestamps = pd.DatetimeIndex([issue+pd.Timedelta(hours=int(h)) for h in offsets]).repeat(len(lat))
    frame = feature_frame(np.tile(lat,7), np.tile(lon,7), timestamps, model["projection"], helpers,
                          None if elevation is None else np.tile(elevation,7))
    frame[TARGET] = values.ravel()
    frame["hour_offset"] = np.repeat(offsets,len(lat))
    frame["rwis_id"] = np.tile(station_ids,7)
    frame["is_imputed"] = imputed.ravel()
    frame["regression_trend_c"] = trend(frame,model)
    frame["residual_c"] = frame[TARGET]-frame.regression_trend_c
    frame["rst_input_type"] = np.where(frame.is_imputed,"imputed_completed","measured")
    observed = frame.copy() if conditioning_policy=="completed" else frame[~frame.is_imputed].copy()
    if observed.empty or not np.isfinite(observed.residual_c).all():
        raise ValueError("Conditioning RST must be nonempty and finite")
    xy,hours,residuals = collapse_conditions(observed[["x_km","y_km"]].to_numpy(),
        observed.hour_offset.to_numpy(float), observed.residual_c.to_numpy(float))
    return issue, observed, xy, hours, residuals


def prior_neighbors(tree, xy, visited, counts, target, maximum):
    """Exact nearest available predecessors per time plane; expand until sufficient."""
    n = len(xy)
    needed = np.minimum(counts,maximum)
    if not needed.any():
        return np.empty(0,dtype=int)
    size = min(n,max(16,maximum*2))
    while True:
        _, candidates = tree.query(target,k=size)
        candidates = np.atleast_1d(candidates).astype(int)
        available = visited[:,candidates]
        if size==n or np.all(available.sum(axis=1)>=needed):
            break
        size = min(n,size*2)
    indices=[]
    for plane in range(len(visited)):
        found=candidates[available[plane]][:int(needed[plane])]
        indices.extend((plane*n+found).tolist())
    return np.asarray(indices,dtype=int)


def simulate_group(cond_xy,cond_hours,cond_values,road_xy,hours,p,draw_seeds,path_seed,
                   neighbor_count=80,ridge=1e-8,progress=None):
    """Gaussian draws on one random path, with covariance-ranked local neighbors.

    All draws have independent innovations. Shared path reuses geometry/solves;
    with finite neighborhoods the resulting joint Gaussian is a path-dependent
    approximation, not an exact dense conditional simulation.
    """
    n,planes=len(road_xy),len(hours)
    total=n*planes
    results=np.full((len(draw_seeds),total),np.nan)
    generators=[np.random.default_rng(seed) for seed in draw_seeds]
    path=np.random.default_rng(path_seed).permutation(total)
    tree=cKDTree(road_xy)
    cond_tree=cKDTree(cond_xy)
    visited=np.zeros((planes,n),bool)
    counts=np.zeros(planes,int)
    sill=float(p["nugget"]+p["partial_sill"])
    if sill<=0:
        raise ValueError("Positive residual variance required")
    diagnostics={"coincident_targets":0,"negative_variances_clipped":0,
                 "least_squares_fallbacks":0,"zero_covariance_targets":0}
    for step,point in enumerate(path):
        plane,location=divmod(int(point),n)
        qxy,qhour=road_xy[location],hours[plane]
        coincident=cond_tree.query_ball_point(qxy,1e-9)
        matching=[i for i in coincident if cond_hours[i]==qhour]
        if matching:
            results[:,point]=cond_values[matching[0]]
            diagnostics["coincident_targets"]+=1
        else:
            previous=prior_neighbors(tree,road_xy,visited,counts,qxy,neighbor_count)
            rd_plane,rd_position=np.divmod(previous,n)
            candidate_xy=np.vstack([cond_xy,road_xy[rd_position]])
            candidate_hours=np.concatenate([cond_hours,hours[rd_plane]])
            d=np.linalg.norm(candidate_xy-qxy,axis=1)
            cross=covariance(d,np.abs(candidate_hours-qhour),p)
            # Every hard observation is considered; road predecessors include
            # nearest K per time plane, sufficient for monotone spatial kernels.
            chosen=np.lexsort((np.arange(len(cross)),-cross))[:neighbor_count]
            chosen=chosen[cross[chosen]>0]
            if not len(chosen):
                mean=np.zeros(len(draw_seeds));variance=sill
                diagnostics["zero_covariance_targets"]+=1
            else:
                loc=candidate_xy[chosen];t=candidate_hours[chosen]
                matrix=covariance(cdist(loc,loc),np.abs(t[:,None]-t[None,:]),p)
                matrix.flat[::len(matrix)+1]+=p["nugget"]+ridge
                vector=cross[chosen]
                try:
                    weights=cho_solve(cho_factor(matrix,lower=True,check_finite=False),vector,check_finite=False)
                except np.linalg.LinAlgError:
                    diagnostics["least_squares_fallbacks"]+=1
                    weights=np.linalg.lstsq(matrix,vector,rcond=1e-12)[0]
                variance=sill-float(vector@weights)
                if variance < -1e-6*max(1,sill):
                    raise ArithmeticError(f"Materially negative conditional variance {variance}; check covariance/coordinates")
                diagnostics["negative_variances_clipped"]+=int(variance<0)
                variance=max(variance,0.)
                hard=chosen<len(cond_values)
                mean=np.full(len(draw_seeds),float(weights[hard]@cond_values[chosen[hard]]))
                if (~hard).any():
                    preceding=previous[chosen[~hard]-len(cond_values)]
                    mean+=results[:,preceding]@weights[~hard]
            innovations=np.array([generator.standard_normal() for generator in generators])
            results[:,point]=mean+np.sqrt(variance)*innovations
        visited[plane,location]=True
        counts[plane]+=1
        if progress is not None and (step==total-1 or (step+1)%5000==0):
            progress(step+1,total)
    if not np.isfinite(results).all():
        raise ArithmeticError("Nonfinite simulated residuals")
    return results.reshape(len(draw_seeds),planes,n),diagnostics


def load_roads(config):
    path=Path(config["support_points_file"])
    columns=["segment_id","physical_id","segment_order","length_km","latitude","longitude"]
    roads=pd.read_csv(path,usecols=columns)
    if config.get("needs_elevation",False):
        elevation_path=Path(config["support_elevation_file"])
        if not elevation_path.exists():
            raise FileNotFoundError(f"Run decision_making/0c_add_elevation.py first: {elevation_path}")
        elevation=pd.read_csv(elevation_path,usecols=["segment_id","elevation_m","latitude","longitude"])
        if elevation.segment_id.duplicated().any():
            raise ValueError("Duplicate elevation support IDs")
        aligned=elevation.set_index("segment_id").reindex(roads.segment_id)
        if not np.isfinite(aligned.elevation_m).all():
            raise ValueError("Some road supports have no elevation; complete Step 0c first")
        if not np.allclose(aligned[["latitude","longitude"]],roads[["latitude","longitude"]],atol=1e-6,rtol=0):
            raise ValueError("Elevation support coordinates mismatch")
        roads["elevation_m"]=aligned.elevation_m.to_numpy(float)
    if roads.empty or roads.segment_id.duplicated().any():
        raise ValueError("Empty support file or duplicate segment IDs")
    if not np.isfinite(roads[["latitude","longitude","length_km"]]).all().all():
        raise ValueError("Nonfinite support coordinates/lengths")
    if not (roads.latitude.between(40,44)&roads.longitude.between(-97,-89)&roads.length_km.gt(0)).all():
        raise ValueError("Support points are outside Iowa or lengths are invalid")
    if config.get("max_support_points"):
        positions=np.linspace(0,len(roads)-1,min(int(config["max_support_points"]),len(roads)),dtype=int)
        roads=roads.iloc[positions].copy()
    return roads.reset_index(drop=True)


def initialize(config,model_metadata):
    STATE.clear()
    STATE.update(config=config,models=model_metadata,roads=load_roads(config),helpers=load_helpers())


def write_support_catalog(roads,output,signature,overwrite):
    path=output/"10_road_support_catalog.h5"
    if path.exists() and not overwrite:
        with h5py.File(path,"r") as store:
            if store.attrs.get("signature")!=signature:
                raise ValueError("Output support catalog belongs to different inputs; use a fresh output folder")
        return
    temporary=path.with_suffix(".h5.tmp")
    with h5py.File(temporary,"w") as store:
        store.attrs["signature"]=signature
        for key in roads.columns:
            values=roads[key].to_numpy()
            if values.dtype.kind in "OUS":
                # Step 0b IDs are ASCII hashes. Fixed-width IDs compress well,
                # variable-length string heaps repeated across 122k supports.
                encoded=np.asarray([str(value).encode("ascii") for value in values])
                store.create_dataset(key,data=encoded,
                                     compression="gzip",compression_opts=4,shuffle=True)
            else:
                store.create_dataset(key,data=values,compression="gzip",shuffle=True)
    temporary.replace(path)


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,default=str)+"\n")
    temporary.replace(path)


def event_plots(roads,trend_c,residual,observed,hours,event,directory,config):
    if not config["make_plots"]:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],"font.size":10,"pdf.fonttype":42})
    plane=int(np.flatnonzero(hours==0)[0]) if 0 in hours else len(hours)-1
    samples=residual[:,plane,:]+trend_c[plane]
    limit=np.quantile(samples,[.01,.99]); vmin,vmax=float(limit[0]),float(limit[1])
    if vmax<=vmin:
        vmax=vmin+1
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    take=np.linspace(0,len(roads)-1,min(len(roads),30000),dtype=int)
    for ax,values,title in [(axes[0],samples[0],"Realization 1"),(axes[1],samples.mean(axis=0),"Realization mean")]:
        picture=ax.scatter(roads.longitude.iloc[take],roads.latitude.iloc[take],c=values[take],s=.5,
                           cmap="coolwarm",vmin=vmin,vmax=vmax,rasterized=True)
        targets=observed[observed.hour_offset.eq(hours[plane])]
        measured=targets[~targets.is_imputed]
        filled=targets[targets.is_imputed]
        ax.scatter(measured.longitude,measured.latitude,s=12,c="black",marker="^",label="Measured RWIS")
        ax.scatter(filled.longitude,filled.latitude,s=12,facecolors="white",edgecolors="black",marker="o",label="Imputed RWIS")
        ax.set(xlabel="Longitude",ylabel="Latitude",title=title)
        ax.set_aspect(1/np.cos(np.deg2rad(42)))
    axes[0].legend(frameon=False,fontsize=7,loc="lower left",bbox_to_anchor=(0,1.12))
    fig.colorbar(picture,ax=axes[:2],label="RST (°C)",shrink=.8)
    axes[2].hist(samples.ravel(),bins=40,color="#17796e",density=True,alpha=.8,label="Road realizations")
    station=observed[observed.hour_offset.eq(hours[plane])]
    if len(station):
        axes[2].hist(station[TARGET],bins=15,histtype="step",density=True,color="black",label="RWIS conditioning inputs")
    axes[2].set(xlabel="RST (°C)",ylabel="Density")
    axes[2].legend(frameon=False,fontsize=8)
    fig.suptitle(f"{event['parent_event_id']} · hour offset {hours[plane]}",fontsize=12)
    for extension in ["png","pdf"]:
        fig.savefig(directory/f"10_sgs_diagnostic.{extension}",dpi=180)
    plt.close(fig)


def process_event(event):
    config,models,roads,helpers=[STATE[key] for key in ["config","models","roads","helpers"]]
    event_id=event["parent_event_id"]
    model=models[str(int(event["cluster"]))]
    output=Path(config["output_directory"])/"events"/event_id
    output.mkdir(parents=True,exist_ok=True)
    h5_path=output/"10_sgs_realizations.h5"
    summary_path=output/"10_sgs_summary.json"
    signature=digest({"run":config["simulation_signature"],"event":event})
    if h5_path.exists() and summary_path.exists() and not config["overwrite"]:
        summary=json.loads(summary_path.read_text())
        accepted={signature}
        accepted.update(digest({"run":run,"event":event}) for run in config.get("plot_compatible_signatures",[]))
        with h5py.File(h5_path,"r") as store:
            valid=(summary.get("signature") in accepted and store.attrs.get("signature")==summary.get("signature")
                   and store["residual_c"].shape==(config["n_realizations"],len(config["simulation_hour_offsets"]),len(roads)))
        if not valid:
            raise ValueError(f"{event_id}: incompatible completed output; use fresh folder or overwrite")
        print(f"Resume: {event_id} already complete",flush=True)
        return summary
    started=time.perf_counter()
    issue,observed,cond_xy,cond_hours,cond_res=load_conditioning(event,model,Path(config["events_directory"]),helpers,config["conditioning_policy"])
    hours=np.array(config["simulation_hour_offsets"],dtype=int)
    road_elevation=roads.elevation_m if "elevation_m" in roads else None
    features=feature_frame(roads.latitude,roads.longitude,pd.DatetimeIndex([issue]*len(roads)),model["projection"],helpers,road_elevation)
    road_xy=features[["x_km","y_km"]].to_numpy(float)
    unique_xy,unique_index,inverse=np.unique(road_xy,axis=0,return_index=True,return_inverse=True)
    trend_c=[]
    for hour in hours:
        f=feature_frame(roads.latitude,roads.longitude,pd.DatetimeIndex([issue+pd.Timedelta(hours=int(hour))]*len(roads)),model["projection"],helpers,road_elevation)
        trend_c.append(trend(f,model))
    trend_c=np.asarray(trend_c)
    realizations=int(config["n_realizations"])
    groups=min(int(config["path_groups"]),realizations)
    residual=np.empty((realizations,len(hours),len(roads)),dtype=config["float_precision"])
    diagnostics=[]; seed_rows=[]
    with threadpool_limits(limits=int(config["threads_per_worker"])):
        for group,indices in enumerate(np.array_split(np.arange(realizations),groups)):
            path_seed=seed_for(config["random_seed"],event_id,f"path:{group}")
            draw_seeds=[seed_for(config["random_seed"],event_id,f"realization:{int(i)}") for i in indices]
            def progress(done,total):
                print(f"  {event_id}, path {group+1}/{groups}: {done:,}/{total:,} locations ({time.perf_counter()-started:.1f}s)",flush=True)
            draws,checks=simulate_group(cond_xy,cond_hours,cond_res,unique_xy,hours,model["variogram"],draw_seeds,path_seed,
                int(config["neighbor_count"]),float(config["kriging_ridge"]),progress)
            residual[indices]=draws[:,:,inverse]
            diagnostics.append(checks)
            seed_rows.extend({"realization":int(i),"draw_seed":str(seed),"path_group":group,"path_seed":str(path_seed)}
                             for i,seed in zip(indices,draw_seeds))
    temporary=h5_path.with_suffix(".h5.tmp")
    with h5py.File(temporary,"w") as store:
        store.attrs.update(signature=signature,event_id=event_id,cluster=int(event["cluster"]),
            sample_id=int(event["sample_id"]),issue_time_utc=issue.isoformat(),
            conditioning_policy=config["conditioning_policy"],
            covariance_form="separable_product",axis_order="realization,hour_offset,support_point",
            support_catalog="../../10_road_support_catalog.h5",nugget_interpretation="unresolved process variation, not observation error",
            scope="issue-time snapshot" if hours.tolist()==[0] else "joint historical space-time field")
        store.create_dataset("hour_offsets",data=hours)
        store.create_dataset("trend_c",data=trend_c.astype(config["float_precision"]),compression="gzip",shuffle=True)
        store.create_dataset("residual_c",data=residual,chunks=(1,1,min(len(roads),16384)),
                             compression="gzip",compression_opts=4,shuffle=True)
        store.attrs["reconstruction"]="rst_c = trend_c[None,:,:] + residual_c"
    temporary.replace(h5_path)
    observed.to_csv(output/"10_observed_rst_conditioning.csv.gz",index=False)
    pd.DataFrame(seed_rows).to_csv(output/"10_realization_seeds.csv",index=False)
    stats=[]
    for realization in range(realizations):
        for plane,hour in enumerate(hours):
            values=trend_c[plane]+residual[realization,plane]
            weights=roads.length_km.to_numpy()
            stats.append({"event_id":event_id,"cluster":int(event["cluster"]),"realization":realization,
                "hour_offset":int(hour),"n_support_points":len(roads),"mean_c":float(values.mean()),
                "sd_c":float(values.std()),"minimum_c":float(values.min()),"q05_c":float(np.quantile(values,.05)),
                "median_c":float(np.median(values)),"q95_c":float(np.quantile(values,.95)),"maximum_c":float(values.max()),
                "length_weighted_mean_c":float(np.average(values,weights=weights)),
                "fraction_minus2_to_plus1":float(np.mean((values>=-2)&(values<=1)))})
    pd.DataFrame(stats).to_csv(output/"10_realization_statistics.csv",index=False)
    event_plots(roads,trend_c,residual,observed,hours,event,output,config)
    summary={"event_id":event_id,"cluster":int(event["cluster"]),"sample_id":int(event["sample_id"]),
        "signature":signature,"h5_file":str(h5_path),"n_realizations":realizations,"n_road_points":len(roads),
        "unique_spatial_points":len(unique_xy),"simulation_hour_offsets":hours.tolist(),
        "conditioning_observations":len(observed),"conditioning_stations":int(observed.rwis_id.nunique()),
        "conditioning_policy":config["conditioning_policy"],
        "measured_conditioning_values":int((~observed.is_imputed).sum()),
        "imputed_conditioning_values":int(observed.is_imputed.sum()),
        "unique_conditioning_cells":len(cond_res),"variogram":model["variogram"],"path_groups":groups,
        "conditioning_residual_skewness":float(skew(cond_res)),
        "conditioning_residual_excess_kurtosis":float(kurtosis(cond_res)),
        "normal_score_transform_applied":False,"numerical_diagnostics":diagnostics,
        "parameter_at_bound":bool(model["variogram"].get("parameter_at_bound",False)),
        "elapsed_seconds":round(time.perf_counter()-started,2),"output_bytes":h5_path.stat().st_size,
        "limitations":["Local-neighborhood, path-dependent Gaussian approximation; not exact full-covariance conditioning.",
            "Shared-path realizations are independent conditional on the chosen path, observations and fixed parameters.",
            "Fitted regression/covariance treated as fixed; Gaussian residuals assumed, not established by this simulation."]}
    if config["conditioning_policy"]=="completed":
        summary["limitations"].append("Completed/imputed RST is fixed as pseudo-observed conditioning; imputation uncertainty is not propagated.")
    atomic_json(summary_path,summary)
    print(f"Completed {event_id}: {len(roads):,} supports, {realizations} realizations, {summary['elapsed_seconds']}s",flush=True)
    return summary


def checked_worker(event):
    try:
        return process_event(event)
    except Exception as error:
        import traceback
        print(f"FAILED {event['parent_event_id']}: {error}",flush=True)
        return {"event_id":event["parent_event_id"],"error":str(error),"traceback":traceback.format_exc()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config",nargs="?",default=str(ROOT/"10_sgs_config.json"))
    parser.add_argument("--max-events",type=int)
    parser.add_argument("--max-support-points",type=int)
    parser.add_argument("--realizations",type=int)
    parser.add_argument("--workers",type=int)
    parser.add_argument("--output-directory")
    parser.add_argument("--hours",type=int,nargs="+")
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--check-only",action="store_true")
    parser.add_argument("--smoke-test",action="store_true")
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    defaults={"simulation_hour_offsets":[0],"path_groups":1,"threads_per_worker":1,
        "make_plots":True,"max_support_points":None,"neighbor_count":80,"kriging_ridge":1e-8,"conditioning_policy":"completed",
        "float_precision":"float32","workers":1,"overwrite":False,"events":None}
    for key,value in defaults.items():
        config.setdefault(key,value)
    if config["conditioning_policy"] not in ("completed","observed_only"):
        raise ValueError("conditioning_policy must be completed or observed_only")
    for key,value in [("n_realizations",args.realizations),("workers",args.workers),
        ("max_support_points",args.max_support_points),("output_directory",args.output_directory),
        ("simulation_hour_offsets",args.hours)]:
        if value is not None:
            config[key]=value
    if args.smoke_test:
        config.update(n_realizations=2,workers=1,max_support_points=256)
        config["output_directory"]=args.output_directory or str(Path(tempfile.gettempdir())/"step10_sgs_smoke")
        args.max_events=1
    config["overwrite"]|=args.overwrite
    for name in ["n_realizations","workers","path_groups","threads_per_worker","neighbor_count"]:
        if isinstance(config[name],bool) or int(config[name])!=config[name] or int(config[name])<1:
            raise ValueError(f"{name} must be a positive integer")
    if (config["float_precision"] not in ["float32","float64"]
            or not np.isfinite(config["kriging_ridge"]) or config["kriging_ridge"]<0):
        raise ValueError("Invalid precision or ridge")
    hours=config["simulation_hour_offsets"]
    if not hours or len(set(hours))!=len(hours) or any(isinstance(h,bool) or int(h)!=h or h not in range(-6,1) for h in hours):
        raise ValueError("simulation_hour_offsets must be unique integers from -6 to 0")
    config["simulation_hour_offsets"]=sorted(map(int,hours))
    if (args.max_events is not None and args.max_events<1) or (config["max_support_points"] is not None and config["max_support_points"]<1):
        raise ValueError("Positive event/support limits required")
    for key in ["events_directory","model_metadata","support_points_file","output_directory"]:
        config[key]=str(resolve(config[key]))
    metadata=json.loads(Path(config["model_metadata"]).read_text())
    events=pd.read_csv(Path(config["events_directory"])/"9_selected_weather_events.csv")
    required=["parent_event_id","cluster","sample_id","sample_hour_utc"]
    if not set(required).issubset(events) or events.parent_event_id.duplicated().any():
        raise ValueError("Expected unique Step 9 parent_event_id, cluster, sample_id and UTC timestamp")
    if config["events"] is not None:
        unknown=set(config["events"])-set(events.parent_event_id)
        if unknown:
            raise ValueError(f"Unknown event IDs: {sorted(unknown)}")
        events=events[events.parent_event_id.isin(config["events"])]
    if args.max_events:
        events=events.head(args.max_events)
    if events.empty:
        raise ValueError("No selected events")
    config["needs_elevation"]=any("elevation_m" in metadata[str(int(k))]["selected_features"] for k in events.cluster.unique())
    if config["needs_elevation"]:
        config["support_elevation_file"]=str(resolve(config.get("support_elevation_file",
            "../../decision_making/0c_elevation_outputs/0c_road_support_points_with_elevation.csv.gz")))
    for cluster in events.cluster.unique():
        model=metadata[str(int(cluster))]
        for term in ["const"]+model["selected_features"]:
            if term not in model["regression_coefficients"]:
                raise ValueError(f"Missing coefficient {term}")
        p=model["variogram"]
        if (not all(np.isfinite(p[key]) for key in ["partial_sill","nugget","space_range_km","time_range_hours"])
                or p["model_family"] not in ["exponential","gaussian","spherical"]
                or p.get("covariance_form")!="separable_product" or p["partial_sill"]<=0
                or p["nugget"]<0 or p["space_range_km"]<=0 or p["time_range_hours"]<=0):
            raise ValueError("Invalid fitted covariance metadata")
        if sha256(model["model_file"])!=model["model_sha256"]:
            raise ValueError("Step 7 model changed after Step 9 export; rerun Step 9 first")
    source_files=[Path(__file__),ROOT.parent/"7_v2_fit_cluster_specific_regression_variograms_with_elevation.py",
        ROOT.parent/"7_fit_cluster_specific_regression_variograms.py",Path(config["support_points_file"]),
        Path(config["model_metadata"]),Path(config["events_directory"])/"9_selected_event_histories.h5"]
    if config["needs_elevation"]:
        source_files.append(Path(config["support_elevation_file"]))
    config["input_hashes"]={str(path):sha256(path) for path in source_files}
    effective={key:config[key] for key in ["n_realizations","neighbor_count","kriging_ridge","random_seed",
        "simulation_hour_offsets","path_groups","float_precision","max_support_points","input_hashes","conditioning_policy"]}
    config["simulation_signature"]=digest(effective)
    # Only the immediately preceding, verified simulation-identical version is
    # compatible. Other input/settings changes must still invalidate the cache.
    previous_hashes=dict(config["input_hashes"])
    previous_hashes[str(Path(__file__))]="6cb5aefdeae9e457d2c0456b806ea43004091af2fe1b030df2e9d9e1f0db1013"
    config["plot_compatible_signatures"]=[]
    if config["conditioning_policy"]=="observed_only":
        legacy={key:value for key,value in effective.items() if key!="conditioning_policy"}
        config["plot_compatible_signatures"]=[digest({**legacy,"input_hashes":previous_hashes})]
    roads=load_roads(config)
    helpers=load_helpers()
    for event in events[required].to_dict("records"):
        load_conditioning(event,metadata[str(int(event["cluster"]))],Path(config["events_directory"]),helpers,config["conditioning_policy"])
    print(f"Validated {len(events)} parent events; {len(roads):,} existing road supports; hours {hours}; no road download.",flush=True)
    if args.check_only:
        return
    raw_mb=len(events)*len(roads)*len(hours)*config["n_realizations"]*np.dtype(config["float_precision"]).itemsize/1024**2
    print(f"Residual arrays before compression: approximately {raw_mb:.1f} MiB for this run; metadata/trends/plots are additional.",flush=True)
    output=Path(config["output_directory"])
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/"10_cluster_model_metadata.json",metadata)
    write_support_catalog(roads,output,digest({"support":config["input_hashes"][config["support_points_file"]],
        "limit":config["max_support_points"]}),config["overwrite"])
    records=[];failures=[]
    event_records=events[required].to_dict("records")
    if config["workers"]==1:
        initialize(config,metadata)
        for result in map(checked_worker,event_records):
            (failures if "error" in result else records).append(result)
    else:
        import multiprocessing
        with ProcessPoolExecutor(max_workers=config["workers"],mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize,initargs=(config,metadata)) as executor:
            for result in executor.map(checked_worker,event_records):
                (failures if "error" in result else records).append(result)
    rows=[{key:record[key] for key in ["event_id","cluster","sample_id","n_realizations","n_road_points",
        "conditioning_observations","elapsed_seconds","output_bytes","h5_file"]} for record in records]
    pd.DataFrame(rows,columns=["event_id","cluster","sample_id","n_realizations","n_road_points",
        "conditioning_observations","elapsed_seconds","output_bytes","h5_file"]).to_csv(output/"10_sgs_manifest.csv",index=False)
    all_stats=[pd.read_csv(Path(record["h5_file"]).parent/"10_realization_statistics.csv") for record in records]
    if all_stats:
        pd.concat(all_stats,ignore_index=True).to_csv(output/"10_all_realization_statistics.csv.gz",index=False)
    pd.DataFrame(failures,columns=["event_id","error","traceback"]).to_csv(output/"10_sgs_failures.csv",index=False)
    atomic_json(output/"10_sgs_run_summary.json",{"config":config,"requested_events":len(events),
        "successful":len(records),"failed":len(failures),"partial_support_run":config["max_support_points"] is not None,
        "scope":"Issue-time map conditioned on seven-hour RWIS inputs" if config["simulation_hour_offsets"]==[0] else "Joint seven-hour/subset space-time simulation",
        "support_note":"Existing <=500m physical-cell midpoints; UTM columns NOT used for covariance. Lat/lon converted using each fitted model's exact local-km convention.",
        "failures":failures})
    if config["make_plots"]:
        spec=importlib.util.spec_from_file_location("sgs_realization_panels",ROOT/"10_s2_plot_cluster_realizations.py")
        plotter=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plotter)
        plotter.plot_first_events(config)
    if failures:
        for failure in failures:
            print(f"FAILED {failure['event_id']}: {failure['error']}",flush=True)
        raise SystemExit(1)
    print(f"Finished: {len(records)}/{len(events)} events. Output: {output}",flush=True)


if __name__=="__main__":
    main()
