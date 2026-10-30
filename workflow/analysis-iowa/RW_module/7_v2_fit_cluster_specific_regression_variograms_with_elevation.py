#!/usr/bin/env python3
"""Fit cluster-specific RST regression and residual covariance for NOWCASTING ONLY.

Uses Step 5 V2's actual fitting histories from the fixed Step 4 V2 dataset.
Cross-validation holds complete events out of regression/variogram fitting,
then excludes the target station's entire history from residual kriging.
No forecasts or CV-image-test event evaluation are produced.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "step7v2_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
TARGET = "road_surface_temp_c_mean"
FEATURES = ["x_km", "y_km", "x_km_sq", "y_km_sq", "x_y_interaction", "elevation_m",
            "hour_sin", "hour_cos", "month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos"]
FAMILIES = ["exponential", "gaussian", "spherical"]


def original_helpers():
    # Read-only reuse of feature definitions and regression report exports.
    path = ROOT / "7_fit_cluster_specific_regression_variograms.py"
    spec = importlib.util.spec_from_file_location("legacy_strk_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    errors = predicted-actual
    return {"n": len(actual), "mae_c": float(np.abs(errors).mean()),
            "mse_c2": float(np.square(errors).mean()), "rmse_c": float(np.sqrt(np.square(errors).mean())),
            "bias_c": float(errors.mean()),
            "r_squared": float(r2_score(actual, predicted)) if len(actual)>1 and np.var(actual)>0 else np.nan}


def predict_trend(model, terms, frame):
    design = scaled_design(frame,terms,getattr(model,"strk_feature_scaling",{}))
    return np.asarray(model.predict(sm.add_constant(design, has_constant="add")))


def fit_feature_scaling(frame, terms):
    means=frame[terms].mean()
    scales=frame[terms].std(ddof=0)
    scales=scales.where(scales>1e-12,1.)
    return {term:{"mean":float(means[term]),"scale":float(scales[term])} for term in terms}


def scaled_design(frame, terms, scaling):
    design=frame[terms].astype(float).copy()
    for term in terms:
        if term in scaling:
            design[term]=(design[term]-scaling[term]["mean"])/scaling[term]["scale"]
    return design


def original_unit_coefficients(model, terms):
    scaling=getattr(model,"strk_feature_scaling",{})
    names=["const"]+terms
    transform=np.eye(len(names))
    for i,term in enumerate(terms,1):
        if term in scaling:
            transform[i,i]=1/scaling[term]["scale"]
            transform[0,i]=-scaling[term]["mean"]/scaling[term]["scale"]
    coefficients=transform@model.params.reindex(names).to_numpy()
    tests=model.t_test(transform)
    ci=np.asarray(tests.conf_int(alpha=.05)).reshape(-1,2)
    table=pd.DataFrame({"term":names,"coefficient":coefficients,
        "standard_error":np.asarray(tests.sd).ravel(),"p_value":np.asarray(tests.pvalue).ravel(),
        "ci_95_lower":ci[:,0],"ci_95_upper":ci[:,1]})
    return dict(zip(names,map(float,coefficients))),table


def ols(frame, terms, config):
    scaling=config.get("_feature_scaling",{})
    if config.get("standardize_regression_covariates",False) and not scaling:
        scaling=fit_feature_scaling(frame,terms)
    design = sm.add_constant(scaled_design(frame,terms,scaling), has_constant="add")
    estimator = sm.OLS(frame[TARGET].astype(float), design)
    if config["regression_standard_errors"] == "event_clustered":
        fitted=estimator.fit(cov_type="cluster", cov_kwds={"groups": frame.sample_id.to_numpy()}, use_t=True)
    elif config["regression_standard_errors"] == "classical":
        fitted=estimator.fit()
    else:
        raise ValueError("regression_standard_errors must be event_clustered or classical.")
    fitted.strk_feature_scaling={term:scaling[term] for term in terms if term in scaling}
    return fitted


def stepwise(frame, config):
    candidates=[term for term in FEATURES if term!="elevation_m" or config.get("include_elevation",False)]
    config=dict(config)
    if config.get("standardize_regression_covariates",False):
        # Fit once to THIS training fold, never to its held-out events.
        config["_feature_scaling"]=fit_feature_scaling(frame,candidates)
    selected, history, visited = [], [], set()
    alpha = config["significance_level"]
    for iteration in range(1, 101):
        state = tuple(sorted(selected))
        if state in visited:
            break
        visited.add(state)
        changed = False
        tests = []
        for term in candidates:
            if term not in selected:
                p = float(ols(frame, selected+[term], config).pvalues.get(term, np.nan))
                if np.isfinite(p):
                    tests.append((p, term))
        if tests and min(tests)[0] < alpha:
            p, term = min(tests)
            selected.append(term)
            history.append({"iteration": iteration, "action": "add", "term": term, "p_value": p})
            changed = True
        while selected:
            fitted = ols(frame, selected, config)
            values = fitted.pvalues.reindex(selected).fillna(np.inf)
            worst = values.idxmax()
            if values[worst] < alpha:
                break
            selected.remove(worst)
            history.append({"iteration": iteration, "action": "remove", "term": worst, "p_value": float(values[worst])})
            changed = True
        if not changed:
            break
    else:
        raise RuntimeError("Stepwise selection did not terminate within 100 iterations.")
    fitted = ols(frame, selected, config)
    if not (fitted.pvalues.reindex(selected).fillna(np.inf) < alpha).all():
        raise RuntimeError("A retained non-intercept term fails the configured significance level.")
    return fitted, selected, pd.DataFrame(history, columns=["iteration", "action", "term", "p_value"])


def correlation(distance, lag, family, space_range, time_range):
    d, u = np.asarray(distance)/space_range, np.asarray(lag)/time_range
    if family == "exponential":
        return np.exp(-d)*np.exp(-u)
    if family == "gaussian":
        return np.exp(-d*d)*np.exp(-u*u)
    if family == "spherical":
        def sph(x):
            clipped = np.minimum(x, 1.)
            return np.maximum(1-1.5*clipped+.5*clipped**3, 0.)
        return sph(d)*sph(u)
    raise ValueError(family)


def covariance(distance, lag, params):
    return params["partial_sill"]*correlation(distance, lag, params["model_family"],
                                            params["space_range_km"], params["time_range_hours"])


def semivariance(distance, lag, params):
    value = params["nugget"]+params["partial_sill"]-covariance(distance, lag, params)
    return np.where((np.asarray(distance)==0)&(np.asarray(lag)==0), 0., value)


def sample_empirical(frame, model, terms, config, seed):
    # Frames are complete event blocks in saved [event,hour,station] order.
    sizes = frame.groupby("sample_id", sort=False).size()
    if not sizes.eq(420).all():
        raise ValueError("Every fitting event must retain all 7x60 completed RST inputs.")
    residuals = (frame[TARGET].to_numpy()-predict_trend(model, terms, frame)).reshape(-1,7,60)
    xy = frame.iloc[:60][["x_km", "y_km"]].to_numpy()
    spatial_distances = cdist(xy, xy)
    rng = np.random.default_rng(seed)
    budget = max(1, int(np.ceil(config["pair_sample_count"]/7)))
    rows = []
    for lag in range(7):
        event = rng.integers(len(residuals), size=budget)
        left_hour = rng.integers(7-lag, size=budget)
        first = rng.integers(60, size=budget)
        second = (first+rng.integers(1,60,size=budget))%60
        if lag:
            same = rng.random(budget)<config["same_station_pair_fraction"]
            second[same] = first[same]
        d = spatial_distances[first, second]
        gamma = .5*np.square(residuals[event,left_hour,first]-residuals[event,left_hour+lag,second])
        keep = d<=config["max_spatial_lag_km"]
        rows.append(pd.DataFrame({"distance_km":d[keep], "lag_hours":lag, "semivariance":gamma[keep]}))
    pairs = pd.concat(rows, ignore_index=True)
    pairs["space_bin"] = np.where(pairs.distance_km.eq(0), -1, np.floor(pairs.distance_km/config["spatial_bin_km"])).astype(int)
    empirical = pairs.groupby(["space_bin","lag_hours"], as_index=False).agg(
        distance_km=("distance_km","mean"), semivariance=("semivariance","mean"), pair_count=("semivariance","size"))
    empirical = empirical[empirical.pair_count>=config["min_pairs_per_bin"]].reset_index(drop=True)
    if len(empirical)<8 or empirical.lag_hours.nunique()<2:
        raise ValueError("Insufficient empirical bins for spatiotemporal fitting.")
    return empirical, float(np.var(residuals)), len(pairs)


def fit_candidates(empirical, residual_variance):
    d, u, y = [empirical[c].to_numpy(float) for c in ["distance_km","lag_hours","semivariance"]]
    weights = np.sqrt(empirical.pair_count.to_numpy()/empirical.pair_count.mean())
    variance = max(residual_variance, 1e-6)
    initial = [.05*variance, .95*variance, 150., 3.]
    lower = [0., 1e-10, 1., .1]
    upper = [5*variance+1, 10*variance+1, 2000., 48.]
    rows = []
    for family in FAMILIES:
        def pack(p):
            return dict(zip(["nugget","partial_sill","space_range_km","time_range_hours"], map(float,p)), model_family=family)
        result = least_squares(lambda p:(semivariance(d,u,pack(p))-y)*weights,
                               initial, bounds=(lower,upper), max_nfev=10000)
        if not result.success:
            raise RuntimeError(f"Variogram fit failed for {family}: {result.message}")
        parameters = pack(result.x)
        parameters.update(covariance_form="separable_product", fit_success=bool(result.success),
            weighted_fit_rmse=float(np.sqrt(np.average(np.square(semivariance(d,u,parameters)-y),weights=np.square(weights)))),
            nugget_to_total_sill=float(result.x[0]/(result.x[0]+result.x[1])),
            parameter_at_bound=bool(np.any(np.isclose(result.x,lower,atol=1e-5)) or np.any(np.isclose(result.x,upper,atol=1e-5))))
        rows.append(parameters)
    return rows


def kriging_geometry(window, target_position, config):
    target = window.iloc[target_position]
    keep = window.rwis_id.ne(target.rwis_id)
    if config["cv_observed_neighbors_only"]:
        keep &= ~window.rst_is_imputed
    candidates = window.loc[keep]
    d = np.hypot(candidates.x_km-target.x_km, candidates.y_km-target.y_km).to_numpy()
    t = np.abs((candidates.hour_utc-target.hour_utc).dt.total_seconds().to_numpy()/3600)
    order = np.lexsort((t,d))[:config["neighbor_count"]]
    chosen = candidates.iloc[order]
    xy = chosen[["x_km","y_km"]].to_numpy()
    hours = chosen.hour_utc.astype("int64").to_numpy()/3.6e12
    return {"indices":chosen.index.to_numpy(), "distances":cdist(xy,xy),
            "lags":np.abs(hours[:,None]-hours[None,:]), "target_distances":d[order], "target_lags":t[order]}


def residual_nowcast(residuals, geometry, params, ridge):
    indices = geometry["indices"]
    if not len(indices):
        return 0., params["nugget"]+params["partial_sill"]
    matrix = covariance(geometry["distances"],geometry["lags"],params)
    matrix.flat[::len(matrix)+1] += params["nugget"]+ridge
    vector = covariance(geometry["target_distances"],geometry["target_lags"],params)
    try:
        weights = cho_solve(cho_factor(matrix,lower=True,check_finite=False),vector,check_finite=False)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(matrix,vector,rcond=None)[0]
    return float(weights@residuals[indices]), float(max(params["nugget"]+params["partial_sill"]-vector@weights,0.))


def select_cv_events(events, config, cluster):
    n = len(events)
    percent = config["cv_event_sample_percent"]
    requested = n if percent is None else int(np.ceil(n*float(percent)/100))
    requested = max(requested,int(config["min_cv_events_per_cluster"]))
    if config["max_cv_events_per_cluster"] is not None:
        requested = min(requested,int(config["max_cv_events_per_cluster"]))
    requested = min(requested,n)
    if requested<2:
        raise ValueError("At least two CV events are required.")
    rng = np.random.default_rng(config["random_seed"]+cluster)
    shuffled = rng.permutation(events.sample_id.to_numpy())[:requested]
    folds = min(int(config["cv_folds"]),requested)
    if folds<2:
        raise ValueError("At least two CV folds are required.")
    return pd.DataFrame({"sample_id":shuffled, "fold":np.arange(requested)%folds})


def evaluate_cv(frame, events, config, cluster):
    sampling = select_cv_events(events,config,cluster)
    records, fold_rows, coefficients = [], [], []
    for fold, holdout in sampling.groupby("fold",sort=True):
        hold_ids = holdout.sample_id.to_numpy()
        training = frame[~frame.sample_id.isin(hold_ids)].reset_index(drop=True)
        if training.sample_id.nunique()<3:
            raise ValueError("Insufficient training events in a CV fold.")
        regression, terms, _ = stepwise(training,config)
        empirical, variance, pair_count = sample_empirical(training,regression,terms,config,config["random_seed"]+cluster*100+int(fold))
        candidates = fit_candidates(empirical,variance)
        original_coefficients,_=original_unit_coefficients(regression,terms)
        for term in regression.params.index:
            scaling=getattr(regression,"strk_feature_scaling",{}).get(term,{"mean":0.,"scale":1.})
            coefficients.append({"fold":fold,"term":term,"coefficient":regression.params[term],
                "p_value":regression.pvalues[term],"training_mean":scaling["mean"],"training_sd":scaling["scale"],
                "coefficient_original_units":original_coefficients[term]})
        for params in candidates:
            fold_rows.append({"fold":fold,"training_events":training.sample_id.nunique(),"held_out_events":len(holdout),
                              "pair_count":pair_count,"selected_terms":";".join(terms),**params})
        for sample_id in hold_ids:
            window = frame[frame.sample_id.eq(sample_id)].reset_index(drop=True)
            assert window.hour_utc.max()==window.issue_time_utc.iloc[0]
            trend = predict_trend(regression,terms,window)
            residuals = window[TARGET].to_numpy()-trend
            targets = window.hour_offset.eq(0)
            if config["cv_observed_targets_only"]:
                targets &= ~window.rst_is_imputed
            for position in np.flatnonzero(targets):
                target = window.iloc[position]
                geometry = kriging_geometry(window,position,config)
                for params in candidates:
                    correction, variance = residual_nowcast(residuals,geometry,params,config["kriging_ridge"])
                    records.append({"cluster":cluster,"fold":fold,"sample_id":int(sample_id),
                        "issue_time_utc":target.issue_time_utc,"target_hour_utc":target.hour_utc,"horizon_h":0,
                        "rwis_id":target.rwis_id,"model_family":params["model_family"],
                        "actual_c":target[TARGET],"target_is_imputed":bool(target.rst_is_imputed),
                        "regression_prediction_c":trend[position],"strk_prediction_c":trend[position]+correction,
                        "residual_kriging_variance_c2":variance,"neighbor_count":len(geometry["indices"])})
        print(f"  Cluster {cluster}, CV fold {fold+1}: {len(holdout)} held-out events",flush=True)
    predictions = pd.DataFrame(records)
    if predictions.empty or set(predictions.horizon_h)!={0}:
        raise RuntimeError("Expected nowcast-only CV predictions.")
    comparison = []
    for family, group in predictions.groupby("model_family"):
        comparison.append({"cluster":cluster,"model_family":family,**metrics(group.actual_c,group.strk_prediction_c),
                           **{f"regression_{key}":value for key,value in metrics(group.actual_c,group.regression_prediction_c).items() if key!="n"}})
    comparison = pd.DataFrame(comparison).sort_values(["rmse_c","mae_c","model_family"])
    if comparison.n.nunique()!=1 or set(comparison.model_family)!=set(FAMILIES):
        raise RuntimeError("Families were not evaluated on identical targets.")
    return comparison,predictions,sampling,pd.DataFrame(fold_rows),pd.DataFrame(coefficients)


def load_inputs(config):
    directory = Path(config["cluster_directory"])
    assigned = pd.read_csv(directory/"5_v2_cluster_assignments_all_samples.csv.gz")
    source = json.loads((directory/"5_v2_effective_config.json").read_text())
    index = pd.read_csv(source["input_index"])
    with np.load(source["input_archive"],allow_pickle=False) as handle:
        arrays = {key:handle[key] for key in ["X","imputation_mask","station_ids","variables","hour_offsets","sample_hour_utc"]}
    if arrays["X"].shape[1:]!=(7,60,5) or not np.array_equal(arrays["hour_offsets"],np.arange(-6,1)) or str(arrays["variables"][0])!=TARGET:
        raise ValueError("Expected Step 4 V2 seven-hour, 60-station, five-variable samples.")
    if not np.isfinite(arrays["X"]).all() or arrays["imputation_mask"].shape!=arrays["X"].shape:
        raise ValueError("Invalid fixed completed input.")
    if len(index)!=len(arrays["X"]) or assigned.sample_id.duplicated().any() or index.sample_id.duplicated().any():
        raise ValueError("Duplicate source/assignment sample IDs.")
    assigned["source_position"] = pd.Index(index.sample_id).get_indexer(assigned.sample_id)
    if (assigned.source_position<0).any():
        raise ValueError("Assignments not present in the source archive.")
    times = pd.to_datetime(arrays["sample_hour_utc"][assigned.source_position],utc=True)
    if not np.array_equal(times.to_numpy(),pd.to_datetime(assigned.sample_hour_utc,utc=True).to_numpy()):
        raise ValueError("Source/assignment timestamps disagree.")
    fitting = assigned[assigned.used_for_kmeans_fit].copy()
    if fitting.overlaps_protected_period.any() or fitting.is_cv_test_event.any():
        raise ValueError("Protected test samples cannot enter Step 7 fitting.")
    all_times = pd.to_datetime(fitting.sample_hour_utc,utc=True).sort_values()
    if not (all_times.diff().dropna()>=pd.Timedelta(hours=7)).all():
        raise ValueError("Step 7 expects the non-overlapping fitting subset selected by Step 5 V2.")
    protected = pd.read_csv(directory/"5_v2_protected_periods.csv")
    for row in protected.itertuples(index=False):
        left,right = pd.to_datetime([row.protected_start_utc,row.protected_end_utc],utc=True)
        if ((all_times>=left)&((all_times-pd.Timedelta(hours=6))<=right)).any():
            raise ValueError("A fitting history overlaps a protected period.")
    stations = pd.read_csv(config["station_file"]).set_index("rwis_id").reindex(arrays["station_ids"].astype(str))
    if config.get("include_elevation",False):
        elevation_path=Path(config["station_elevation_file"])
        if not elevation_path.is_absolute():
            elevation_path=ROOT/elevation_path
        if not elevation_path.exists():
            raise FileNotFoundError(f"Run decision_making/0c_add_elevation.py first: {elevation_path}")
        elevation=pd.read_csv(elevation_path)
        if elevation.rwis_id.duplicated().any():
            raise ValueError("Duplicate elevation station IDs")
        elevation=elevation.set_index("rwis_id").reindex(stations.index)
        if not np.isfinite(elevation.elevation_m.to_numpy()).all():
            raise ValueError("Missing RWIS elevation; complete Step 0c first")
        if not np.allclose(elevation[["latitude","longitude"]],stations[["latitude","longitude"]],atol=1e-6,rtol=0):
            raise ValueError("Elevation station coordinates do not match the training station locations")
        for column in elevation.columns:
            if column not in stations.columns:
                stations[column]=elevation[column]
    if not np.isfinite(stations[["latitude","longitude"]].to_numpy()).all():
        raise ValueError("Missing station coordinates.")
    projection = {"origin_latitude":float(stations.latitude.median()),"origin_longitude":float(stations.longitude.median()),
                  "units":"kilometres","method":"local_equirectangular"}
    return arrays,fitting,stations,projection


def event_frame(arrays,events,stations,projection,helpers):
    positions = events.source_position.to_numpy(int)
    issue = pd.DatetimeIndex(pd.to_datetime(events.sample_hour_utc,utc=True)).tz_localize(None)
    time = (issue.to_numpy()[:,None]+arrays["hour_offsets"][None,:].astype("timedelta64[h]")).astype("datetime64[ns]")
    n = len(events)
    frame = pd.DataFrame({"sample_id":np.repeat(events.sample_id.to_numpy(),420),
        "issue_time_utc":np.repeat(issue.to_numpy(),420),"hour_utc":np.repeat(time.ravel(),60),
        "hour_offset":np.tile(np.repeat(arrays["hour_offsets"],60),n),
        "rwis_id":np.tile(arrays["station_ids"].astype(str),n*7),
        "latitude":np.tile(stations.latitude.to_numpy(),n*7),"longitude":np.tile(stations.longitude.to_numpy(),n*7),
        TARGET:arrays["X"][positions,:,:,0].ravel(),"rst_is_imputed":arrays["imputation_mask"][positions,:,:,0].ravel()})
    frame["hour_local"] = pd.to_datetime(frame.hour_utc,utc=True).dt.tz_convert("America/Chicago").dt.tz_localize(None)
    if "elevation_m" in stations:
        frame["elevation_m"]=np.tile(stations.elevation_m.to_numpy(float),n*7)
    if frame.duplicated(["hour_utc","rwis_id"]).any():
        raise ValueError("Duplicate station-hour training observations.")
    return helpers.add_features(frame,projection)


def figures(empirical,params,predictions,cluster,output,config):
    d = np.linspace(0,empirical.distance_km.max(),55)
    u = np.linspace(0,6,35)
    xx,yy = np.meshgrid(d,u)
    zz = semivariance(xx,yy,params)
    fig = plt.figure(figsize=(8.5,6.5),constrained_layout=True)
    ax = fig.add_subplot(111,projection="3d")
    surface = ax.plot_surface(xx,yy,zz,cmap="viridis",alpha=.7,linewidth=0)
    ax.scatter(empirical.distance_km,empirical.lag_hours,empirical.semivariance,color=".2",alpha=.65,s=12,label="Empirical bins")
    ax.view_init(elev=25,azim=-135)
    ax.set(xlabel="Spatial lag (km)",ylabel="Temporal lag (hours)",zlabel=r"$\gamma(h,u)$",
           title=f"Cluster {cluster}: {params['model_family'].capitalize()} residual semivariogram")
    ax.zaxis.labelpad = 4
    ax.legend(frameon=False)
    fig.colorbar(surface,ax=ax,shrink=.65,pad=.1,label=r"Semivariance ($^\circ$C$^2$)")
    fig.savefig(output/"7_v2_selected_spatiotemporal_variogram_3d.png",dpi=config["plot_dpi"],bbox_inches="tight")
    fig.savefig(output/"7_v2_selected_spatiotemporal_variogram_3d.pdf",bbox_inches="tight")
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    image = axes[0].contourf(xx,yy,zz,levels=18,cmap="viridis")
    axes[0].scatter(empirical.distance_km,empirical.lag_hours,s=8,c="white",alpha=.7)
    axes[0].set(xlabel="Spatial lag (km)",ylabel="Temporal lag (hours)",title="Selected covariance model")
    fig.colorbar(image,ax=axes[0],label="Semivariance")
    axes[1].scatter(predictions.actual_c,predictions.strk_prediction_c,s=8,alpha=.3,color="#355f7c")
    lo = min(predictions.actual_c.min(),predictions.strk_prediction_c.min())
    hi = max(predictions.actual_c.max(),predictions.strk_prediction_c.max())
    axes[1].plot([lo,hi],[lo,hi],"--",color=".4")
    axes[1].set(xlabel="Observed RST (degrees C)",ylabel="CV nowcast (degrees C)",title="Held-out development events")
    fig.savefig(output/"7_v2_variogram_and_cv_nowcasts.png",dpi=config["plot_dpi"])
    fig.savefig(output/"7_v2_variogram_and_cv_nowcasts.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"7_v2_cluster_strk_config.json")
    parser.add_argument("--output-directory",type=Path)
    parser.add_argument("--clusters",nargs="+",type=int)
    parser.add_argument("--max-fit-events-per-cluster",type=int,help="Explicit reduced-run limit; default is all Step 5 fitting events.")
    parser.add_argument("--cv-events",type=int,help="Override minimum and maximum CV events for a small verification run.")
    parser.add_argument("--pair-sample-count",type=int)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--overwrite",action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.max_fit_events_per_cluster is not None:
        config["max_fit_events_per_cluster"] = args.max_fit_events_per_cluster
    if args.cv_events is not None:
        config["min_cv_events_per_cluster"] = config["max_cv_events_per_cluster"] = args.cv_events
    if args.pair_sample_count is not None:
        config["pair_sample_count"] = args.pair_sample_count
    for key in ["cluster_directory","station_file"]:
        config[key] = str((ROOT/config[key]).resolve())
    if config.get("include_elevation",False):
        config["station_elevation_file"]=str((ROOT/config["station_elevation_file"]).resolve())
    if config["max_temporal_lag_hours"]!=6 or config["minimum_fit_events"]<4:
        raise ValueError("This implementation uses exactly seven input hours and at least four training events.")
    if config["min_cv_events_per_cluster"]<2 or (config["max_cv_events_per_cluster"] is not None and config["max_cv_events_per_cluster"]<config["min_cv_events_per_cluster"]):
        raise ValueError("Require 2 <= min CV events <= max CV events (unless max is null).")
    if config["cv_event_sample_percent"] is not None and not 0<=config["cv_event_sample_percent"]<=100:
        raise ValueError("CV percentage must be between 0 and 100, or null.")
    if config["neighbor_count"]<1 or config["cv_folds"]<2 or not 0<=config["same_station_pair_fraction"]<=1:
        raise ValueError("Invalid neighbour count, CV fold count or same-station pair fraction.")
    helpers = original_helpers()
    output = args.output_directory or ROOT/config["output_directory"]
    output.mkdir(parents=True,exist_ok=True)
    source_config = json.loads((Path(config["cluster_directory"])/"5_v2_effective_config.json").read_text())
    source_paths = [Path(__file__), ROOT/"7_fit_cluster_specific_regression_variograms.py", Path(config["station_file"]),
        Path(source_config["input_archive"]),Path(source_config["input_index"]),
        Path(config["cluster_directory"])/"5_v2_cluster_assignments_all_samples.csv.gz",
        Path(config["cluster_directory"])/"5_v2_protected_periods.csv"]
    if config.get("include_elevation",False):
        source_paths.append(Path(config["station_elevation_file"]))
    input_hashes = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    signature = hashlib.sha256(json.dumps({"config":config,"input_hashes":input_hashes},sort_keys=True).encode()).hexdigest()
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],"font.size":10,"pdf.fonttype":42})
    with threadpool_limits(limits=config["thread_limit"]):
        arrays,assigned,stations,projection = load_inputs(config)
        available_clusters = sorted(assigned.cluster.unique().astype(int).tolist())
        clusters = args.clusters if args.clusters is not None else available_clusters
        if not set(clusters).issubset(available_clusters):
            raise ValueError("Requested cluster is absent from fitting assignments.")
        summaries,pooled,comparison_rows = [],[],[]
        for cluster in clusters:
            directory = output/f"cluster_{cluster:02d}"
            summary_path = directory/"7_v2_cluster_summary.json"
            if summary_path.exists() and not args.overwrite:
                existing = json.loads(summary_path.read_text())
                if not args.resume or existing["run_signature"]!=signature:
                    raise FileExistsError("Cluster already exists; use --resume with identical settings or --overwrite.")
                summaries.append(existing)
                pooled.append(pd.read_csv(directory/"7_v2_selected_cv_nowcast_predictions.csv.gz"))
                comparison_rows.append(pd.read_csv(directory/"7_v2_variogram_comparison.csv"))
                print(f"Reused cluster {cluster}.",flush=True)
                continue
            directory.mkdir(parents=True,exist_ok=True)
            events = assigned[assigned.cluster.eq(cluster)].copy()
            cap = config["max_fit_events_per_cluster"]
            if cap is not None and len(events)>cap:
                events = events.sample(int(cap),random_state=config["random_seed"]+cluster).sort_values("sample_hour_utc")
            if len(events)<config["minimum_fit_events"]:
                raise ValueError(f"Cluster {cluster} has too few fitting events.")
            events.to_csv(directory/"7_v2_fitting_event_audit.csv",index=False)
            frame = event_frame(arrays,events,stations,projection,helpers)
            print(f"Cluster {cluster}: {len(events)} fitting histories, {len(frame):,} completed station-hours; nowcasts only.",flush=True)
            comparison,predictions,sampling,fold_params,fold_coefficients = evaluate_cv(frame,events,config,cluster)
            best_family = str(comparison.iloc[0].model_family)
            regression,terms,history = stepwise(frame,config)
            history.to_csv(directory/"7_v2_stepwise_selection_history.csv",index=False)
            report_frame=frame.copy()
            report_frame[terms]=scaled_design(frame,terms,getattr(regression,"strk_feature_scaling",{}))
            regression_metrics = helpers.regression_exports(regression,terms,report_frame,directory)
            raw_coefficients,raw_table=original_unit_coefficients(regression,terms)
            raw_table.to_csv(directory/"7_regression_coefficients_original_units.csv",index=False)
            (directory/"7_v2_regression_scaling.json").write_text(json.dumps(
                getattr(regression,"strk_feature_scaling",{}),indent=2)+"\n")
            empirical,residual_variance,pair_count = sample_empirical(frame,regression,terms,config,config["random_seed"]+cluster*100+99)
            final_candidates = fit_candidates(empirical,residual_variance)
            params = next(p for p in final_candidates if p["model_family"]==best_family)
            comparison = comparison.merge(pd.DataFrame(final_candidates),on="model_family",validate="one_to_one")
            comparison["selected"] = comparison.model_family.eq(best_family)
            comparison["selection_scope"] = "development event-held-out nowcast CV; not independent test performance"
            comparison.to_csv(directory/"7_v2_variogram_comparison.csv",index=False)
            selected = predictions[predictions.model_family.eq(best_family)].copy()
            predictions.to_csv(directory/"7_v2_all_candidate_cv_nowcast_predictions.csv.gz",index=False)
            selected.to_csv(directory/"7_v2_selected_cv_nowcast_predictions.csv.gz",index=False)
            sampling.merge(events[["sample_id","sample_hour_utc"]],on="sample_id",validate="one_to_one").to_csv(directory/"7_v2_selected_cv_events_and_folds.csv",index=False)
            fold_params.to_csv(directory/"7_v2_fold_variogram_parameters.csv",index=False)
            fold_coefficients.to_csv(directory/"7_v2_fold_regression_coefficients.csv",index=False)
            empirical.to_csv(directory/"7_v2_empirical_spatiotemporal_variogram.csv",index=False)
            package = {"cluster":cluster,"target":TARGET,"projection":projection,"regression_model":regression,
                "regression_scaling":getattr(regression,"strk_feature_scaling",{}),
                "regression_coefficients_original_units":raw_coefficients,
                "selected_features":terms,"candidate_features":[term for term in FEATURES if term!="elevation_m" or config.get("include_elevation",False)],"variogram":params,"lookback_hours":6,
                "horizons":[0],"nowcasting_only":True,"neighbor_count":config["neighbor_count"],
                "covariance_form":"separable_product","config":config,"run_signature":signature,
                "cv_design":"Event-fold refitting followed by station-history-excluded residual nowcasting; fixed cluster assignments and Step 4 imputation."}
            joblib.dump(package,directory/"7_v2_cluster_strk_model.joblib",compress=3)
            figures(empirical,params,selected,cluster,directory,config)
            summary = {"cluster":cluster,"run_signature":signature,"fitting_event_count":len(events),
                "unique_station_hour_count":len(frame),"fitting_imputed_rst_fraction":float(frame.rst_is_imputed.mean()),
                "selected_terms":terms,"regression":regression_metrics,"selected_variogram":params,
                "selected_cv_metrics":metrics(selected.actual_c,selected.strk_prediction_c),
                "cv_event_count":len(sampling),"cv_fold_count":sampling.fold.nunique(),"sampled_residual_pair_count":pair_count,
                "nowcasting_only":True,"horizons":[0]}
            summary_path.write_text(json.dumps(summary,indent=2)+"\n")
            summaries.append(summary); pooled.append(selected); comparison_rows.append(comparison)
        all_predictions = pd.concat(pooled,ignore_index=True)
        pd.concat(comparison_rows,ignore_index=True).to_csv(output/"7_v2_variogram_comparison_all_clusters.csv",index=False)
        all_predictions.to_csv(output/"7_v2_selected_cv_nowcast_predictions_all_clusters.csv.gz",index=False)
        per_event=[]
        for (cluster,sample_id),group in all_predictions.groupby(["cluster","sample_id"]):
            per_event.append({"cluster":cluster,"sample_id":sample_id,"horizon_h":0,**metrics(group.actual_c,group.strk_prediction_c)})
        pd.DataFrame(per_event).to_csv(output/"7_v2_cv_nowcast_metrics_by_event.csv",index=False)
        per_cluster=[]
        for cluster,group in all_predictions.groupby("cluster"):
            for method,column in [("regression","regression_prediction_c"),("strk","strk_prediction_c")]:
                per_cluster.append({"cluster":cluster,"method":method,"horizon_h":0,**metrics(group.actual_c,group[column])})
        pd.DataFrame(per_cluster).to_csv(output/"7_v2_cv_nowcast_metrics_by_cluster.csv",index=False)
        pd.DataFrame([{"method":method,"horizon_h":0,**metrics(all_predictions.actual_c,all_predictions[column])}
                      for method,column in [("regression","regression_prediction_c"),("strk","strk_prediction_c")]]).to_csv(output/"7_v2_overall_cv_nowcast_metrics.csv",index=False)
        run={"config":config,"input_hashes":input_hashes,"processed_clusters":clusters,"available_clusters":available_clusters,
            "run_signature":signature,"nowcasting_only":True,"horizons":[0],
            "reduced_fit_event_run":config["max_fit_events_per_cluster"] is not None,
            "input_policy":"Fixed completed Step 4 data; Step 5 V2 fitting subset only; CV-image test events never fitted or evaluated here.",
            "caveats":"CV selects a covariance family, so selected CV metrics are not independent test estimates. Cluster labels and upstream imputation are fixed before these folds. Residual kriging variance excludes regression/parameter uncertainty."}
        (output/"7_v2_run_summary.json").write_text(json.dumps(run,indent=2)+"\n")
        print("Saved nowcasting-only models and CV summaries to:",output,flush=True)


if __name__=="__main__":
    main()
