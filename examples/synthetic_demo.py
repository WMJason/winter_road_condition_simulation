"""Small synthetic demonstration using the released aggregation/STRK/SGS helpers.

This is not Iowa data and does not reproduce the paper's numerical results.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'winter_public_demo_mpl'))
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
RW = ROOT/'workflow/analysis-iowa/RW_module'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(output, seed):
    output.mkdir(parents=True, exist_ok=True)
    aggregate = load('demo_aggregation', RW/'2a_build_rwis_station_hourly_modeling_table.py')
    preparation = load('demo_preparation', RW/'4_prepare_imputed_weather_regime_samples.py')
    clustering = load('demo_clustering', RW/'5_v2_cluster_rst_weather_regimes_kmeans.py')
    strk = load('demo_strk', RW/'7_v2_fit_cluster_specific_regression_variograms_with_elevation.py')
    sgs = load('demo_sgs', RW/'after_8py_RW_simulation/10_s1_sgs_road_rst_fields.py')
    helpers = strk.original_helpers()
    rng = np.random.default_rng(seed)
    lat, lon = np.meshgrid(np.linspace(41, 43, 6), np.linspace(-95, -91, 10), indexing='ij')
    lat, lon = lat.ravel(), lon.ravel()
    station_ids = np.array([f'SYN{i:02d}' for i in range(60)])
    projection = dict(origin_latitude=float(lat.mean()), origin_longitude=float(lon.mean()))
    elevation = 280 + 50*np.sin(np.arange(60)/9)
    spatial = preparation.haversine_matrix(lat, lon)
    offsets = np.arange(-6, 1)
    d = np.tile(spatial, (7, 7))
    lag = np.abs(np.repeat(offsets, 60)[:, None]-np.repeat(offsets, 60)[None, :])
    covariance = .6*np.exp(-d/80-lag/3)+np.eye(420)*.02
    residual = (rng.normal(size=(36, 420))@cholesky(covariance, lower=True).T).reshape(36, 7, 60)
    issue = pd.date_range('2016-01-04', periods=36, freq='48h', tz='UTC')+pd.to_timedelta(np.arange(36)%12, unit='h')
    hours = pd.DatetimeIndex([t+pd.Timedelta(hours=int(h)) for t in issue for h in offsets])
    local_hour = hours.tz_convert('America/Chicago').hour.to_numpy().reshape(36, 7)
    rst = (-5+5*(np.arange(36)%3))[:, None, None]+.5*(lon-lon.mean())[None, None, :]
    rst = rst+1.2*np.sin(2*np.pi*local_hour/24)[:, :, None]+residual
    air = rst-1+rng.normal(0, .3, rst.shape)
    dew = air-3-rng.random(rst.shape)
    humidity = np.clip(72+4*(np.arange(36)%3)[:, None, None]+rng.normal(0, 2, rst.shape), 0, 100)
    wind = np.maximum(0, 8+2*(np.arange(36)%3)[:, None, None]+rng.normal(0, 1, rst.shape))
    raw = pd.DataFrame(dict(id=np.arange(36*420), rwis_id=np.tile(station_ids, 36*7),
          timestamp=np.repeat(hours.tz_localize(None).astype(str), 60), latitude=np.tile(lat, 36*7),
          longitude=np.tile(lon, 36*7), air_temp_f=air.ravel()*1.8+32,
          dewpoint_f=dew.ravel()*1.8+32, humidity=humidity.ravel(), wind_speed=wind.ravel(),
          tfs0=rst.ravel()*1.8+32, tfs1=np.nan, tfs2=np.nan, tfs3=np.nan))
    duplicate = raw.copy()
    duplicate['id'] += len(raw)
    duplicate['timestamp'] = (pd.to_datetime(duplicate.timestamp)+pd.Timedelta(minutes=30)).astype(str)
    hourly = aggregate.build_station_hourly_table(aggregate.add_time_fields(pd.concat([raw, duplicate], ignore_index=True)))
    for variable in preparation.VARIABLES:
        hourly.loc[rng.random(len(hourly)) < .01, variable] = np.nan
    timeline, observed = preparation.build_arrays(hourly, station_ids.tolist())
    climatology = preparation.training_climatology(timeline, observed)
    completed, sources = preparation.impute_values(timeline, observed, climatology, spatial)
    x = completed.reshape(36, 7, 60, 5)
    fit = np.arange(36) < 24
    cfg = dict(use_pca=True, pca_explained_variance=.95, pca_whiten=False,
               max_silhouette_samples=24, diagnostic_sampling_seed=seed, k_candidates=[2, 3, 4, 5],
               random_seeds=[seed, seed+1], n_init=10, max_iter=200,
               minimum_cluster_share=.05, selected_k=None, selection_method='elbow')
    standardized, transformed, scaler, pca = clustering.features(x, fit, cfg)
    diagnostics, candidates = clustering.fit_candidates(standardized, transformed, fit, cfg, output)
    k, elbow = clustering.choose_k(diagnostics, cfg)
    kmeans = candidates[k]
    labels = kmeans.predict(transformed)
    pd.DataFrame(dict(sample_id=np.arange(36), cluster=labels, used_for_fit=fit,
                      issue_utc=issue.astype(str))).to_csv(output/'synthetic_event_assignments.csv', index=False)
    frame = hourly[['rwis_id', 'latitude', 'longitude', 'hour_utc', 'hour_local']].copy()
    # Restore the fixed [event, hour, station] order expected by the research helpers.
    frame = frame.sort_values(['hour_utc', 'rwis_id']).reset_index(drop=True)
    frame['hour_utc'] = pd.to_datetime(frame.hour_utc, utc=True).dt.tz_localize(None)
    frame['hour_local'] = pd.to_datetime(frame.hour_utc, utc=True).dt.tz_convert('America/Chicago').dt.tz_localize(None)
    frame[preparation.VARIABLES] = completed.reshape(-1, 5)
    frame['sample_id'] = np.repeat(np.arange(36), 420)
    frame['hour_offset'] = np.tile(np.repeat(offsets, 60), 36)
    frame['rst_is_imputed'] = sources[:, :, 0].reshape(-1) != 0
    frame['elevation_m'] = np.tile(elevation, 36*7)
    frame = helpers.add_features(frame, projection)
    model_cfg = json.loads((RW/'7_v2_cluster_strk_config.json').read_text())
    model_cfg.update(pair_sample_count=3500, min_pairs_per_bin=5)
    outputs = []
    first_maps = None
    road_lat, road_lon = np.meshgrid(np.linspace(41, 43, 8), np.linspace(-95, -91, 12), indexing='ij')
    road_lat, road_lon = np.r_[lat, road_lat.ravel()], np.r_[lon, road_lon.ravel()]
    for cluster in range(k):
        fitting_ids = np.flatnonzero(fit & (labels == cluster))
        training = frame[frame.sample_id.isin(fitting_ids)].copy()
        model, terms, selection = strk.stepwise(training, model_cfg)
        empirical, variance, _ = strk.sample_empirical(training, model, terms, model_cfg, seed+cluster)
        variants = strk.fit_candidates(empirical, variance)
        # The small demo ranks by fit RMSE; the Iowa workflow uses event/station holdouts.
        params = min(variants, key=lambda item: item['weighted_fit_rmse'])
        pd.DataFrame(variants).to_csv(output/f'synthetic_cluster_{cluster}_variograms.csv', index=False)
        coefficients, coefficient_table = strk.original_unit_coefficients(model, terms)
        coefficient_table.to_csv(output/f'synthetic_cluster_{cluster}_regression.csv', index=False)
        center = kmeans.cluster_centers_[cluster]
        parent = int(fitting_ids[np.argmin(np.linalg.norm(transformed[fitting_ids]-center, axis=1))])
        window = frame[frame.sample_id.eq(parent)].copy().reset_index(drop=True)
        trend = strk.predict_trend(model, terms, window)
        cond = window[strk.TARGET].to_numpy()-trend
        coords = window[['x_km', 'y_km']].to_numpy()
        geometry = strk.kriging_geometry(window, 360, {**model_cfg, 'cv_observed_neighbors_only': False})
        correction, _ = strk.residual_nowcast(cond, geometry, params, 1e-8)
        sample_prediction = float(trend[360]+correction)
        roads = sgs.feature_frame(road_lat, road_lon, [issue[parent]]*len(road_lat), projection, helpers,
                                 np.r_[elevation, np.full(len(road_lat)-60, elevation.mean())])
        road_xy = roads[['x_km', 'y_km']].to_numpy()
        issue_window = window.loc[window.hour_offset.eq(0)]
        for term in terms:
            np.testing.assert_allclose(roads[term].to_numpy()[:60], issue_window[term].to_numpy(),
                                       err_msg=f'Inconsistent conditioning/road feature: {term}')
        fields, info = sgs.simulate_group(coords, window.hour_offset.to_numpy(float), cond, road_xy,
                          np.array([0.]), params, [seed+cluster*10+i for i in range(3)],
                          seed+100+cluster, neighbor_count=40)
        fields = fields[:, 0]+sum(roads[term].to_numpy()*coefficients[term] for term in terms)+coefficients['const']
        np.testing.assert_allclose(fields[:, :60], np.broadcast_to(issue_window[strk.TARGET].to_numpy(), (3, 60)), atol=1e-7)
        for realization, field in enumerate(fields):
            outputs.append(dict(cluster=cluster, parent_sample_id=parent, realization=realization+1,
                                mean_rst_c=float(field.mean()), sd_rst_c=float(field.std()),
                                min_rst_c=float(field.min()), max_rst_c=float(field.max()),
                                conditioning_max_error_c=float(np.abs(field[:60]-window.loc[window.hour_offset.eq(0), strk.TARGET]).max()),
                                example_station_holdout_prediction_c=sample_prediction))
        if first_maps is None:
            first_maps = fields
        np.savez_compressed(output/f'synthetic_cluster_{cluster}_sgs.npz', rst_c=fields.astype('float32'),
                            latitude=road_lat, longitude=road_lon)
    pd.DataFrame(outputs).to_csv(output/'synthetic_realization_statistics.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True, layout='constrained')
    for i, ax in enumerate(axes):
        im = ax.scatter(road_lon, road_lat, c=first_maps[i], s=12, cmap='coolwarm',
                        vmin=float(first_maps.min()), vmax=float(first_maps.max()))
        ax.set(title=f'Synthetic realization {i+1}', xlabel='Longitude')
    axes[0].set_ylabel('Latitude')
    fig.colorbar(im, ax=axes, label='RST (degrees C)')
    fig.savefig(output/'synthetic_sgs_examples.png', dpi=180)
    plt.close(fig)
    report = dict(synthetic=True, reproduces_iowa_results=False, seed=seed, history_shape=[7, 60, 5],
                  sample_count=36, fitting_samples=int(fit.sum()), selected_k=k, elbow_k=elbow,
                  imputed_cells=int((sources != 0).sum()), raw_records=len(raw)*2, hourly_records=len(hourly),
                  retained_pca_components=int(pca.n_components_), realization_count=len(outputs),
                  conditioning_max_error_c=max(row['conditioning_max_error_c'] for row in outputs),
                  covariance_selection='Demo uses empirical-fit RMSE, not the full Iowa event-fold selection procedure')
    (output/'demo_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'examples/generated')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output.resolve(), args.seed)


if __name__ == '__main__':
    main()
