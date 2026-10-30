import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RW = ROOT/'workflow/analysis-iowa/RW_module'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NumericalCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.strk = load('tested_strk', RW/'7_v2_fit_cluster_specific_regression_variograms_with_elevation.py')
        cls.sgs = load('tested_sgs', RW/'after_8py_RW_simulation/10_s1_sgs_road_rst_fields.py')

    def test_strk_sgs_covariance_agreement(self):
        d, t = np.array([0., 2., 30., 150.]), np.array([0., 1., 2., 6.])
        for family in ['exponential', 'gaussian', 'spherical']:
            p = dict(nugget=.05, partial_sill=1.5, space_range_km=80., time_range_hours=3., model_family=family)
            np.testing.assert_allclose(self.strk.covariance(d, t, p), self.sgs.covariance(d, t, p))

    def test_sgs_hard_conditions_reproducibility_and_variation(self):
        conditions = np.array([[0., 0.], [10., 0.]])
        roads = np.array([[0., 0.], [5., 0.], [10., 0.], [15., 0.]])
        p = dict(nugget=.05, partial_sill=1., space_range_km=15., time_range_hours=2., model_family='exponential')
        args = (conditions, np.zeros(2), np.array([-1., 2.]), roads, np.array([0.]), p, [12, 34], 99)
        first, _ = self.sgs.simulate_group(*args, neighbor_count=4)
        second, _ = self.sgs.simulate_group(*args, neighbor_count=4)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first[:, 0, 0], -1.)
        np.testing.assert_allclose(first[:, 0, 2], 2.)
        self.assertTrue(np.isfinite(first).all())
        self.assertFalse(np.allclose(first[0, 0, [1, 3]], first[1, 0, [1, 3]]))

    def test_scaling_coefficients_preserve_predictions(self):
        rng = np.random.default_rng(21)
        x = rng.normal(size=200)
        frame = pd.DataFrame(dict(x_km=x, sample_id=np.repeat(np.arange(20), 10),
                                  road_surface_temp_c_mean=2+3*x+rng.normal(0, .1, len(x))))
        model = self.strk.ols(frame, ['x_km'], {'standardize_regression_covariates': True,
                                              'regression_standard_errors': 'event_clustered'})
        original, _ = self.strk.original_unit_coefficients(model, ['x_km'])
        np.testing.assert_allclose(self.strk.predict_trend(model, ['x_km'], frame),
                                   original['const']+frame.x_km*original['x_km'])

    def test_protected_period_metadata_is_reusable(self):
        cluster = load('tested_cluster', RW/'5_v2_cluster_rst_weather_regimes_kmeans.py')
        config = json.loads((RW/'5_v2_kmeans_config.json').read_text())
        events, intervals = cluster.test_events(RW/config['visual_matches'], 6, 24)
        self.assertEqual(len(events), 9)
        self.assertEqual(int(events.test_image_records.sum()), 607)
        self.assertTrue(len(intervals) > 0)

    def test_exported_centroids_match_portable_parameters(self):
        with np.load(ROOT/'models/iowa_kmeans_parameters.npz', allow_pickle=False) as saved:
            params = {name: saved[name] for name in saved.files}
        with np.load(ROOT/'results/iowa/5_v2_multivariable_weather_regimes/5_v2_centroids.npz', allow_pickle=False) as saved:
            centroids = saved['centroids_original_units'].reshape(7, 2100)
        standardized = (centroids-params['scaler_mean'])/params['scaler_scale']
        projected = (standardized-params['pca_mean'])@params['pca_components'].T
        if bool(params['pca_whiten']):
            projected /= np.sqrt(params['pca_explained_variance'])
        distances = np.linalg.norm(projected[:, None]-params['centers'][None, :], axis=2)
        np.testing.assert_array_equal(np.argmin(distances, axis=1), np.arange(7))


if __name__ == '__main__':
    unittest.main()
