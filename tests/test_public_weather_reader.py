"""Small self-contained reader tests; do not load the full dataset in CI."""
import csv
import io
import tempfile
import unittest
from pathlib import Path
import zipfile

import numpy as np

from tools.read_weather_events import read_scenario


class PublicWeatherReaderTest(unittest.TestCase):
    def test_reconstruction_and_realization_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            buffer = io.BytesIO()
            np.savez_compressed(buffer, hour_offsets=np.array([0]),
                                trend_c=np.array([[1., 2.]], dtype='float32'),
                                residual_c=np.array([[[.5, -.5]], [[2., 3.]]], dtype='float32'),
                                drivability=np.array([[0, 1], [2, 1]], dtype='uint8'),
                                maintenance_need=np.array([[1, 0], [2, 2]], dtype='uint8'),
                                low_natural_illumination=np.array([0, 1], dtype='uint8'),
                                solar_altitude_degrees=np.array([1., -1.], dtype='float32'))
            with zipfile.ZipFile(root / 'part.zip', 'w') as archive:
                archive.writestr('events/parent.npz', buffer.getvalue())
            with (root / 'scenario_manifest.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['scenario_id', 'realization_index', 'archive', 'member'])
                writer.writeheader()
                writer.writerow(dict(scenario_id='parent__SGS02', realization_index=1,
                                     archive='part.zip', member='events/parent.npz'))
            result = read_scenario('parent__SGS02', root)
            np.testing.assert_array_equal(result['rst_c'], [3., 5.])
            np.testing.assert_array_equal(result['drivability'], [2, 1])
            self.assertEqual(result['rst_c'].dtype, np.dtype('float32'))
            with self.assertRaises(ValueError):
                read_scenario('missing', root)


if __name__ == '__main__':
    unittest.main()
