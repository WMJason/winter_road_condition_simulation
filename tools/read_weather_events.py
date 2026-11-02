"""Read public reference fields directly from downloaded ZIP archives."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

DEFAULT_DATA = Path(__file__).resolve().parents[1] / 'data' / 'weather_events'


def read_npz(archive_path, member):
    with zipfile.ZipFile(archive_path) as archive:
        with np.load(io.BytesIO(archive.read(member)), allow_pickle=False) as arrays:
            return {key: arrays[key] for key in arrays.files}


def read_scenario(scenario_id, directory=DEFAULT_DATA):
    directory = Path(directory)
    with (directory / 'scenario_manifest.csv').open(newline='', encoding='utf-8') as stream:
        matches = [row for row in csv.DictReader(stream) if row['scenario_id'] == scenario_id]
    if len(matches) != 1:
        raise ValueError(f'Expected one matching scenario, found {len(matches)}: {scenario_id}')
    metadata = matches[0]
    parent = read_npz(directory / metadata['archive'], metadata['member'])
    i = int(metadata['realization_index'])
    h = int(np.flatnonzero(parent['hour_offsets'] == 0)[0])
    return {'metadata': metadata,
            'rst_c': parent['trend_c'][h] + parent['residual_c'][i, h],
            'drivability': parent['drivability'][i],
            'maintenance_need': parent['maintenance_need'][i],
            'low_natural_illumination': parent['low_natural_illumination'],
            'solar_altitude_degrees': parent['solar_altitude_degrees']}


def verify(directory=DEFAULT_DATA):
    directory = Path(directory)
    summary = json.loads((directory / 'dataset_summary.json').read_text())
    for item in summary['archives']:
        path = directory / item['file']
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024**2), b''):
                digest.update(block)
        if path.stat().st_size != item['bytes'] or digest.hexdigest() != item['sha256']:
            raise ValueError(f'Missing or changed archive: {path}')
    with (directory / 'scenario_manifest.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 700 or len({r['scenario_id'] for r in rows}) != 700:
        raise ValueError('Expected 700 unique scenarios')
    support = read_npz(directory / 'weather_support.zip', 'support_points.npz')
    if len(support['latitude']) != summary['support_points']:
        raise ValueError('Support-point count differs from the dataset summary')
    for cluster in range(7):
        if sum(int(row['cluster']) == cluster for row in rows) != 100:
            raise ValueError(f'Expected 100 realizations in group {cluster}')
    with (directory / 'source_checksums.csv').open(newline='') as stream:
        sources = {row['parent_event_id']: row for row in csv.DictReader(stream)}
    if set(sources) != {row['parent_event_id'] for row in rows}:
        raise ValueError('Source-checksum parent identifiers differ')
    for event in sorted({r['parent_event_id'] for r in rows}):
        group = [r for r in rows if r['parent_event_id'] == event]
        if len(group) != 10 or {int(r['realization_index']) for r in group} != set(range(10)):
            raise ValueError(f'Incomplete parent: {event}')
        with zipfile.ZipFile(directory / group[0]['archive']) as archive:
            payload = archive.read(group[0]['member'])
            metadata = json.loads(archive.read(f'events/{event}.json'))
        if hashlib.sha256(payload).hexdigest() != sources[event]['npz_sha256']:
            raise ValueError(f'Parent checksum differs: {event}')
        for row in group:
            if row['archive'] != group[0]['archive'] or row['member'] != group[0]['member']:
                raise ValueError(f'Parent archive mapping differs: {event}')
            if row['scenario_id'] != f'{event}__SGS{int(row["realization_index"]) + 1:02d}':
                raise ValueError(f'Scenario identifier differs: {event}')
            for key, value in metadata.items():
                if str(value) != row[key]:
                    raise ValueError(f'Parent metadata differs: {event}, {key}')
        with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
            parent = {key: loaded[key] for key in loaded.files}
        n = len(support['latitude'])
        if parent['residual_c'].shape != (10, 1, n):
            raise ValueError(f'Invalid RST dimensions: {event}')
        if parent['trend_c'].shape != (1, n) or not np.array_equal(parent['hour_offsets'], [0]):
            raise ValueError(f'Invalid trend or issue-time dimensions: {event}')
        if not np.isfinite(parent['residual_c']).all() or not np.isfinite(parent['trend_c']).all():
            raise ValueError(f'Nonfinite RST arrays: {event}')
        for field in ('drivability', 'maintenance_need'):
            if parent[field].shape != (10, n) or not np.isin(parent[field], [0, 1, 2]).all():
                raise ValueError(f'Invalid visual categories: {event}')
    print(f'Validated {len(rows)} scenarios, 70 parents, {len(support["latitude"])} support points; all archive hashes match.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--scenario', default='C00_R01_S5447__SGS01')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        verify(args.data)
    else:
        result = read_scenario(args.scenario, args.data)
        print(json.dumps(result['metadata'], indent=2))
        print(f'RST: {len(result["rst_c"])} support points; mean {result["rst_c"].mean():.3f} degrees C')
        for field in ('drivability', 'maintenance_need'):
            print(field, dict(zip(*np.unique(result[field], return_counts=True))))
