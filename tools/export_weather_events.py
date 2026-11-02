"""Export exact saved reference fields into browser-uploadable data archives."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 24 * 1024**2


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            result.update(block)
    return result.hexdigest()


def arrays_bytes(arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def export(source, output):
    rst_root = source / '10_sgs_completed_rwis_fields_elevation'
    visual_root = source / '11_visual_ground_truth_outputs'
    with (rst_root / '10_sgs_manifest.csv').open(newline='') as stream:
        parents = list(csv.DictReader(stream))
    with (visual_root / '11_visual_manifest.csv').open(newline='') as stream:
        visual_rows = {row['event_id']: row for row in csv.DictReader(stream)}
    if len(parents) != 70 or len({p['event_id'] for p in parents}) != 70:
        raise ValueError('Expected exactly 70 distinct parent events')
    if {p['event_id'] for p in parents} != set(visual_rows):
        raise ValueError('RST and visual parent identifiers differ')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory; existing releases are never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    catalog_path = rst_root / '10_road_support_catalog.h5'
    with h5py.File(catalog_path, 'r') as catalog:
        support = {key: catalog[key][:] for key in catalog}
    n = len(support['latitude'])
    if n != 122329:
        raise ValueError('Unexpected support-point count')
    # Visual arrays must use exactly the same support-point order as the RST arrays.
    with h5py.File(visual_root / '11_visual_support_catalog.h5', 'r') as catalog:
        if not np.array_equal(catalog['sgs_support_index'][:], np.arange(n)):
            raise ValueError('Visual support order differs')
        for key in ('latitude', 'longitude'):
            if not np.array_equal(catalog[key][:], support[key]):
                raise ValueError(f'Visual catalog differs: {key}')
    with zipfile.ZipFile(output / 'weather_support.zip', 'w', compression=zipfile.ZIP_STORED) as archive:
        archive.writestr('support_points.npz', arrays_bytes(support))
    rows, sources, archive = [], [], None
    part = 0
    try:
        for parent in sorted(parents, key=lambda x: x['event_id']):
            event = parent['event_id']
            rst_path = rst_root / 'events' / event / '10_sgs_realizations.h5'
            visual_path = visual_root / 'events' / event / '11_visual_ground_truth.h5'
            rst_hash, visual_hash = sha256(rst_path), sha256(visual_path)
            with h5py.File(rst_path, 'r') as rst, h5py.File(visual_path, 'r') as visual:
                if visual.attrs['source_rst_sha256'] != rst_hash:
                    raise ValueError(f'{event}: visual conditions were generated from a different RST file')
                if not visual.attrs.get('complete', False):
                    raise ValueError(f'{event}: incomplete visual output')
                for key in ('event_id', 'cluster', 'issue_time_utc'):
                    if rst.attrs[key] != visual.attrs[key]:
                        raise ValueError(f'{event}: RST/visual metadata differ: {key}')
                if str(rst.attrs['event_id']) != event or int(rst.attrs['cluster']) != int(parent['cluster']):
                    raise ValueError(f'{event}: manifest identity differs')
                if str(visual.attrs['signature']) != visual_rows[event]['signature']:
                    raise ValueError(f'{event}: visual manifest signature differs')
                arrays = {key: rst[key][:] for key in ('hour_offsets', 'trend_c', 'residual_c')}
                arrays.update({key: visual[key][:] for key in ('drivability', 'maintenance_need',
                                                              'low_natural_illumination', 'solar_altitude_degrees')})
                if arrays['residual_c'].shape != (10, len(arrays['hour_offsets']), n):
                    raise ValueError(f'{event}: unexpected RST shape')
                if list(arrays['hour_offsets']) != [0]:
                    raise ValueError(f'{event}: this release expects issue-time fields')
                for key in ('drivability', 'maintenance_need'):
                    if arrays[key].shape != (10, n) or not np.isin(arrays[key], [0, 1, 2]).all():
                        raise ValueError(f'{event}: invalid reference categories')
                for key in ('trend_c', 'residual_c', 'solar_altitude_degrees'):
                    if not np.isfinite(arrays[key]).all():
                        raise ValueError(f'{event}: nonfinite {key}')
                metadata = {'parent_event_id': event, 'cluster': int(parent['cluster']),
                            'issue_time_utc': str(rst.attrs['issue_time_utc']),
                            'issue_time_iowa_local': str(visual.attrs['issue_time_iowa_local']),
                            'rst_signature': str(rst.attrs['signature']),
                            'visual_signature': str(visual.attrs['signature'])}
            payload = arrays_bytes(arrays)
            if len(payload) + 4096 >= LIMIT:
                raise ValueError(f'{event}: parent cannot fit in one browser-upload archive')
            if archive is None or archive.fp.tell() + len(payload) + 65536 > LIMIT:
                if archive is not None:
                    archive.close()
                part += 1
                name = f'weather_events_part_{part:02d}.zip'
                archive = zipfile.ZipFile(output / name, 'w', compression=zipfile.ZIP_STORED)
            member = f'events/{event}.npz'
            archive.writestr(member, payload)
            archive.writestr(f'events/{event}.json', json.dumps(metadata, indent=2) + '\n')
            # Read back every array, not just the archive header, before recording completion.
            with np.load(io.BytesIO(payload), allow_pickle=False) as restored:
                for key, value in arrays.items():
                    if restored[key].dtype != value.dtype or not np.array_equal(restored[key], value):
                        raise ValueError(f'{event}: round-trip changed {key}')
            for realization in range(10):
                rows.append({**metadata, 'scenario_id': f'{event}__SGS{realization + 1:02d}',
                             'realization_index': realization, 'archive': name, 'member': member,
                             'support_points': n})
            sources.append({**metadata, 'rst_source_sha256': rst_hash,
                            'visual_source_sha256': visual_hash,
                            'npz_sha256': hashlib.sha256(payload).hexdigest()})
            print(f'Exported {event}: 10 realizations, archive {part}', flush=True)
    finally:
        if archive is not None:
            archive.close()
    for filename, records in [('scenario_manifest.csv', rows), ('source_checksums.csv', sources)]:
        with (output / filename).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    archives = [{'file': p.name, 'bytes': p.stat().st_size, 'sha256': sha256(p)}
                for p in sorted(output.glob('*.zip'))]
    if any(item['bytes'] >= LIMIT for item in archives):
        raise ValueError('Browser-upload archive limit exceeded')
    if len(rows) != 700 or any(sum(int(p['cluster']) == k for p in parents) != 10 for k in range(7)):
        raise ValueError('Expected seven groups with ten parents and 100 realizations each')
    summary = {'schema_version': 1, 'parent_events': 70, 'realizations': 700,
               'weather_groups': 7, 'support_points': n, 'hour_offsets': [0],
               'total_archive_bytes': sum(a['bytes'] for a in archives), 'archives': archives,
               'export_validated': True, 'source_arrays_preserved_exactly': True,
               'support_source_sha256': sha256(catalog_path),
               'raw_images_included': False, 'raw_station_histories_included': False,
               'license_status': 'Pending author/coauthor decision'}
    (output / 'dataset_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'archives'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Research after_8py_RW_simulation directory')
    parser.add_argument('--output', type=Path, default=ROOT / 'data' / 'weather_events')
    arguments = parser.parse_args()
    export(arguments.source.resolve(), arguments.output.resolve())
