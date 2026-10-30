"""Validate compact release contents without access to the original archive."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def files():
    return [p for p in ROOT.rglob('*') if p.is_file()
            and not any(part in {'.git', '.venv', '__pycache__', 'generated'} for part in p.relative_to(ROOT).parts)]


def main():
    paths = files()
    for path in paths:
        relative = path.relative_to(ROOT)
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(relative))
        if path.suffix in {'.py', '.json', '.md', '.csv', '.yml'}:
            text = path.read_text()
            assert ('/'+'Users/') not in text and ('G:'+chr(92)+'Mingjian') not in text, f'Personal path in {relative}'
        assert path.suffix not in {'.db', '.sqlite', '.h5', '.hdf5', '.joblib', '.pkl', '.graphml', '.tif'}, relative
        assert path.stat().st_size < 10*1024**2, f'Large file: {relative}'
    with (ROOT/'provenance/export_manifest.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['exported_sha256'] == 'consolidated':
                continue
            content = (ROOT/row['destination']).read_bytes()
            assert hashlib.sha256(content).hexdigest() == row['exported_sha256'], row['destination']
    sgs = ROOT/'results/iowa/sgs'
    manifest = pd.read_csv(sgs/'10_sgs_manifest.csv')
    assert len(manifest) == 70 and manifest.event_id.nunique() == 70
    assert int(manifest.n_realizations.sum()) == 700
    assert manifest.n_road_points.eq(122329).all()
    assert manifest.conditioning_observations.eq(420).all()
    statistics = pd.read_csv(sgs/'all_realization_statistics.csv')
    assert len(statistics) == 700 and statistics.parent_event_id.nunique() == 70
    seeds = pd.read_csv(sgs/'all_realization_seeds.csv', dtype=str)
    assert len(seeds) == 700
    assert seeds.draw_seed.str.fullmatch(r'\d+').all() and seeds.path_seed.str.fullmatch(r'\d+').all()
    with np.load(ROOT/'models/iowa_kmeans_parameters.npz', allow_pickle=False) as model:
        assert model['scaler_mean'].shape == (2100,)
        assert model['centers'].shape == (7, 45)
    models = json.loads((ROOT/'models/iowa_strk_parameters.json').read_text())
    assert set(models) == set(map(str, range(7)))
    summary = dict(audit_passed=True, file_count=len(paths), total_bytes=sum(p.stat().st_size for p in paths),
                   original_data_required_for_full_rerun=True, python=platform.python_version(),
                   local_platform=platform.system(), windows_execution_tested=False,
                   license_status='Pending', checks=['Python syntax', 'No machine-specific home paths',
                   'No observation databases or large simulation archives', 'Source-export checksum accounting',
                   '70 parents / 700 SGS realization summaries', 'Portable numerical parameter shapes'])
    (ROOT/'provenance/release_validation.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
