"""Run a selected Iowa stage without changing the original research folder."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RW = ROOT/'workflow/analysis-iowa/RW_module'
STAGES = {
    'aggregate': ('2a_build_rwis_station_hourly_modeling_table.py', []),
    'qc': ('2b_filter_unreasonable_rwis_values.py', []),
    'stations': ('3_most_overlapped_rwis.py', []),
    'prepare': ('4_v2_prepare_imputed_weather_regime_samples.py', ['--rebuild']),
    'cluster': ('5_v2_cluster_rst_weather_regimes_kmeans.py', []),
    'visualize': ('6_v2_visualize_rst_kmeans_clusters.py', []),
    'fit': ('7_v2_fit_cluster_specific_regression_variograms_with_elevation.py', []),
    'evaluate': ('8_v2_evaluate_cluster_specific_strk.py', []),
    'select': ('after_8py_RW_simulation/9_pick_weather_events.py', []),
    'sgs': ('after_8py_RW_simulation/10_s1_sgs_road_rst_fields.py', []),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=list(STAGES)+['all'], required=True)
    parser.add_argument('--db', type=Path, help='Original SQLite archive; required by aggregation')
    parser.add_argument('--dry-run', action='store_true', help='Print commands only; does not validate input completeness')
    args, extra = parser.parse_known_args()
    selected = list(STAGES) if args.stage == 'all' else [args.stage]
    if args.stage == 'all' and extra:
        parser.error('Pass stage-specific arguments with a single --stage, not --stage all.')
    for name in selected:
        file, defaults = STAGES[name]
        command = [sys.executable, str(RW/file), *defaults]
        if name == 'aggregate':
            database = args.db.expanduser().resolve() if args.db else RW.parent/'ROAD_DATA_2015-2019_IOWA.db'
            if not args.dry_run and not database.is_file():
                parser.error('The full RWIS SQLite archive is not distributed. Supply --db /path/to/archive.db.')
            command += ['--db', str(database)]
        if args.stage != 'all':
            command += extra
        print(f'[{name}] '+subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
