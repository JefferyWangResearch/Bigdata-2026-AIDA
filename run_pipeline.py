"""Run the documented stages; fresh retrieval creates new data and result directories."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGES = {
    'occurrence': ['src/download_occurrence.py', 'src/clean_occurrence.py',
                   'src/build_grid.py', 'src/build_labels.py', 'src/make_figures.py', 'src/validate.py'],
    'effort-audit': ['src/task2/setup.py', 'src/task2/reduce_effort_pool.py', 'src/task2/effort.py'],
    'wind': ['src/task2/weather.py', 'src/task2/station_check.py', 'src/task2/test_exposure_math.py',
             'src/task2/exposure.py', 'src/task2/analyze.py', 'src/task2/figures.py', 'src/task2/validate.py'],
    'features': ['src/task3/setup.py', 'src/task3/spatial.py', 'src/task3/effort.py', 'src/task3/climate.py'],
    'models': ['src/task3/model.py', 'src/task3/radius_sensitivity.py',
               'src/task3/dictionary.py', 'src/task3/figures.py'],
    'confirmation': ['src/task4/config.py', 'src/task4/engine.py', 'src/task4/spatial.py',
                     'src/task4/climate.py', 'src/task4/tables.py', 'src/task4/figures.py',
                     'src/task4/validate.py'],
    'publication': ['paper/build_assets.py', 'summarize_results.py'],
}
INITIALIZERS = {
    'src/task2/setup.py': 'data/processed/task2/candidate_pool_full8.csv',
    'src/task2/reduce_effort_pool.py': 'logs/task2/effort_sampling_amendment.json',
    'src/task3/setup.py': 'logs/task3/prior_tasks_manifest.json',
    'src/task4/config.py': 'logs/task4/task3_input_manifest.json',
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES)
    parser.add_argument('--list', action='store_true', help='Print execution order without downloads or file creation')
    args = parser.parse_args()
    if args.list or args.stage is None:
        for stage, scripts in STAGES.items():
            print(stage + (' (optional historical wind analysis)' if stage == 'wind' else ''))
            for script in scripts:
                print('  ' + script)
        return
    for base in ['data/raw', 'data/interim', 'data/processed', 'outputs', 'logs', 'figures']:
        (ROOT / base).mkdir(parents=True, exist_ok=True)
    for stage in ['task2', 'task3', 'task4']:
        for base in ['data/raw', 'data/processed', 'outputs', 'logs', 'figures']:
            (ROOT / base / stage).mkdir(parents=True, exist_ok=True)
    for name in ['figures', 'generated', 'audit']:
        (ROOT / 'paper' / name).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
               MPLCONFIGDIR=str(ROOT / 'logs' / 'matplotlib'))
    for script in STAGES[args.stage]:
        marker = INITIALIZERS.get(script)
        if marker and (ROOT / marker).exists():
            print('Using existing initialization: ' + marker, flush=True)
            continue
        print('Running ' + script, flush=True)
        subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, env=env, check=True)

if __name__ == '__main__':
    main()
