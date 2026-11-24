"""Write a numerical evidence summary from newly generated tables, without preset conclusions."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
TABLES = {
    'Information groups: temporal gains': 'decomposition_gain_summary.csv',
    'Long-term normals and annual anomalies: temporal gains': 'climate_decomposition_gain_summary.csv',
    'Geographic holdout and buffer sensitivity': 'spatial_temporal_gain_summary.csv',
    'Anomalies after normals: geographic holdout': 'climate_spatial_gain_summary.csv',
    'Geographic region sensitivity': 'spatial_fold_gain_summary.csv',
}

def main():
    folder = ROOT / 'outputs' / 'task4'
    paragraphs = ['# Evidence summary from the current run',
                  'The tables below are read from this run. No fixed GO decision or historical narrative is substituted for the computed results. Positive gains favor the augmented feature set; retain negative years and regional counterexamples.',
                  'AP is average precision, not trapezoidal PR-AUC. Paired-year intervals describe variation across the eight dependent test years; they are not independent-sample significance tests. The target is a first recorded occurrence, not biological arrival.']
    for title, filename in TABLES.items():
        table = pd.read_csv(folder / filename)
        paragraphs += ['## ' + title, 'Source: `' + filename + '`', table.to_markdown(index=False)]
    paragraphs += ['## Interpretation boundaries',
                   'Compare mean and median gains, positive-year counts, uncertainty intervals, and geographic sensitivity together. Historical observation dates do not reconstruct which records had already been published at each forecast origin. Reassess the manuscript conclusions if current downloads change the results.']
    (folder / 'current_run_evidence.md').write_text('\n\n'.join(paragraphs) + '\n')
    print('Wrote outputs/task4/current_run_evidence.md')

if __name__ == '__main__':
    main()
