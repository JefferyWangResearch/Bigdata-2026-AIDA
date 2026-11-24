from pathlib import Path
import sys,json,hashlib,datetime
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'src'))
from common import FWD,INV,assign
RAW=R/'data/raw/task3'; D=R/'data/processed/task3'; O=R/'outputs/task3'; LOG=R/'logs/task3'; FIG=R/'figures'
for p in [RAW,D,O,LOG]:p.mkdir(exist_ok=True,parents=True)
SEED=20261004

def stamp(path,url):
    path.with_suffix('.meta.json').write_text(json.dumps(dict(url=url,retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2))

if __name__=='__main__':
    import shutil
    p=LOG/'prior_tasks_manifest.json'
    if not p.exists():
        files={str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for base in ['src','data','outputs','figures','logs'] for f in (R/base).rglob('*') if f.is_file() and 'task3' not in str(f.relative_to(R)) and 'T3_' not in f.name and '__pycache__' not in str(f)}
        p.write_text(json.dumps(files,indent=2))
    protocol='''# Task 3 frozen minimal protocol
Written before any held-out Task 3 model evaluation.

Grid 4 km, radius 50 km. Training labels 2012 onward; held-out years 2018–2025. Use all pre-year source history including pre-2012 records. Candidate cells never observed before target year, distance <=50 km. Complete unsampled held-out candidates; source pressure sum exp(-d/25) over all historical occupied cells. Counts within 25 and 50 km. Every feature's maximum date <=December31(targetyear-1).

M0: nearest-source km, source counts25/50km, source pressureL25, log1p(mean vascular-plant count in preceding3years). M1 adds preceding calendar year's mean Tmax(C), total precipitation(mm), mean VPD(kPa), mean nondirectional wind(m/s). M2 adds days above cell-specific 1981–2010 pooled daily p95 for Tmax/VPD/wind; max consecutive days precipitation<1mm (reset at year boundary); precipitation total minus baseline mean annual total. No seasonal feature expansion. Quantiles linear interpolation; annual data require complete daily coverage, otherwise NaN. Cell climatology means/SD/quantiles retained for audit.

Effort: uniformly query GBIF vascular-plant vector tiles, country US, occurrenceStatus PRESENT, annual event-year. Map quantized tile positions to EPSG3310 grid. Use existing exact Task2 counts for external proxy audit only, never to define test domain or selectively replace tile data. Effort is an approximate observer-activity proxy; GBIF snapshot event times do not establish historical database availability. Assess proxy before outcome models.

Models fixed before test evaluation: LogisticRegression C=1, L2, max_iter=2000, train-only median imputation and StandardScaler. RF500trees,max_depth=12,min_samples_leaf=10,max_features=sqrt. No class weighting. Train includes all positives and independently selected 5% of negatives (fixed seed); all feature groups share sampled rows. Correct sampled odds by negative inclusion probability0.05. No hyperparameter search. Threshold0.5 on corrected probabilities for precision/recall/F1; rare-event ranking is primary. All past training years refit for every origin; no unrefit validation model.

Primary PR-AUC explicitly average precision (AP), not trapezoidal PR integration. Other metrics ROC-AUC, Brier, precision, recall,F1; prevalence and AP/prevalence. Paired year differences M1-M0 and M2-M1, bootstrap20000 resamples of8years for mean95% interval, equal-year summaries and leave-one-year-out means. Quantile calibration bins; no claim of calibrated probabilities solely from correction. M2 logistic coefficients with penalized curvature intervals, explicitly descriptive/non-robust due repeated cells, never causal or confirmatory tests.

Required robustness: independently reconstruct strict known uncertainty<=1km sources AND labels; repeat M0/M1/M2, and primary labels with preceding-year effort. Same-year diagnostic omitted from minimal prospective run. Do not exclude years or cells on observed outcomes or climate completeness; train-only median climate imputation. Missingness audited.

Conditional expansion only if M2 mean and median AP gain positive, >=6/8years positive, year bootstrap lower bound>0 in at least one model, and no directional reversal under strict/effort sensitivity. This is a conservative operational trigger for optional ablation, not a revised success threshold. Otherwise retain minimal analysis and report uncertainty. Final A/B/C/D combines per-year paired evidence, strict sensitivity, observation effort and baseline lift using user decision rules. No post hoc feature/threshold/model search.

Preselected map year2025; show RF M0/M1/M2 in parallel, not best-model-picked risk. Observational target newly observed cells, not biological arrival, absence, or causal climate response. Preserve Task2 NO SIGNAL.
'''
    (O/'protocol.md').write_text(protocol)
