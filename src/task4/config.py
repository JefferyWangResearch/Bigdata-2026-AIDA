from pathlib import Path
import sys,json,hashlib,datetime
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[2]; O=R/'outputs/task4'; D=R/'data/processed/task4'; LOG=R/'logs/task4'; FIG=R/'figures/task4'; T3=R/'data/processed/task3'
for p in [O,D,LOG,FIG]:p.mkdir(parents=True,exist_ok=True)
SEED=20261004
SP=['distance_nearest_source_km','source_count_25km','source_count_50km','source_pressure_L25']; E=['log_previous_3yr_effort']; M0=SP+E
CL=['annual_mean_tmax','annual_total_precipitation','annual_mean_vpd','annual_mean_wind_speed']
NORMAL=['normal_mean_tmax','normal_total_precipitation','normal_mean_vpd','normal_mean_wind_speed']; ANOM=['tmax_anomaly','precipitation_anomaly','vpd_anomaly','wind_speed_anomaly']
GROUPS={'E0':E,'S0':SP,'M0':M0,'M1':M0+CL,'C0':M0,'CN':M0+NORMAL,'CA':M0+ANOM,'CNA':M0+NORMAL+ANOM}
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def load_base():
    d=pd.read_csv(T3/'features_primary_3yr.csv.gz');assert d.prediction_year.between(2012,2025).all();assert (d.source_max_year<d.prediction_year).all();assert (d.climate_year==d.prediction_year-1).all();assert d.distance_nearest_source_km.le(50).all();assert not d.duplicated(['cell_id','prediction_year']).any();assert d.loc[d.label==1,'sample_keep'].all()
    return d

def setup():
    import shutil
    m=LOG/'task3_input_manifest.json'
    if not m.exists():
        inputs=[T3/'features_primary_3yr.csv.gz',R/'data/processed/grid_4km.csv',R/'data/processed/next_year_labels.csv.gz',R/'outputs/task3/model_results_by_year.csv',R/'outputs/task3/protocol.md',R/'src/task3/model.py']
        inputs +=list(T3.glob('climatology_*_1981_2010.csv'))+list(T3.glob('daily_*_*.npy'))+list((R/'outputs/task3/primary_3yr').glob('*_predictions.csv.gz'))
        # Only1981–2010 daily arrays are reused for normals.
        inputs=[p for p in inputs if not p.name.startswith('daily_') or 1981<=int(p.stem.split('_')[-1])<=2010]
        m.write_text(json.dumps({str(p.relative_to(R)):sha(p) for p in inputs},indent=2))
        preserved={str(p.relative_to(R)):sha(p) for folder in ['outputs','figures','src'] for p in (R/folder).rglob('*') if p.is_file() and 'task4' not in str(p.relative_to(R)) and '__pycache__' not in str(p)}
        (LOG/'prior_outputs_manifest.json').write_text(json.dumps(preserved,indent=2))
    protocol='''# Task4 frozen confirmation protocol

Frozen before Task4 outcome fits. Execute A, then B, then C, then report. No new species,climate variables,thresholds,seasons,models or tuned hyperparameters.4km grid,50km primary candidate domain,testyears2018–2025,training labels2012 onward. Reuse Task3 features and exact sample_keep mask (all positives,5%negative Bernoulli inclusion); all evaluation candidates retained. Training-only median imputation and StandardScaler,Logistic C1 L2 max_iter2000,RF500trees depth12 min_leaf10 max_features sqrt class_weightNone seed20261004 n_jobs4. Refit all eligible past training rows for every origin. Sampled probabilities corrected by q=.05 odds factor as in Task3. Primary AP is average precision,not trapezoidal integration. Report prevalence,lift,ROC-AUC,Brier,not threshold0.5 metrics.

A: independently fit E0,S0,M0,M1. Compare every M0/M1 prediction to Task3 saved predictions; tolerances1e-10 probability and AP. Paired year differences M0-E0,M0-S0,M1-M0.

B: after A, cluster all California4km grid centers in EPSG3310 with KMeans(n_clusters5,random_state20261004,n_init10,algorithm=lloyd). Coordinates only; no standardization of x/y because common meter units. Canonical fold IDs sorted by centroid northing then easting for reproducible labels. Freeze before B model evaluation. For each targetyear andheldoutfold,fit prior-year rows only outsidefold. Primary no buffer; sensitivity exclude outsidefold training cells whose centers are<=25km of ANY heldout-region grid center (all grid cells,not target-selected cells). No alternate buffer. Existing historical covariates may contain pre-year source history and effort inside heldout regions: supervised region holdout,not a never-observed or source-free region. Never recompute sources with future events. Both M0/M1,Logistic/RF,all5folds and8years. Concatenate disjoint region predictions BEFORE year-level metrics; do not average fold AP. Retain fold-specific metrics and support,NaNAP/ROC for zero-positive folds; never remove them from concatenated tests. If training has one class,use its fixed empirical prior for both models and flag failure; no alternate classifier search.

C: exact fixed1981–2010 climatology. Recompute equal-weight mean of30 annual means forTmax,VPD,wind from cacheddailyarrays (Task3 pooleddailymeans have slightly different leapyear weighting). Precipitation normal is mean30annualtotals. Require complete annual daily coverage,keep coast missing and impute only from eligible training rows. Previous-year anomalies = original Task3 annualfeature minus correspondingnormal. CNA contains normals+anomalies but NOT raw climate. Fit C0,CN,CA,CNA temporal; copy verified A M0 predictions for identical C0 rather than redundant fit. Spatial confirmation CN/CNA with same5folds,no buffer only; no additionalC variants or M2.

All main contrasts:8year differences,mean,median,positiveyears,20000 paired-year bootstrap mean95%CI(seed20261004),leave-one-year-out min/max mean. Eight correlated origins are not independent large-sample inference. Compare paired models within identical testsets; spatialfold cross-fitting mixes fold-specific fits,so calibration/relative risk scales must be assessed through retained Brier scores.

Final categories exactly user A/B/C/D. Evidence assessment uses fullyear/model/fold patterns and uncertainty,not one favorable mean. Preserve Task2 directionalwind NO SIGNAL andTask3 nonrobust extremes. No automatic new experiments after report. All phases include raw prediction checkpoints and provenance. No network download required.
'''
    if not (O/'protocol.md').exists():(O/'protocol.md').write_text(protocol)
    if not (LOG/'protocol_freeze.json').exists():(LOG/'protocol_freeze.json').write_text(json.dumps(dict(sha256=sha(O/'protocol.md'),timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
if __name__=='__main__':setup()
