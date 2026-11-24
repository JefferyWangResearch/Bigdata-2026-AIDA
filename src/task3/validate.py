from setup import *
from model import GROUPS
from sklearn.metrics import average_precision_score,brier_score_loss
from scipy.spatial.distance import cdist
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def main():
    checks={};old=json.loads((LOG/'prior_tasks_manifest.json').read_text());bad=[]
    for name,h in old.items():
        p=R/name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:bad.append(name)
    assert not bad,bad;checks['prior_tasks_unchanged']=len(old)
    results=pd.read_csv(O/'model_results_by_year.csv');assert len(results)==48;assert set(results.prediction_year)==set(range(2018,2026));assert results.training_max_year.eq(results.prediction_year-1).all()
    assert results.groupby('prediction_year').n_test.nunique().eq(1).all();assert results.groupby('prediction_year').n_positive_test.nunique().eq(1).all();assert results.drop_duplicates('prediction_year').n_positive_test.sum()==318
    checks['complete_primary_test_cell_years']=int(results.drop_duplicates('prediction_year').n_test.sum());checks['positive_primary_test_cell_years']=318
    for quality,effort in [('primary','3yr'),('strict','3yr'),('primary','1yr')]:
        name=f'{quality}_{effort}';d=pd.read_csv(D/f'features_{name}.csv.gz');assert not d.duplicated(['cell_id','prediction_year']).any();assert (d.source_max_year<d.prediction_year).all();assert (d.climate_year==d.prediction_year-1).all();assert (pd.to_datetime(d.feature_max_date).dt.year<d.prediction_year).all();assert (d.distance_nearest_source_km<=50).all()
        r=pd.read_csv(O/f'model_results_{name}.csv');assert len(r)==48
        for row in r.itertuples():
            pred=pd.read_csv(O/name/f'{row.prediction_year}_{row.model}_{row.feature_group}_predictions.csv.gz');truth=d[d.prediction_year==row.prediction_year].sort_values('cell_id');assert len(pred)==len(truth);assert pred.cell_id.tolist()==truth.cell_id.tolist();assert np.array_equal(pred.label,truth.label);assert np.isclose(average_precision_score(pred.label,pred.probability),row.PR_AUC);assert np.isclose(brier_score_loss(pred.label,pred.probability),row.Brier)
    checks['no_future_features_and_unsampled_tests_all_three_analyses']='PASS'
    grid=pd.read_csv(R/'data/processed/grid_4km.csv');occ=pd.read_csv(R/'data/processed/occurrences_with_cells.csv');first=occ.groupby('cell_id').year.min();g=grid.merge(first.rename('first_year'),on='cell_id',how='left');d=pd.read_csv(D/'features_primary_3yr.csv.gz')
    for year in [2018,2025]:
        sample=d[d.prediction_year==year].sample(10,random_state=SEED);hist=g[g.first_year<year];dist=cdist(sample[['x','y']],hist[['x','y']])/1000
        assert np.allclose(sample.distance_nearest_source_km,dist.min(1));assert np.allclose(sample.source_pressure_L25,np.exp(-dist/25).sum(1));assert np.array_equal(sample.source_count_25km,(dist<=25).sum(1))
    checks['historical_spatial_feature_spot_checks']='PASS'
    # Refit one complete early held-out fold independently; compare saved scores.
    tr=d[(d.prediction_year<2018)&d.sample_keep];te=d[d.prediction_year==2018];features=GROUPS['M2'];errs={}
    for model,est in [('Logistic',LogisticRegression(C=1,max_iter=2000,random_state=SEED)),('RF',RandomForestClassifier(n_estimators=500,max_depth=12,min_samples_leaf=10,max_features='sqrt',class_weight=None,n_jobs=4,random_state=SEED))]:
        pipe=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),est);pipe.fit(tr[features],tr.label);p=pipe.predict_proba(te[features])[:,1];p=.05*p/(1-p+.05*p);saved=pd.read_csv(O/f'primary_3yr/2018_{model}_M2_predictions.csv.gz').probability.values;err=float(np.max(np.abs(p-saved)));assert err<1e-10;errs[model]=err
    checks['independent_refit_max_probability_error']=errs
    assert json.loads((O/'effort_proxy_audit.json').read_text())['passed'];checks['uniform_effort_proxy_audit']='PASS'
    for code in ['pr','tmmx','vpd','vs']:
        manifest=json.loads((LOG/f'climate_manifest_{code}.json').read_text());assert [m['year'] for m in manifest]==list(range(1981,2025))
        for m in manifest:assert hashlib.sha256((R/m['path']).read_bytes()).hexdigest()==m['sha256']
    checks['raw_climate_hashes_verified']=176
    for name in ['T3_model_ablation_prauc.png','T3_climate_gain_by_year.png','T3_extreme_gain_by_year.png','T3_2025_expansion_risk_map.png','T3_calibration.png']:assert (FIG/name).stat().st_size>10000
    checks['required_figures']=5
    # Conditional radii: complete Task1 domains, unchanged primary, finite predictions/coefficients.
    radius_metrics=0
    expected=pd.read_csv(R/'data/processed/next_year_labels.csv.gz')
    for radius in [25,100]:
        name=f'primary_r{radius}_3yr'
        if not (O/f'model_results_{name}.csv').exists():continue
        rr=pd.read_csv(O/f'model_results_{name}.csv');assert len(rr)==48
        for a in rr.itertuples():
            ref=expected[(expected.prediction_year==a.prediction_year)&expected[f'candidate_{radius}km']]
            pp=pd.read_csv(O/name/f'{a.prediction_year}_{a.model}_{a.feature_group}_predictions.csv.gz')
            assert set(pp.cell_id)==set(ref.cell_id);assert len(pp)==len(ref);assert np.isfinite(pp.probability).all();assert np.isclose(average_precision_score(pp.label,pp.probability),a.PR_AUC);radius_metrics+=1
    for p in O.glob('*/**/*_coefficients.csv'):
        a=pd.read_csv(p);assert np.isfinite(a.select_dtypes('number')).all().all(),str(p)
    checks['conditional_radius_metrics_checked']=radius_metrics
    checks['status']='PASS';(LOG/'validation.json').write_text(json.dumps(checks,indent=2));print(checks,flush=True)
if __name__=='__main__':main()
