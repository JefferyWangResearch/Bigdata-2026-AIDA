from config import *
from engine import metrics,GROUPS
from scipy.spatial import cKDTree

def main():
    checks={}
    for name in ['task3_input_manifest.json','prior_outputs_manifest.json']:
        m=json.loads((LOG/name).read_text());changed=[f for f,h in m.items() if not (R/f).exists() or sha(R/f)!=h];assert not changed,changed;checks[name]=dict(files_unchanged=len(m),status='PASS')
    d=load_base();folds=pd.read_csv(R/'data/processed/task4_spatial_folds.csv');assert len(folds)==26294 and not folds.cell_id.duplicated().any() and set(folds.spatial_fold)==set(range(1,6));d=d.merge(folds,on='cell_id');buf=pd.read_csv(D/'spatial_buffer_membership.csv.gz');g=pd.read_csv(D/'spatial_fold_geometry.csv')
    for k in range(1,6):
        ref=cKDTree(g.loc[g.spatial_fold==k,['x','y']]).query(g[['x','y']])[0]/1000;got=buf[buf.heldout_fold==k];assert got.cell_id.tolist()==g.cell_id.tolist();assert np.allclose(got.distance_to_heldout_km,ref);assert np.array_equal(got.excluded_25km,ref<=25)
    checks['folds_coordinate_only_and_buffer_distances']='PASS'
    nfiles=0;fallbacks=0
    for p in (O/'predictions').glob('*.json'):
        r=json.loads(p.read_text());pp=pd.read_csv(p.with_suffix('.csv.gz'));year=r['prediction_year'];k=r['spatial_fold'];b=r['buffer_km'];truth=d[d.prediction_year==year]
        tr=d[d.prediction_year<year]
        if k>=0:
            truth=truth[truth.spatial_fold==k];tr=tr[tr.spatial_fold!=k]
            if b:tr=tr[~tr.cell_id.isin(buf.loc[(buf.heldout_fold==k)&buf.excluded_25km,'cell_id'])]
            assert not set(tr.cell_id)&set(g.loc[g.spatial_fold==k,'cell_id'])
        assert len(tr)==r['n_train_full'];assert int(tr.sample_keep.sum())==r['n_train'];assert tr.label.sum()==r['n_positive_train'];assert int(tr.prediction_year.max())==r['training_max_year']<year
        assert set(pp.cell_id)==set(truth.cell_id) and not pp.cell_id.duplicated().any();truth=truth.set_index('cell_id');assert np.array_equal(pp.label.values,truth.loc[pp.cell_id,'label'].values);assert np.isfinite(pp.probability).all()
        assert pd.Timestamp(r['test_max_feature_date']).year<year;assert pd.Timestamp(r['training_max_feature_date']).year<year
        mm=metrics(pp.label,pp.probability)
        for metric in ['PR_AUC','ROC_AUC','Brier']:assert np.isclose(mm[metric],r[metric],equal_nan=True)
        nfiles+=1;fallbacks+=r['single_class_fallback']
    checks['prediction_checkpoints_validated']=nfiles;checks['single_class_fallbacks']=fallbacks
    for phase,outname,groups,buffers in [('B','spatial_temporal_results_by_year.csv',['M0','M1'],[0,25]),('C_spatial','climate_spatial_results_by_year.csv',['CN','CNA'],[0])]:
        table=pd.read_csv(O/outname)
        for r in table.itertuples():
            p=pd.read_csv(O/f'predictions/{phase}_{r.prediction_year}_crossfit_buffer{r.buffer_km}_{r.model}_{r.feature_group}.csv.gz');ref=d[d.prediction_year==r.prediction_year];assert len(p)==len(ref);assert set(p.cell_id)==set(ref.cell_id);assert set(p.spatial_fold)==set(range(1,6));assert not p.cell_id.duplicated().any();assert np.isclose(metrics(p.label,p.probability)['PR_AUC'],r.PR_AUC)
    checks['all_crossfit_tests_complete_and_unsampled']='PASS'
    for name,expected in [('decomposition_results_by_year.csv',64),('spatial_temporal_results_by_year.csv',64),('climate_decomposition_results_by_year.csv',64),('climate_spatial_results_by_year.csv',32)]:
        x=pd.read_csv(O/name);assert len(x)==expected;assert set(x.prediction_year)==set(range(2018,2026));assert np.isfinite(x[['PR_AUC','ROC_AUC','Brier']]).all().all()
    cc=pd.read_csv(D/'climate_decomposition_panel.csv.gz');assert len(cc)==len(d);assert cc.groupby('cell_id')[NORMAL].nunique().max().max()<=1;assert (cc.climate_year==cc.prediction_year-1).all()
    for raw,n,a in zip(CL,NORMAL,ANOM):assert np.allclose(cc[raw],cc[n]+cc[a],equal_nan=True)
    assert not set(CL)&set(GROUPS['CNA']);assert not set(CL)&set(GROUPS['CA']);assert not set(CL)&set(GROUPS['CN']);checks['normal_anomaly_no_raw_duplicate_or_future_features']='PASS'
    repro=pd.read_csv(O/'task3_M0_M1_reproduction.csv');assert len(repro)==32 and repro.max_probability_difference.max()<1e-10;checks['Task3_max_probability_difference']=float(repro.max_probability_difference.max())
    # Independent normal calculation for several cells, including leap-year weighting.
    normal=pd.read_csv(D/'climate_normals_1981_2010.csv')
    for code,name in zip(['tmmx','pr','vpd','vs'],NORMAL):
        ix=normal.index[normal[name].notna()][::5000].tolist();annual=[]
        for y in range(1981,2011):
            a=np.load(T3/f'daily_{code}_{y}.npy',mmap_mode='r')[:,ix].astype('float64');annual.append(a.sum(0) if code=='pr' else a.mean(0))
        assert np.allclose(normal.loc[ix,name],np.mean(annual,axis=0))
    checks['independent_equal_year_climatology_spot_checks']='PASS'
    for name in ['T4_information_decomposition.png','T4_added_value_breakdown.png','T4_spatial_folds.png','T4_temporal_vs_spatial_generalization.png','T4_climate_normal_vs_anomaly.png','T4_dynamic_climate_gain_by_year.png']:assert (FIG/name).stat().st_size>10000
    checks['status']='PASS';(LOG/'validation.json').write_text(json.dumps(checks,indent=2));print(checks,flush=True)
if __name__=='__main__':main()
