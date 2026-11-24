from setup import *
import os,warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score,roc_auc_score,precision_score,recall_score,f1_score,brier_score_loss
M0=['distance_nearest_source_km','source_count_25km','source_count_50km','source_pressure_L25','log_previous_3yr_effort']
ORD=['annual_mean_tmax','annual_total_precipitation','annual_mean_vpd','annual_mean_wind_speed']
EXT=['hot_days_p95','high_vpd_days_p95','max_consecutive_dry_days','annual_precip_anomaly','high_wind_days_p95']
GROUPS={'M0':M0,'M1':M0+ORD,'M2':M0+ORD+EXT}

def prepare(quality,effort_kind,only_spatial=False):
    sp=pd.read_csv(D/f'spatial_{quality}.csv.gz');cl=None if only_spatial else pd.read_csv(D/'climate_panel.csv.gz');e=pd.read_csv(D/'effort_panel.csv.gz');wide=e.pivot(index='cell_id',columns='year',values='count');rows=[]
    assert json.loads((O/'effort_proxy_audit.json').read_text())['passed']
    for year in range(2012,2026):
        a=wide[[year-3,year-2,year-1]];assert a.notna().all().all();v=a.mean(axis=1) if effort_kind=='3yr' else a[year-1];ee=v.rename('effort').reset_index();ee['prediction_year']=year;ee['log_previous_3yr_effort']=np.log1p(ee.effort);rows.append(ee)
    panel=sp.copy() if only_spatial else sp.merge(cl,on=['cell_id','prediction_year','feature_max_date'],validate='one_to_one')
    if only_spatial:panel['climate_year']=panel.prediction_year-1
    panel=panel.merge(pd.concat(rows),on=['cell_id','prediction_year'],validate='one_to_one')
    assert len(panel)==len(sp);assert (panel.climate_year==panel.prediction_year-1).all();assert (panel.source_max_year<panel.prediction_year).all();assert (pd.to_datetime(panel.feature_max_date).dt.year<panel.prediction_year).all()
    panel=panel.sort_values(['prediction_year','cell_id']).reset_index(drop=True)
    panel['sample_keep']=(panel.label==1)|(np.random.default_rng(SEED).random(len(panel))<.05)
    stage='_M0stage' if only_spatial else ''
    panel.to_csv(D/f'features_{quality}_{effort_kind}{stage}.csv.gz',index=False)
    if not only_spatial:panel.groupby('prediction_year')[ORD+EXT].agg(lambda s:s.isna().mean()).to_csv(O/f'missing_features_{quality}_{effort_kind}.csv')
    if not only_spatial and quality=='primary' and effort_kind=='3yr':panel.loc[panel.prediction_year<2018,M0+ORD+EXT].corr().to_csv(O/'pretest_feature_correlations.csv')
    return panel

def run(quality,effort_kind,only_spatial=False):
    name=f'{quality}_{effort_kind}';panel=prepare(quality,effort_kind,only_spatial);out=O/name;out.mkdir(exist_ok=True);results=[];coefs=[]
    for year in range(2018,2026):
        fulltrain=panel[panel.prediction_year<year];train=fulltrain[fulltrain.sample_keep];test=panel[panel.prediction_year==year]
        assert train.prediction_year.max()==year-1 and test.prediction_year.nunique()==1
        for model in ['Logistic','RF']:
            for group,features in GROUPS.items():
                if only_spatial and group != "M0":continue
                key=f'{year}_{model}_{group}';metrics=out/f'{key}.json';predfile=out/f'{key}_predictions.csv.gz';coefpath=out/f'{key}_coefficients.csv'
                if metrics.exists() and predfile.exists():
                    results.append(json.loads(metrics.read_text()))
                    if coefpath.exists():coefs.extend(pd.read_csv(coefpath).to_dict('records'))
                    continue
                estimator=LogisticRegression(C=1,max_iter=2000,random_state=SEED) if model=='Logistic' else RandomForestClassifier(n_estimators=500,max_depth=12,min_samples_leaf=10,max_features='sqrt',class_weight=None,n_jobs=4,random_state=SEED)
                pipe=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),estimator)
                pipe.fit(train[features],train.label)
                raw=pipe.predict_proba(test[features])[:,1];pred=.05*raw/(1-raw+.05*raw);hard=pred>=.5
                r=dict(analysis=name,prediction_year=year,model=model,feature_group=group,n_train=len(train),n_train_full=len(fulltrain),n_positive_train=int(train.label.sum()),n_test=len(test),n_positive_test=int(test.label.sum()),positive_rate=float(test.label.mean()),PR_AUC=float(average_precision_score(test.label,pred)),ROC_AUC=float(roc_auc_score(test.label,pred)),precision=float(precision_score(test.label,hard,zero_division=0)),recall=float(recall_score(test.label,hard,zero_division=0)),F1=float(f1_score(test.label,hard,zero_division=0)),Brier=float(brier_score_loss(test.label,pred)),threshold=.5,negative_inclusion_probability=.05,training_max_year=int(train.prediction_year.max()),feature_max_date=test.feature_max_date.max())
                r['AP_over_prevalence']=r['PR_AUC']/r['positive_rate'];r['prevalence_Brier']=r['positive_rate']*(1-r['positive_rate']);r['Brier_skill_vs_test_prevalence']=1-r['Brier']/r['prevalence_Brier']
                p=test[['cell_id','prediction_year','label','center_lon','center_lat']].copy();p['model']=model;p['feature_group']=group;p['probability']=pred;p.to_csv(predfile,index=False)
                if model=='Logistic':
                    z=pipe[:-1].transform(train[features]);z=np.column_stack([np.ones(len(z)),z]);pt=estimator.predict_proba(z[:,1:])[:,1];pen=np.eye(z.shape[1]);pen[0,0]=0;h=z.T@((pt*(1-pt))[:,None]*z)+pen;se=np.sqrt(np.diag(np.linalg.pinv(h)))[1:];b=estimator.coef_[0]
                    cc=[dict(analysis=name,prediction_year=year,feature_group=group,feature=f,beta_per_train_SD=float(b[i]),curvature_CI_low=float(b[i]-1.96*se[i]),curvature_CI_high=float(b[i]+1.96*se[i]),interval_type='penalized likelihood curvature; ignores repeated-cell/year dependence; descriptive only') for i,f in enumerate(features)]
                    pd.DataFrame(cc).to_csv(coefpath,index=False);coefs.extend(cc)
                metrics.write_text(json.dumps(r,indent=2));results.append(r);print(name,key,'AP',round(r['PR_AUC'],5),'lift',round(r['AP_over_prevalence'],2),flush=True)
    stage='_M0stage' if only_spatial else ''
    pd.DataFrame(results).to_csv(O/f'model_results_{name}{stage}.csv',index=False);pd.DataFrame(coefs).to_csv(O/f'logistic_coefficients_{name}{stage}.csv',index=False)
    return pd.DataFrame(results)

def summaries(d,suffix=''):
    paired=[]
    for (model,year),a in d.groupby(['model','prediction_year']):
        x=a.set_index('feature_group').PR_AUC
        for lo,hi in [('M0','M1'),('M1','M2')]:paired.append(dict(prediction_year=year,model=model,comparison=f'{hi}-{lo}',delta_PR_AUC=x[hi]-x[lo],relative_gain=(x[hi]-x[lo])/x[lo]))
    p=pd.DataFrame(paired);p.to_csv(O/f'paired_feature_gain_by_year{suffix}.csv',index=False);stats=[]
    for (model,comparison),a in p.groupby(['model','comparison']):
        v=a.delta_PR_AUC.values;rng=np.random.default_rng(SEED);b=rng.choice(v,(20000,len(v)),replace=True).mean(axis=1)
        stats.append(dict(comparison=comparison,model=model,mean_delta_PR_AUC=v.mean(),median_delta_PR_AUC=np.median(v),CI_low=np.quantile(b,.025),CI_high=np.quantile(b,.975),years_positive=int((v>0).sum()),years_total=len(v),median_relative_gain=a.relative_gain.median(),leave_one_year_out_min_mean=min(np.delete(v,i).mean() for i in range(len(v))),leave_one_year_out_max_mean=max(np.delete(v,i).mean() for i in range(len(v)))))
    stats=pd.DataFrame(stats);stats.to_csv(O/f'incremental_gain_summary{suffix}.csv',index=False)
    summary=d.groupby(['model','feature_group']).agg(median_PR_AUC=('PR_AUC','median'),mean_PR_AUC=('PR_AUC','mean'),median_ROC_AUC=('ROC_AUC','median'),median_Brier=('Brier','median'),median_AP_over_prevalence=('AP_over_prevalence','median')).reset_index()
    summary['positive_years_vs_previous_model']=[np.nan if r.feature_group=='M0' else int(stats[(stats.model==r.model)&(stats.comparison==('M1-M0' if r.feature_group=='M1' else 'M2-M1'))].years_positive.iloc[0]) for r in summary.itertuples()];summary.to_csv(O/f'model_summary{suffix}.csv',index=False)

if __name__=='__main__':
    if '--spatial-stage' in sys.argv:
        for quality,effort in [('primary','3yr'),('strict','3yr'),('primary','1yr')]:run(quality,effort,True)
        raise SystemExit(0)
    d=run('primary','3yr');d.to_csv(O/'model_results_by_year.csv',index=False);summaries(d)
    for quality,effort in [('strict','3yr'),('primary','1yr')]:summaries(run(quality,effort),f'_{quality}_{effort}')
