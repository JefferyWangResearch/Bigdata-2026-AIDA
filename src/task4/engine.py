from config import *
import warnings,time
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss

def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);assert len(y)>0 and np.isfinite(p).all() and ((p>=0)&(p<=1)).all();prev=float(y.mean())
    return dict(n_test=len(y),n_positive_test=int(y.sum()),prevalence=prev,PR_AUC=float(average_precision_score(y,p)) if y.sum()>0 else np.nan,AP_over_prevalence=float(average_precision_score(y,p)/prev) if prev>0 else np.nan,ROC_AUC=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,Brier=float(brier_score_loss(y,p)))

def fit_predict(trainfull,test,model,group,phase,year,fold=-1,buffer_km=0):
    train=trainfull[trainfull.sample_keep].copy();features=GROUPS[group]
    assert train.prediction_year.max()<year and test.prediction_year.eq(year).all();assert train.loc[train.label==1].shape[0]==trainfull.label.sum();assert len(train)>0
    if fold>=0:assert (trainfull.spatial_fold!=fold).all() and test.spatial_fold.eq(fold).all();assert not set(trainfull.cell_id)&set(test.cell_id)
    key=f'{phase}_{year}_fold{fold}_buffer{buffer_km}_{model}_{group}';folder=O/'predictions';folder.mkdir(exist_ok=True);dest=folder/f'{key}.csv.gz';meta=folder/f'{key}.json'
    sig=hashlib.sha256(pd.util.hash_pandas_object(train[features+['label','prediction_year','cell_id']],index=False).values.tobytes()+pd.util.hash_pandas_object(test[features+['label','prediction_year','cell_id']],index=False).values.tobytes()+json.dumps(dict(model=model,group=group,seed=SEED,C=1,trees=500,depth=12,leaf=10,q=.05),sort_keys=True).encode()).hexdigest()
    if dest.exists() and meta.exists():
        rr=json.loads(meta.read_text());assert rr['input_signature']==sig,'Task4 cache inputs changed; archive checkpoint instead of silently reusing'
        return pd.read_csv(dest),rr
    est=LogisticRegression(C=1,max_iter=2000,random_state=SEED) if model=='Logistic' else RandomForestClassifier(n_estimators=500,max_depth=12,min_samples_leaf=10,max_features='sqrt',class_weight=None,n_jobs=4,random_state=SEED)
    pipe=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),est)
    start=time.monotonic();fallback=train.label.nunique()<2
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        if fallback:pred=np.repeat(float(trainfull.label.mean()),len(test))
        else:
            pipe.fit(train[features],train.label);assert np.isfinite(pipe[:-1].transform(train[features])).all();raw=pipe.predict_proba(test[features])[:,1];pred=.05*raw/(1-raw+.05*raw)
            if model=='Logistic':assert np.isfinite(est.coef_).all() and np.isfinite(est.intercept_).all()
    a=test[['cell_id','prediction_year','label']].copy();a['probability']=pred;a['spatial_fold']=fold;a['model']=model;a['feature_group']=group;a['buffer_km']=buffer_km
    rr=dict(phase=phase,prediction_year=year,spatial_fold=fold,buffer_km=buffer_km,model=model,feature_group=group,n_train_full=len(trainfull),n_train=len(train),n_positive_train=int(train.label.sum()),training_max_year=int(train.prediction_year.max()),training_max_feature_date=train.feature_max_date.max(),test_max_feature_date=test.feature_max_date.max(),single_class_fallback=bool(fallback),fit_seconds=time.monotonic()-start,input_signature=sig,warnings=sorted(set(str(w.message) for w in ws)),**metrics(test.label,pred))
    a.to_csv(dest,index=False);meta.write_text(json.dumps(rr,indent=2,allow_nan=True));return a,rr

def gain_tables(results,comparisons,prefix,strata=None):
    strata=strata or [];pair=[]
    for keys,a in results.groupby(['model']+strata+['prediction_year']):
        if not isinstance(keys,tuple):keys=(keys,)
        key=dict(zip(['model']+strata+['prediction_year'],keys));x=a.set_index('feature_group').PR_AUC
        for lo,hi in comparisons:pair.append(dict(**key,comparison=f'{hi}-{lo}',delta_AP=float(x[hi]-x[lo]),relative_gain=float((x[hi]-x[lo])/x[lo])))
    p=pd.DataFrame(pair);p.to_csv(O/f'{prefix}_gains_by_year.csv',index=False);rows=[]
    for keys,a in p.groupby(['model']+strata+['comparison']):
        if not isinstance(keys,tuple):keys=(keys,)
        key=dict(zip(['model']+strata+['comparison'],keys));a=a.sort_values('prediction_year');v=a.delta_AP.values;assert len(v)==8 and np.isfinite(v).all();boot=np.random.default_rng(SEED).choice(v,(20000,8),replace=True).mean(1)
        rows.append(dict(**key,mean_delta_AP=v.mean(),median_delta_AP=np.median(v),CI_low=np.quantile(boot,.025),CI_high=np.quantile(boot,.975),years_positive=int((v>0).sum()),years_total=8,leave_one_year_out_min_mean=min(np.delete(v,i).mean() for i in range(8)),leave_one_year_out_max_mean=max(np.delete(v,i).mean() for i in range(8))))
    s=pd.DataFrame(rows);s.to_csv(O/f'{prefix}_gain_summary.csv',index=False);return p,s

def summary(results,prefix,strata=None):
    out=results.groupby(['model','feature_group']+(strata or [])).agg(median_AP=('PR_AUC','median'),mean_AP=('PR_AUC','mean'),median_AP_over_prevalence=('AP_over_prevalence','median'),median_ROC_AUC=('ROC_AUC','median'),median_Brier=('Brier','median')).reset_index();out.to_csv(O/f'{prefix}_summary.csv',index=False);return out

def temporal(panel,groups,phase):
    rows=[]
    for year in range(2018,2026):
        tr=panel[panel.prediction_year<year];te=panel[panel.prediction_year==year]
        for model in ['Logistic','RF']:
            for group in groups:
                _,r=fit_predict(tr,te,model,group,phase,year);rows.append(r);print(phase,year,model,group,'AP',round(r['PR_AUC'],6),flush=True)
    return pd.DataFrame(rows)

if __name__=='__main__':
    # Phase A is completed and reproduced before constructing spatial folds.
    d=load_base();r=temporal(d,['E0','S0','M0','M1'],'A');r.to_csv(O/'decomposition_results_by_year.csv',index=False);gain_tables(r,[('E0','M0'),('S0','M0'),('M0','M1')],'decomposition');summary(r,'decomposition')
    checks=[]
    for a in r[r.feature_group.isin(['M0','M1'])].itertuples():
        got=pd.read_csv(O/f'predictions/A_{a.prediction_year}_fold-1_buffer0_{a.model}_{a.feature_group}.csv.gz');prior=pd.read_csv(R/f'outputs/task3/primary_3yr/{a.prediction_year}_{a.model}_{a.feature_group}_predictions.csv.gz');assert got.cell_id.tolist()==prior.cell_id.tolist();err=float(np.abs(got.probability-prior.probability).max());assert err<1e-10
        checks.append(dict(prediction_year=a.prediction_year,model=a.model,feature_group=a.feature_group,max_probability_difference=err))
    pd.DataFrame(checks).to_csv(O/'task3_M0_M1_reproduction.csv',index=False);print('PHASE A COMPLETE;32 Task3 prediction sets reproduced',flush=True)
