from setup import *
import statsmodels.api as sm

B=2000;N_PERM=500
FEATURES=['log_distance','log_density','log_effort','x','y']

def covariates(df):
    d=df.copy();d['log_distance']=np.log1p(d.distance);d['log_density']=np.log1p(d.source_density);d['log_effort']=np.log1p(d.effort_previous_3yr_mean);d['log_same_effort']=np.log1p(d.effort_same_year)
    return d

def match(d,effort=True):
    d=d.reset_index(drop=True);features=FEATURES if effort else [f for f in FEATURES if f!='log_effort'];sd=d[features].std().replace(0,1);z=(d[features]-d[features].mean())/sd
    pairs=[]
    for i,row in d.loc[d.label.eq(1)].iterrows():
        neg=d.index[d.label.eq(0)&d.region.eq(row.region)]
        if len(neg)<3:continue
        scores=((z.loc[neg]-z.loc[i])**2).sum(axis=1);js=scores.nsmallest(3).index.values
        pairs.append((i,js))
    assert pairs,'No matches';return d,np.array([p[0] for p in pairs]),np.array([p[1] for p in pairs])

def fit_regression(d,effort_col='log_effort',annual=False):
    cols=['log_distance','log_density',effort_col,'x','y','mean_alignment'];x=d[cols].copy();x=(x-x.mean())/x.std().replace(0,1)
    cats=['region'] if annual else ['region','year'];extra=pd.get_dummies(d[cats].astype(str),drop_first=True,dtype=float);x=sm.add_constant(pd.concat([x,extra],axis=1),has_constant='add')
    try:
        fit=sm.GLM(d.label,x,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':d.cell_id},maxiter=100)
        ci=fit.conf_int().loc['mean_alignment'];return dict(coefficient=float(fit.params['mean_alignment']),ci_low=float(ci.iloc[0]),ci_high=float(ci.iloc[1]),p=float(fit.pvalues['mean_alignment']),converged=bool(fit.converged),n=len(d),n_positive=int(d.label.sum()))
    except Exception as exc:return dict(error=str(exc),n=len(d),n_positive=int(d.label.sum()))

def main():
    exp=pd.read_csv(D/'exposures.csv.gz');selected=pd.read_csv(D/'candidate_pool.csv')[['cell_id','year','quality']];exp=exp.merge(selected,on=['cell_id','year','quality'],validate='many_to_one');eff=pd.read_csv(R/'data/processed/observation_effort_cell_year.csv.gz');d=covariates(exp.merge(eff[['cell_id','year','effort_same_year','effort_previous_year','effort_previous_3yr_mean']],on=['cell_id','year'],validate='many_to_one'))
    assert d.log_effort.notna().all()
    yearly=[];summary=[];pairrows=[];balances=[];permrows=[];reg=[]
    for key,config in d.groupby(['quality','radius','window','L'],sort=True):
        quality,radius,window,L=key;info=dict(coordinate_quality_definition=quality,radius=radius,wind_window=window,L=L);deltas=[];nulls=[];bootyears=[];pooledpos=[];pooledneg=[];wspdeltas=[];wspbootyears=[];wspallpos=[];wspallneg=[];npositives=0
        for year,sub in config.groupby('year'):
            sub,pi,ni=match(sub);A=sub.mean_alignment.values;P=sub.perpendicular_alignment.values;W=sub.WSP.values
            pm=float(np.median(A[pi]));nm=float(np.median(A[ni]));delta=pm-nm
            rng=np.random.default_rng(SEED+int(year));rot=rng.uniform(0,2*np.pi,N_PERM)
            rotated=A[:,None]*np.cos(rot)-P[:,None]*np.sin(rot)
            nul=np.median(rotated[pi],axis=0)-np.median(rotated[ni.reshape(-1)],axis=0);nulls.append(nul)
            ix=rng.integers(0,len(pi),size=(B,len(pi)));bs=np.median(A[pi[ix]],axis=1)-np.median(A[ni[ix]].reshape(B,-1),axis=1);bootyears.append(bs)
            wbs=np.median(W[pi[ix]],axis=1)-np.median(W[ni[ix]].reshape(B,-1),axis=1);wspbootyears.append(wbs);wspallpos.extend(W[pi]);wspallneg.extend(W[ni.reshape(-1)])
            wlo,whi=np.quantile(wbs,[.025,.975])
            lo,hi=np.quantile(bs,[.025,.975]);wd=float(np.median(W[pi])-np.median(W[ni]));wspdeltas.append(wd)
            yearly.append(dict(**info,year=year,n_positive=len(pi),n_matched_negative=len(ni)*3,n_unique_matched_negative=len(np.unique(ni)),positive_alignment_median=pm,negative_alignment_median=nm,delta_alignment=delta,ci_low=lo,ci_high=hi,positive_WSP_median=np.median(W[pi]),negative_WSP_median=np.median(W[ni]),delta_WSP=wd,WSP_ci_low=wlo,WSP_ci_high=whi,reversed_delta_alignment=-delta,permutation_p=(1+(nul>=delta).sum())/(N_PERM+1)))
            deltas.append(delta);pooledpos.extend(A[pi]);pooledneg.extend(A[ni.reshape(-1)]);npositives+=len(pi)
            for feature in FEATURES:
                a=sub[feature].values;den=np.sqrt((np.var(a[pi],ddof=1)+np.var(a[ni],ddof=1))/2)
                balances.append(dict(**info,year=year,feature=feature,positive_mean=np.mean(a[pi]),negative_mean=np.mean(a[ni]),standardized_mean_difference=(np.mean(a[pi])-np.mean(a[ni]))/den if den else 0))
            if quality=='primary' and radius==50 and L==25 and window=='Sep-Jan':
                for p,ns in zip(pi,ni):
                    for n in ns:pairrows.append(dict(year=year,positive_cell=sub.loc[p,'cell_id'],negative_cell=sub.loc[n,'cell_id'],positive_alignment=A[p],negative_alignment=A[n],positive_WSP=W[p],negative_WSP=W[n],positive_effort=sub.loc[p,'effort_previous_3yr_mean'],negative_effort=sub.loc[n,'effort_previous_3yr_mean']))
                for mode,col in [('lagged_effort','log_effort'),('same_year_effort_retrospective','log_same_effort')]:reg.append(dict(year=year,mode=mode,**fit_regression(sub,col,True)))
        observed=float(np.mean(deltas));null=np.mean(nulls,axis=0);rng=np.random.default_rng(SEED)
        bootmat=np.array(bootyears).T;which=rng.integers(0,len(deltas),size=(B,len(deltas)));boots=bootmat[np.arange(B)[:,None],which].mean(axis=1);lo,hi=np.quantile(boots,[.025,.975]);p=(1+int((null>=observed).sum()))/(N_PERM+1)
        wboot=np.array(wspbootyears).T[np.arange(B)[:,None],which].mean(axis=1);wlo,whi=np.quantile(wboot,[.025,.975])
        summary.append(dict(**info,WSP_bootstrap_CI_low=wlo,WSP_bootstrap_CI_high=whi,pooled_WSP_median_difference=float(np.median(wspallpos)-np.median(wspallneg)),observed_effect=observed,bootstrap_CI_low=lo,bootstrap_CI_high=hi,permutation_p=p,years_positive=int((np.array(deltas)>0).sum()),n_years=len(deltas),n_positive=npositives,pooled_median_difference=float(np.median(pooledpos)-np.median(pooledneg)),mean_yearly_delta_WSP=float(np.mean(wspdeltas)),reversed_effect=-observed,null_median=np.median(null)))
        if quality=='primary' and radius==50 and L==25 and window=='Sep-Jan':
            for k,value in enumerate(null):permrows.append(dict(permutation=k+1,statistic=value,observed_statistic=observed,empirical_p=p))
            for mode,col in [('lagged_effort','log_effort'),('same_year_effort_retrospective','log_same_effort')]:reg.append(dict(year='pooled',mode=mode,**fit_regression(config,col)))
        print('analysis',key,observed,p,flush=True)
    pd.DataFrame(yearly).to_csv(O/'alignment_all_definitions_by_year.csv',index=False)
    y=pd.DataFrame(yearly);primary=y.loc[y.coordinate_quality_definition.eq('primary')&y.radius.eq(50)&y.L.eq(25)&y.wind_window.eq('Sep-Jan')]
    primary.to_csv(R/'outputs/task2_alignment_by_year.csv',index=False)
    pd.DataFrame(summary).to_csv(R/'outputs/task2_sensitivity_results.csv',index=False);pd.DataFrame(permrows).to_csv(R/'outputs/wind_permutation_results.csv',index=False)
    pd.DataFrame(pairrows).to_csv(D/'matched_pairs_primary.csv',index=False);pd.DataFrame(balances).to_csv(O/'matching_balance.csv',index=False);pd.DataFrame(reg).to_csv(O/'effort_regression.csv',index=False)
    # Expose all four requested alignment definitions, not only the primary one.
    metrics=[];primarydata=d.loc[d.quality.eq('primary')&d.radius.eq(50)&d.L.eq(25)&d.window.eq('Sep-Jan')]
    for year,sub in primarydata.groupby('year'):
        for strategy in ['spatial_only','spatial_and_effort']:
            sub,pi,ni=match(sub,effort=strategy=='spatial_and_effort')
            for metric in ['mean_alignment','positive_alignment','wind_speed_weighted_alignment','high_wind_weighted_alignment','WSP','reversed_WSP']:
                x=sub[metric].values;metrics.append(dict(year=year,strategy=strategy,metric=metric,positive_median=np.median(x[pi]),negative_median=np.median(x[ni]),difference=np.median(x[pi])-np.median(x[ni])))
    pd.DataFrame(metrics).to_csv(O/'alignment_metrics_effort_sensitivity.csv',index=False)
    primarydata.to_csv(D/'primary_analysis_pool.csv',index=False)
if __name__=='__main__':main()
