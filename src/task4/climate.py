from config import *
from engine import temporal,gain_tables,summary
from spatial import run_spatial

def normals_panel():
    assert (O/'spatial_temporal_gain_summary.csv').exists(),'Complete Phase B before C'
    g=pd.read_csv(R/'data/processed/grid_4km.csv');norm=g[['cell_id']].copy();audit=[]
    for code,name in zip(['tmmx','pr','vpd','vs'],NORMAL):
        annual=[]
        for y in range(1981,2011):
            a=np.load(T3/f'daily_{code}_{y}.npy',mmap_mode='r');assert a.shape[1]==len(g);valid=np.isfinite(a).all(0);stat=np.sum(a,axis=0,dtype=np.float64) if code=='pr' else np.mean(a,axis=0,dtype=np.float64);stat[~valid]=np.nan;annual.append(stat)
        annual=np.stack(annual);mean=np.mean(annual,axis=0);norm[name]=mean
        old=pd.read_csv(T3/f'climatology_{code}_1981_2010.csv');assert old.cell_id.tolist()==g.cell_id.tolist();reference=old.annual_total_mean if code=='pr' else old.daily_mean
        audit.append(dict(variable=code,normal=name,n_complete_cells=int(np.isfinite(mean).sum()),max_absolute_difference_vs_task3_pooled_daily_or_float32_sum=float(np.nanmax(np.abs(mean-reference))),definition='equal weight30annualtotals' if code=='pr' else 'equal weight30annualmeans; differs slightly from pooled dailymean due leap-year weights'))
    norm.to_csv(D/'climate_normals_1981_2010.csv',index=False);pd.DataFrame(audit).to_csv(O/'climate_normal_audit.csv',index=False)
    p=load_base().merge(norm,on='cell_id',validate='many_to_one')
    for raw,normal,anom in zip(CL,NORMAL,ANOM):p[anom]=p[raw]-p[normal];assert np.allclose(p[normal]+p[anom],p[raw],equal_nan=True)
    p=p.merge(pd.read_csv(R/'data/processed/task4_spatial_folds.csv'),on='cell_id',validate='many_to_one');p.to_csv(D/'climate_decomposition_panel.csv.gz',index=False)
    rows=[]
    for name in NORMAL+ANOM:rows.append(dict(feature=name,type='normal' if name in NORMAL else 'anomaly',time_window='1981–2010' if name in NORMAL else 'calendar t-1 minus1981–2010normal',definition='equal-weight mean of30complete annual summaries' if name in NORMAL else 'Task3 preceding-year absolute variable minus matched normal',missing_fraction=float(p[name].isna().mean())))
    pd.DataFrame(rows).to_csv(O/'climate_feature_dictionary.csv',index=False);return p

if __name__=='__main__':
    p=normals_panel();r=temporal(p,['CN','CA','CNA'],'C')
    c0=pd.read_csv(O/'decomposition_results_by_year.csv');c0=c0[c0.feature_group=='M0'].copy();c0['feature_group']='C0';c0['phase']='C0_from_verified_A_M0';r=pd.concat([c0,r],ignore_index=True);r.to_csv(O/'climate_decomposition_results_by_year.csv',index=False);gain_tables(r,[('C0','CN'),('C0','CA'),('CN','CNA')],'climate_decomposition');summary(r,'climate_decomposition')
    sr=run_spatial(p,['CN','CNA'],'C_spatial',[0]);sr.to_csv(O/'climate_spatial_results_by_year.csv',index=False);gain_tables(sr,[('CN','CNA')],'climate_spatial',['buffer_km']);summary(sr,'climate_spatial',['buffer_km']);print('PHASE C COMPLETE',flush=True)
