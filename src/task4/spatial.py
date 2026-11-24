from config import *
from engine import fit_predict,metrics,gain_tables,summary
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

def freeze_folds():
    assert (O/'task3_M0_M1_reproduction.csv').exists(),'Complete Phase A first'
    g=pd.read_csv(R/'data/processed/grid_4km.csv');dest=R/'data/processed/task4_spatial_folds.csv'
    km=KMeans(n_clusters=5,random_state=SEED,n_init=10,algorithm='lloyd');raw=km.fit_predict(g[['x','y']]);order=np.lexsort((km.cluster_centers_[:,0],km.cluster_centers_[:,1]));mapping={int(old):new+1 for new,old in enumerate(order)};g['spatial_fold']=[mapping[int(k)] for k in raw]
    if dest.exists():assert pd.read_csv(dest).equals(g[['cell_id','spatial_fold']]),'Frozen spatial folds changed'
    else:g[['cell_id','spatial_fold']].to_csv(dest,index=False)
    g.to_csv(D/'spatial_fold_geometry.csv',index=False)
    p=load_base().merge(g[['cell_id','spatial_fold']],on='cell_id',validate='many_to_one');held=p[p.prediction_year>=2018]
    a=g.groupby('spatial_fold').agg(n_grid_cells=('cell_id','size'),fraction_weight_sum=('california_fraction','sum'),centroid_x=('x','mean'),centroid_y=('y','mean')).reset_index();a['fraction_california_area']=a.fraction_weight_sum/g.california_fraction.sum();a=a.merge(held.groupby('spatial_fold').agg(candidate_cell_years=('cell_id','size'),positives=('label','sum')).reset_index(),on='spatial_fold');a.to_csv(O/'spatial_fold_audit.csv',index=False)
    held.groupby(['spatial_fold','prediction_year']).agg(n_candidates=('label','size'),n_positives=('label','sum')).to_csv(O/'spatial_fold_year_support.csv')
    distances=[]
    for k in range(1,6):
        dd=cKDTree(g.loc[g.spatial_fold==k,['x','y']]).query(g[['x','y']])[0]/1000;a=g[['cell_id','spatial_fold']].copy();a['heldout_fold']=k;a['distance_to_heldout_km']=dd;a['excluded_25km']=dd<=25;distances.append(a)
    pd.concat(distances).to_csv(D/'spatial_buffer_membership.csv.gz',index=False)
    if not (LOG/'spatial_fold_freeze.json').exists():(LOG/'spatial_fold_freeze.json').write_text(json.dumps(dict(sha256=sha(dest),timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),coordinates_only=True,kmeans=dict(n_clusters=5,n_init=10,algorithm='lloyd',random_state=SEED),fold_labels='centroid south to north; easting tie break',buffer_rule='center-to-center nearest ANY full-grid heldoutfold cell <=25km; heldoutregion itself always excluded'),indent=2))
    return p

def run_spatial(panel,groups,phase,buffers):
    memberships=pd.read_csv(D/'spatial_buffer_membership.csv.gz');foldrows=[];yearrows=[]
    for buffer in buffers:
        allowed={k:set(memberships.loc[(memberships.heldout_fold==k)&(memberships.spatial_fold!=k)&(~memberships.excluded_25km if buffer else True),'cell_id']) for k in range(1,6)}
        for year in range(2018,2026):
            for model in ['Logistic','RF']:
                for group in groups:
                    predictions=[]
                    for k in range(1,6):
                        tr=panel[(panel.prediction_year<year)&panel.cell_id.isin(allowed[k])];te=panel[(panel.prediction_year==year)&(panel.spatial_fold==k)]
                        assert not set(tr.cell_id)&set(panel.loc[panel.spatial_fold==k,'cell_id'])
                        if buffer:assert not set(tr.cell_id)&set(memberships.loc[(memberships.heldout_fold==k)&memberships.excluded_25km,'cell_id'])
                        pred,r=fit_predict(tr,te,model,group,phase,year,k,buffer);predictions.append(pred);foldrows.append(r)
                    pred=pd.concat(predictions).sort_values('cell_id');truth=panel[panel.prediction_year==year].sort_values('cell_id');assert pred.cell_id.tolist()==truth.cell_id.tolist() and not pred.cell_id.duplicated().any();assert np.array_equal(pred.label,truth.label)
                    pred.to_csv(O/f'predictions/{phase}_{year}_crossfit_buffer{buffer}_{model}_{group}.csv.gz',index=False)
                    r=dict(prediction_year=year,model=model,feature_group=group,buffer_km=buffer,**metrics(pred.label,pred.probability));yearrows.append(r);print(phase,'buffer',buffer,year,model,group,'crossfitAP',round(r['PR_AUC'],6),flush=True)
    pd.DataFrame(foldrows).to_csv(O/f'{phase}_fold_results.csv',index=False);return pd.DataFrame(yearrows)

if __name__=='__main__':
    p=freeze_folds();r=run_spatial(p,['M0','M1'],'B',[0,25]);r.to_csv(O/'spatial_temporal_results_by_year.csv',index=False);gain_tables(r,[('M0','M1')],'spatial_temporal',['buffer_km']);summary(r,'spatial_temporal',['buffer_km']);print('PHASE B COMPLETE',flush=True)
