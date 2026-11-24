from common import *
from scipy.spatial import cKDTree

def longest(years):
    best=[];run=[]
    for y in sorted(years):
        run=run+[int(y)] if run and y==run[-1]+1 else [int(y)]
        if len(run)>len(best):best=run.copy()
    return best

def annual(d):
    first=d.groupby('cell_id').year.min();a=d.groupby('year').agg(total_observations=('key','size'),occupied_cells=('cell_id','nunique'))
    a=a.reindex(range(int(d.year.min()),2027),fill_value=0)
    a['newly_observed_cells']=first.value_counts().reindex(a.index,fill_value=0);a.index.name='year';return a.reset_index(),first

def main():
    d=pd.read_csv(ROOT/'data/processed/occurrences_with_cells.csv');grid=pd.read_csv(ROOT/'data/processed/grid_4km.csv');a,first=annual(d)
    a.to_csv(ROOT/'outputs/observations_by_year.csv',index=False)
    grid['first_observation_year']=grid.cell_id.map(first)
    grid.to_csv(ROOT/'data/processed/first_observation_cells.csv',index=False)
    # Inspect complete modern years, retaining all older records as source history.
    allrows=[];summary=[]
    for y in range(2000,2026):
        prior=grid.first_observation_year.lt(y);c=grid.loc[~prior].copy();c['prediction_year']=y
        c['ever_observed_before_t']=False;c['new_occurrence_label']=c.first_observation_year.eq(y).astype(int)
        c['distance_nearest_previous_occurrence_km']=cKDTree(grid.loc[prior,['x','y']]).query(c[['x','y']])[0]/1000 if prior.any() else np.inf
        for radius in [25,50,100]:c[f'candidate_{radius}km']=c.distance_nearest_previous_occurrence_km.le(radius)
        allrows.append(c[['cell_id','prediction_year','first_observation_year','ever_observed_before_t','new_occurrence_label','distance_nearest_previous_occurrence_km','candidate_25km','candidate_50km','candidate_100km']])
        for radius in [0,25,50,100]:
            sub=c if radius==0 else c.loc[c[f'candidate_{radius}km']];n=int(sub.new_occurrence_label.sum())
            summary.append(dict(year=y,candidate_set='statewide' if radius==0 else f'{radius}km',positives=n,negative_candidates=len(sub)-n,n_candidates=len(sub),positive_rate=n/len(sub) if len(sub) else np.nan,excluded_new_cells=int(c.new_occurrence_label.sum())-n))
    labels=pd.concat(allrows);labels.to_csv(ROOT/'data/processed/next_year_labels.csv.gz',index=False)
    pd.DataFrame(summary).to_csv(ROOT/'outputs/candidate_counts_by_year.csv',index=False)
    counts=d.groupby(['cell_id','year']).size().rename('n_salsola_observations')
    # Explicit zero-filled panel only for modern audit interval; zeros mean no records.
    panel=pd.MultiIndex.from_product([grid.cell_id,range(2000,2027)],names=['cell_id','year']).to_frame(index=False)
    panel['n_salsola_observations']=counts.reindex(pd.MultiIndex.from_frame(panel),fill_value=0).values
    panel['presence']=panel.n_salsola_observations.gt(0).astype(int)
    panel.to_csv(ROOT/'data/processed/cell_year_panel.csv.gz',index=False)
    sensitivities=[];u=pd.read_csv(ROOT/'data/interim/clean_unrestricted_precision.csv')
    for name,sub,size in [('primary_4km',d,4000),('known_uncertainty_le_1km',u.loc[u.uncertainty_1km_known],4000),('unrestricted_precision',u,4000),('primary_8km_grid',d,8000)]:
        ad,f=annual(assign(sub,size));ad['definition']=name;sensitivities.append(ad)
    sens=pd.concat(sensitivities);sens.to_csv(ROOT/'outputs/precision_grid_sensitivity.csv',index=False)
    gates=[]
    for name,sub in sens.groupby('definition'):
        for threshold in [20,30]:
            run=longest(sub.loc[sub.year.between(2000,2025)&sub.newly_observed_cells.ge(threshold),'year'])
            gates.append(dict(definition=name,threshold=threshold,longest_consecutive_years=len(run),start=min(run) if run else None,end=max(run) if run else None))
    pd.DataFrame(gates).to_csv(ROOT/'outputs/temporal_support.csv',index=False)
    assert len(labels)==len(labels.drop_duplicates(['cell_id','prediction_year']))
    assert not ((labels.first_observation_year<labels.prediction_year)).any()
    assert labels.new_occurrence_label.sum()==first.between(2000,2025).sum()
    print(a.loc[a.year>=2000].to_string(index=False));print(pd.DataFrame(gates).to_string(index=False))
if __name__=='__main__':main()
