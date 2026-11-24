from setup import *
from scipy.spatial.distance import cdist
from model import run,summaries

def main():
    g=pd.read_csv(R/'data/processed/grid_4km.csv');occ=pd.read_csv(R/'data/processed/occurrences_with_cells.csv');first=occ.groupby('cell_id').year.min().rename('first_year');g=g.merge(first,on='cell_id',how='left');labels=pd.read_csv(R/'data/processed/next_year_labels.csv.gz')
    audits=[]
    for radius in [25,100]:
        blocks=[]
        for year in range(2012,2026):
            hist=g[g.first_year<year];a=g[g.first_year.isna()|(g.first_year>=year)].copy();dist=cdist(a[['x','y']],hist[['x','y']])/1000;a['distance_nearest_source_km']=dist.min(1);a['source_count_25km']=(dist<=25).sum(1);a['source_count_50km']=(dist<=50).sum(1);a['source_pressure_L25']=np.exp(-dist/25).sum(1);a=a[a.distance_nearest_source_km<=radius].copy();a['label']=(a.first_year==year).astype(int);a['prediction_year']=year;a['feature_max_date']=f'{year-1}-12-31';a['source_max_year']=int(hist.first_year.max());a['n_historical_sources']=len(hist);blocks.append(a.drop(columns='first_year'))
            ref=labels[(labels.prediction_year==year)&labels[f'candidate_{radius}km']];m=a.merge(ref[['cell_id','new_occurrence_label']],on='cell_id',how='outer',indicator=True);assert m['_merge'].eq('both').all() and (m.label==m.new_occurrence_label).all();audits.append(dict(radius_km=radius,prediction_year=year,n_test=len(a),n_positive=int(a.label.sum())))
        pd.concat(blocks).to_csv(D/f'spatial_primary_r{radius}.csv.gz',index=False);r=run(f'primary_r{radius}','3yr');summaries(r,f'_radius{radius}');assert len(r)==48;assert (r.training_max_year==r.prediction_year-1).all()
    pd.DataFrame(audits).to_csv(O/'radius_candidate_audit.csv',index=False)
    rows=[]
    for radius in [25,50,100]:
        p=O/('incremental_gain_summary.csv' if radius==50 else f'incremental_gain_summary_radius{radius}.csv');d=pd.read_csv(p);d['radius_km']=radius;rows.append(d)
    pd.concat(rows).to_csv(O/'radius_incremental_gains.csv',index=False)
    (LOG/'radius_validation.json').write_text(json.dumps(dict(status='PASS',radii=[25,100],primary_radius=50,candidate_labels_match_task1=True,models_per_radius=48,settings_unchanged=True),indent=2))
if __name__=='__main__':main()
