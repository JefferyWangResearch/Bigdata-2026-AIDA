from setup import *
from scipy.spatial.distance import cdist

def main():
    g=pd.read_csv(R/'data/processed/grid_4km.csv');primary=pd.read_csv(R/'data/processed/occurrences_with_cells.csv');strict=pd.read_csv(R/'data/interim/clean_unrestricted_precision.csv');strict=assign(strict[strict.uncertainty_1km_known==True]); audits=[]
    for quality,occ in [('primary',primary),('strict',strict)]:
        first=occ.groupby('cell_id').year.min().rename('first_year');gg=g.merge(first,on='cell_id',how='left');blocks=[]
        for year in range(2012,2026):
            sources=gg[gg.first_year<year];cand=gg[gg.first_year.isna()|(gg.first_year>=year)].copy();dist=cdist(cand[['x','y']],sources[['x','y']])/1000
            cand['distance_nearest_source_km']=dist.min(axis=1);cand['source_count_25km']=(dist<=25).sum(axis=1);cand['source_count_50km']=(dist<=50).sum(axis=1);cand['source_pressure_L25']=np.exp(-dist/25).sum(axis=1)
            cand=cand[cand.distance_nearest_source_km<=50].copy();cand['label']=(cand.first_year==year).astype(int);cand['prediction_year']=year;cand['feature_max_date']=f'{year-1}-12-31';cand['source_max_year']=int(sources.first_year.max());cand['n_historical_sources']=len(sources)
            assert cand.source_max_year.max()<year;assert not (cand.first_year<year).any()
            audits.append(dict(quality=quality,year=year,n_candidates=len(cand),n_positive=int(cand.label.sum()),n_sources=len(sources),max_source_year=int(sources.first_year.max())))
            blocks.append(cand.drop(columns=['first_year']))
        pd.concat(blocks).to_csv(D/f'spatial_{quality}.csv.gz',index=False)
    pd.DataFrame(audits).to_csv(O/'candidate_audit.csv',index=False)
    expected=pd.read_csv(R/'data/processed/next_year_labels.csv.gz');got=pd.read_csv(D/'spatial_primary.csv.gz');expected=expected[(expected.prediction_year>=2012)&expected.candidate_50km]
    merged=got.merge(expected[['cell_id','prediction_year','new_occurrence_label']],on=['cell_id','prediction_year'],how='outer',indicator=True)
    assert merged['_merge'].eq('both').all();assert (merged.label==merged.new_occurrence_label).all()
    print('SPATIAL PASS',len(got),flush=True)
if __name__=='__main__':main()
