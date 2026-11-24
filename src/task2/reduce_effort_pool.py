"""Pre-outcome engineering amendment due persistent GBIF rate limits; all positives retained."""
from setup import *
import datetime

def main():
    original=D/'candidate_pool_full8.csv'
    if not original.exists():(D/'candidate_pool.csv').rename(original)
    pool=pd.read_csv(original);parts=[]
    for (quality,year),c in pool.groupby(['quality','year']):
        selected=set(c.loc[c.label.eq(1),'cell_id']);neg=c.loc[c.label.eq(0)]
        for row in c.loc[c.label.eq(1)].itertuples():
            n=neg.loc[neg.region.eq(row.region)]
            score=((np.log1p(n.distance)-np.log1p(row.distance))/.5)**2+((np.log1p(n.source_density)-np.log1p(row.source_density))/.7)**2+((n.x-row.x)/100000)**2+((n.y-row.y)/100000)**2
            selected.update(n.loc[score.nsmallest(3).index,'cell_id'])
        parts.append(c.loc[c.cell_id.isin(selected)])
    result=pd.concat(parts);result.to_csv(D/'candidate_pool.csv',index=False)
    grid=pd.read_csv(R/'data/processed/grid_4km.csv');cells=grid.loc[grid.cell_id.isin(result.cell_id)];cells.to_csv(D/'effort_query_cells.csv',index=False)
    allmeta=list(RAW.glob('effort_*.meta.json'));cached=sum((RAW/f'effort_{c}.json').exists() for c in cells.cell_id)
    note=dict(amended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='Persistent GBIF 429; before any positive-negative wind comparison',original_pool_cell_years=len(pool),new_pool_cell_years=len(result),original_positives=int(pool.label.sum()),retained_positives=int(result.label.sum()),n_effort_cells=len(cells),cached_required_cells=cached,remaining=len(cells)-cached,selection='reduce spatial prescreen from8 to3 neighbors per positive; final effort matching still3 with replacement among year-region pool; all positive cells retained')
    assert note['original_positives']==note['retained_positives'];(R/'logs/task2/effort_sampling_amendment.json').write_text(json.dumps(note,indent=2));print(note)
if __name__=='__main__':main()
