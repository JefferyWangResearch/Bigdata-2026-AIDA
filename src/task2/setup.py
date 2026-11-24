from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from scipy.spatial import cKDTree
R=ROOT;D=R/'data/processed/task2';O=R/'outputs/task2';RAW=R/'data/raw/task2'

def main():
    old={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src','outputs','data/processed','figures'] for p in (R/folder).rglob('*') if p.is_file() and 'task2' not in str(p.relative_to(R)).lower() and '__pycache__' not in str(p) and p.name!='.DS_Store'}
    (R/'logs/task2/task1_manifest.json').write_text(json.dumps(old,indent=2))
    grid=pd.read_csv(R/'data/processed/grid_4km.csv');occ=pd.read_csv(R/'data/interim/clean_unrestricted_precision.csv');mainocc=pd.read_csv(R/'data/processed/occurrences_with_cells.csv');pool=[];sources=[]
    for quality,records in [('primary',mainocc),('strict',assign(occ.loc[occ.uncertainty_1km_known]))]:
        first=records.groupby('cell_id').year.min();g=grid.copy();g['first_year']=g.cell_id.map(first)
        for year in range(2018,2026):
            s=g.loc[g.first_year.lt(year)].copy();s['year']=year;s['quality']=quality;sources.append(s)
            c=g.loc[~g.first_year.lt(year)].copy();tree=cKDTree(s[['x','y']]);dist=tree.query(c[['x','y']])[0]/1000;c['distance']=dist
            c=c.loc[c.distance.le(100)].copy();c['source_density']=np.array([len(x) for x in tree.query_ball_point(c[['x','y']],50000)])
            c['label']=c.first_year.eq(year).astype(int);c['year']=year;c['quality']=quality
            # fixed geographic thirds; no climate or wind information in pool construction.
            c['region']=pd.cut(c.center_lat,[-90,35,38,90],labels=['south','central','north']).astype(str)
            pos=c.loc[c.label.eq(1)];neg=c.loc[c.label.eq(0)]
            selected=set(pos.cell_id)
            for row in pos.itertuples():
                n=neg.loc[neg.region.eq(row.region)].copy()
                score=((np.log1p(n.distance)-np.log1p(row.distance))/.5)**2+((np.log1p(n.source_density)-np.log1p(row.source_density))/.7)**2+((n.x-row.x)/100000)**2+((n.y-row.y)/100000)**2
                selected.update(n.loc[score.nsmallest(8).index,'cell_id'])
            pool.append(c.loc[c.cell_id.isin(selected)])
    pd.concat(pool).to_csv(D/'candidate_pool.csv',index=False);pd.concat(sources).to_csv(D/'historical_sources.csv',index=False)
    ids=pd.concat(pool).cell_id.unique();grid.loc[grid.cell_id.isin(ids)].to_csv(D/'effort_query_cells.csv',index=False)
    print('Candidate cell-years',sum(len(x) for x in pool),'effort cells',len(ids))
if __name__=='__main__':main()
