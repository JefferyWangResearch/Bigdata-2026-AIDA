from setup import *
import requests,time,datetime,concurrent.futures,threading
LOCAL=threading.local()
def session():
    if not hasattr(LOCAL,"session"):
        LOCAL.session=requests.Session()
        LOCAL.session.headers["User-Agent"]="Salsola-directional-research-pilot/1.0"
    return LOCAL.session
from shapely.geometry import box
from shapely.ops import transform

def fetch(row):
    path=RAW/f'effort_{row.cell_id}.json';meta=RAW/f'effort_{row.cell_id}.meta.json'
    if not path.exists():
        poly=transform(INV.transform,box(row.gx*4000,row.gy*4000,(row.gx+1)*4000,(row.gy+1)*4000))
        params=dict(taxonKey=7707728,country='US',year='2015,2025',hasCoordinate='true',occurrenceStatus='PRESENT',geometry=poly.wkt,facet='year',facetLimit=100,limit=300)
        for attempt in range(12):
            try:
                r=session().get('https://api.gbif.org/v1/occurrence/search',params=params,timeout=90);
                if r.status_code==429:
                    wait=max(60,int(r.headers.get('Retry-After','60')));print('GBIF rate limit; backing off',wait,flush=True);time.sleep(wait);continue
                r.raise_for_status();d=r.json()
                assert 'facets' in d or d['count']==0
                path.write_text(json.dumps(d));meta.write_text(json.dumps(dict(url=r.url,retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),count=d['count'],sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
                time.sleep(.5)
                break
            except Exception:
                if attempt==11:raise
                time.sleep(min(60,2**attempt))
    d=json.loads(path.read_text());counts={int(v['name']):v['count'] for f in d.get('facets',[]) if f['field'].upper()=='YEAR' for v in f['counts']}
    assert sum(counts.values())==d['count']
    records=pd.DataFrame(d['results']);out=[]
    for y in range(2015,2026):
        sub=records.loc[records.year.eq(y)] if len(records) and 'year'in records else pd.DataFrame()
        unique=sub.speciesKey.nunique() if len(sub) and 'speciesKey'in sub else 0
        out.append(dict(cell_id=row.cell_id,year=y,all_plant_observation_count=counts.get(y,0),unique_species_count=unique,unique_species_is_lower_bound=d['count']>len(d['results']),records_sampled_for_richness=len(sub),log_observation_effort=np.log1p(counts.get(y,0)),query_total_records=d['count']))
    return out

def main():
    cells=pd.read_csv(D/'effort_query_cells.csv');results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        for k,out in enumerate(ex.map(fetch,cells.itertuples(index=False))):
            results.extend(out)
            if k%50==0:print('effort cells',k+1,'/',len(cells),flush=True)
    df=pd.DataFrame(results).sort_values(['cell_id','year'])
    df['effort_same_year']=df.all_plant_observation_count
    df['effort_previous_year']=df.groupby('cell_id').all_plant_observation_count.shift(1)
    df['effort_previous_3yr_mean']=df.groupby('cell_id').all_plant_observation_count.transform(lambda s:s.shift(1).rolling(3,min_periods=3).mean())
    df.to_csv(R/'data/processed/observation_effort_cell_year.csv.gz',index=False)
    print('DONE',len(df),flush=True)
if __name__=='__main__':main()
