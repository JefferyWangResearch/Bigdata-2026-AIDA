from setup import *
import requests,time,mercantile,mapbox_vector_tile
from pyproj import Transformer
from scipy.stats import spearmanr
from concurrent.futures import ThreadPoolExecutor
WM=Transformer.from_crs(3857,3310,always_xy=True)
grid=pd.read_csv(R/'data/processed/grid_4km.csv');idx={c:i for i,c in enumerate(grid.cell_id)}
# density endpoint preserves much finer locations than adhoc's geohash aggregates;
# verbose per-year properties let us build every year's counts from the same snapshot.
# Cover all four corners of every state-intersecting grid cell; skip irrelevant ocean tiles.
tile_set=set()
for dx,dy in [(-2000,-2000),(-2000,2000),(2000,-2000),(2000,2000)]:
    lo,la=INV.transform(grid.x.values+dx,grid.y.values+dy)
    tile_set.update(mercantile.tile(a,b,8) for a,b in zip(lo,la))
tiles=sorted(tile_set)

def fetch(tile):
    p=RAW/f'effort_density_2009_2024_{tile.z}_{tile.x}_{tile.y}.mvt'
    if not p.exists():
        for attempt in range(8):
            try:
                rr=requests.get(f'https://api.gbif.org/v2/map/occurrence/density/{tile.z}/{tile.x}/{tile.y}.mvt',params=dict(taxonKey=7707728,country='US',year='2009,2024',verbose='true',srs='EPSG:3857'),timeout=(30,180))
                if rr.status_code==429:time.sleep(max(60,int(rr.headers.get('Retry-After','60'))));continue
                if rr.status_code==204:payload=b''
                else:rr.raise_for_status();payload=rr.content;mapbox_vector_tile.decode(payload)
                p.write_bytes(payload);stamp(p,rr.url);time.sleep(.5);break
            except Exception as e:
                print('effort retry',tile,str(e)[:100],flush=True)
                if attempt==7:raise
                time.sleep(min(60,2**attempt))
        if not p.exists():raise RuntimeError(str(p))
    return tile,p

def main():
    counts=np.zeros((16,len(grid)),dtype=np.int64);quant=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for n,(tile,p) in enumerate(pool.map(fetch,tiles)):
            if not p.stat().st_size:continue
            layer=mapbox_vector_tile.decode(p.read_bytes(),default_options={'y_coord_down':True})['occurrence'];ext=layer['extent'];bounds=mercantile.xy_bounds(tile);features=layer['features']
            xy=np.array([f['geometry']['coordinates'] for f in features]);assert xy.ndim==2
            keep=(xy[:,0]>=0)&(xy[:,0]<ext)&(xy[:,1]>=0)&(xy[:,1]<ext)
            xx=bounds.left+xy[:,0]/ext*(bounds.right-bounds.left);yy=bounds.top-xy[:,1]/ext*(bounds.top-bounds.bottom);xx,yy=WM.transform(xx,yy)
            cells=[f'{x}_{y}' for x,y in zip(np.floor(xx/4000).astype(int),np.floor(yy/4000).astype(int))];ix=np.array([idx.get(c,-1) for c in cells]);keep &=ix>=0
            for y in range(2009,2025):
                vals=np.array([int(f['properties'].get(str(y),0)) for f in features]);np.add.at(counts[y-2009],ix[keep],vals[keep])
            quant.append(dict(tile_x=tile.x,tile_y=tile.y,zoom=tile.z,extent=ext,n_features=len(features),webmercator_quantization_m=(bounds.right-bounds.left)/ext))
            print('EFFORT TILE',n+1,'/',len(tiles),flush=True)
    rows=[]
    for i,y in enumerate(range(2009,2025)):
        d=grid[['cell_id']].copy();d['year']=y;d['count']=counts[i];rows.append(d)
    panel=pd.concat(rows);panel.to_csv(D/'effort_panel.csv.gz',index=False);pd.DataFrame(quant).to_csv(O/'effort_tile_resolution.csv',index=False)
    exact=pd.read_csv(R/'data/processed/observation_effort_cell_year.csv.gz');exact=exact[exact.year<=2024];m=panel.merge(exact[['cell_id','year','all_plant_observation_count']],on=['cell_id','year'])
    audits=[]
    for y,a in m.groupby('year'):
        rho=spearmanr(a['count'],a.all_plant_observation_count).statistic;ratio=a['count'].sum()/a.all_plant_observation_count.sum();audits.append(dict(year=int(y),n_cells=len(a),spearman_log_count=float(rho),total_ratio=float(ratio),passed=bool(rho>=.95 and .9<=ratio<=1.1)))
    pd.DataFrame(audits).to_csv(O/'effort_proxy_audit_by_year.csv',index=False);m.to_csv(O/'effort_proxy_audit_cells.csv',index=False)
    audit=dict(method='GBIF density zoom8 point tiles, verbose year counts; uniform quantized spatial proxy; independent exact4km Task2 comparison; no selective replacement',passed=all(a['passed'] for a in audits),yearly=audits)
    (O/'effort_proxy_audit.json').write_text(json.dumps(audit,indent=2));print(audit,flush=True)
    if not audit['passed']:raise RuntimeError('Effort audit failed')
if __name__=='__main__':main()
