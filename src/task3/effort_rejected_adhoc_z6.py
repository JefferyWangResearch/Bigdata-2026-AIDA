from setup import *
import requests,time,mercantile,mapbox_vector_tile
from pyproj import Transformer
from scipy.stats import spearmanr
WM=Transformer.from_crs(3857,3310,always_xy=True)
grid=pd.read_csv(R/'data/processed/grid_4km.csv'); idx={c:i for i,c in enumerate(grid.cell_id)}
# Entire CA grid, not the Task2 outcome-selected matching pool.
tiles=list(mercantile.tiles(-124.6,32.3,-114.0,42.2,6))
s=requests.Session()
def year(y):
    out=D/f'effort_{y}.csv'
    if out.exists():return pd.read_csv(out)
    counts=np.zeros(len(grid),dtype=np.int64)
    for tile in tiles:
        p=RAW/f'effort_{y}_{tile.z}_{tile.x}_{tile.y}.mvt'
        if not p.exists():
            for attempt in range(8):
                try:
                    rr=s.get(f'https://api.gbif.org/v2/map/occurrence/adhoc/{tile.z}/{tile.x}/{tile.y}.mvt',params=dict(taxonKey=7707728,country='US',year=y,occurrenceStatus='PRESENT',bin='square',squareSize=1,srs='EPSG:3857'),timeout=120)
                    if rr.status_code==429:time.sleep(max(60,int(rr.headers.get('Retry-After','60'))));continue
                    if rr.status_code==204: payload=b''
                    else:rr.raise_for_status();payload=rr.content;mapbox_vector_tile.decode(payload)
                    p.write_bytes(payload);stamp(p,rr.url);time.sleep(.6);break
                except Exception:
                    if attempt==7:raise
                    time.sleep(min(60,2**attempt))
            if not p.exists():raise RuntimeError('Tile failed '+str(p))
        if not p.stat().st_size:continue
        layer=mapbox_vector_tile.decode(p.read_bytes(),default_options={'y_coord_down':True})['occurrence'];ext=layer['extent'];bounds=mercantile.xy_bounds(tile)
        for f in layer['features']:
            geom=f['geometry'];a=np.array(geom['coordinates'][0] if geom['type']=='Polygon' else [geom['coordinates']]);xy=(a.min(axis=0)+a.max(axis=0))/2
            if not (0<=xy[0]<ext and 0<=xy[1]<ext):continue
            x=bounds.left+xy[0]/ext*(bounds.right-bounds.left); yy=bounds.top-xy[1]/ext*(bounds.top-bounds.bottom)
            xx,yy=WM.transform(x,yy);cell=f'{int(np.floor(xx/4000))}_{int(np.floor(yy/4000))}'
            if cell in idx:counts[idx[cell]]+=int(f['properties']['total'])
    d=grid[['cell_id']].copy();d['year']=y;d['count']=counts;d.to_csv(out,index=False);print('EFFORT',y,'total',counts.sum(),flush=True);return d
if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--audit-only',action='store_true');args=a.parse_args()
    test=year(2015);exact=pd.read_csv(R/'data/processed/observation_effort_cell_year.csv.gz');exact=exact[exact.year==2015]
    m=test.merge(exact[['cell_id','all_plant_observation_count']],on='cell_id');rho=spearmanr(np.log1p(m['count']),np.log1p(m.all_plant_observation_count)).statistic;ratio=m['count'].sum()/m.all_plant_observation_count.sum()
    audit=dict(year=2015,n_cells=len(m),spearman_log_count=float(rho),total_ratio=float(ratio),method='GBIF zoom6 quantized tile bin centers versus independent exact 4km queries; comparison only, no replacement',passed=bool(rho>=.95 and .9<=ratio<=1.1))
    (O/'effort_proxy_audit.json').write_text(json.dumps(audit,indent=2));m.to_csv(O/'effort_proxy_audit_cells.csv',index=False);print(audit,flush=True)
    if not audit['passed']:raise RuntimeError('Effort spatial proxy audit failed; investigate before modeling')
    if not args.audit_only:pd.concat([year(y) for y in range(2009,2025)],ignore_index=True).to_csv(D/'effort_panel.csv.gz',index=False)
