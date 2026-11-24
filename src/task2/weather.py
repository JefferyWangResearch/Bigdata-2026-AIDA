from setup import *
import requests,time,datetime
from netCDF4 import Dataset,num2date
BASE='https://tds-proxy.nkn.uidaho.edu/thredds/ncss/grid/agg_met_{v}_1979_CurrentYear_CONUS.nc'
VARS={'vs':'daily_mean_wind_speed','th':'daily_mean_wind_direction'}

def direction_tests():
    def uv(fr,speed=1):a=np.deg2rad(fr);return -speed*np.sin(a),-speed*np.cos(a)
    assert np.allclose(uv(270),(1,0),atol=1e-12)
    assert np.allclose(uv(0),(0,-1),atol=1e-12)
    assert np.allclose(uv(90),(-1,0),atol=1e-12)
    assert np.isclose(np.cos(np.deg2rad((270+180)%360-90)),1)
    (R/'logs/wind_direction_validation.json').write_text(json.dumps(dict(status='PASS',from_west_transport_east=list(uv(270)),from_north_transport_south=list(uv(0)),convention='degrees clockwise from north, meteorological FROM; TO=(FROM+180)%360; east=-s*sin(FROM), north=-s*cos(FROM)',bearing='WGS84 geodesic source to target',gate='Tests before exposure calculations'),indent=2))

def main():
    direction_tests();sources=pd.read_csv(D/'historical_sources.csv').drop_duplicates('cell_id').sort_values('cell_id');pieces=[];manifest=[];mapping=[]
    for y in range(2016,2025):
        blocks={}
        for code,var in VARS.items():
            p=RAW/f'gridmet_{code}_{y}.nc';params=dict(var=var,north=42.1,south=32.4,west=-124.5,east=-114.1,horizStride=1,time_start=f'{y}-01-01T00:00:00Z',time_end=f'{y}-12-31T23:59:59Z',timeStride=1,accept='netcdf')
            if not p.exists():
                for attempt in range(4):
                    try:
                        rr=requests.get(BASE.format(v=code),params=params,timeout=(30,300));rr.raise_for_status();p.with_suffix('.part').write_bytes(rr.content)
                        with Dataset(p.with_suffix('.part')) as nc:
                            t=nc.variables['day'];dates=pd.to_datetime([str(x)[:10] for x in num2date(t[:],t.units)]);assert dates.equals(pd.date_range(f'{y}-01-01',f'{y}-12-31'))
                        p.with_suffix('.part').replace(p)
                        p.with_suffix('.meta.json').write_text(json.dumps(dict(url=rr.url,retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
                        break
                    except Exception as e:
                        print('retry',code,y,repr(e),flush=True)
                        if attempt==3:raise
                        time.sleep(2**attempt)
            with Dataset(p) as nc:
                t=nc.variables['day'];dates=pd.to_datetime([str(x)[:10] for x in num2date(t[:],t.units)]);assert dates.equals(pd.date_range(f'{y}-01-01',f'{y}-12-31'))
                lat=nc.variables['lat'][:];lon=nc.variables['lon'][:];v=nc.variables[var];arr=np.ma.filled(v[:],np.nan).astype(np.float32)
                meta=json.loads(p.with_suffix('.meta.json').read_text());meta.update(variable=var,units=v.units,standard_name=getattr(v,'standard_name',''),long_name=getattr(v,'long_name',''),dates=[str(dates.min()),str(dates.max())]);manifest.append(meta)
                if code=='th':assert 'direction' in v.description.lower() and 'clockwise from north' in v.units.lower()
                vals=[]
                for row in sources.itertuples():
                    iy=int(np.abs(lat-row.center_lat).argmin());ix=int(np.abs(lon-row.center_lon).argmin());vals.append(arr[:,iy,ix]);
                    if y==2016 and code=='vs':mapping.append(dict(cell_id=row.cell_id,gridmet_lat=float(lat[iy]),gridmet_lon=float(lon[ix]),method='nearest native gridMET cell; no coastal imputation'))
                blocks[code]=np.stack(vals,axis=1)
            (R/'logs/task2/weather_manifest.json').write_text(json.dumps(manifest,indent=2));print(code,y,p.stat().st_size,flush=True)
        np.savez_compressed(D/f'wind_sources_{y}.npz',dates=dates.astype(str).values.astype('U10'),cell_ids=sources.cell_id.values.astype(str),speed=blocks['vs'],direction_from=blocks['th'])
    pd.DataFrame(mapping).to_csv(O/'weather_source_mapping.csv',index=False)
    print('WEATHER COMPLETE',flush=True)
if __name__=='__main__':main()
