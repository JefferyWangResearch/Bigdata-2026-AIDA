from setup import *
import requests,time,calendar,warnings
from netCDF4 import Dataset,num2date
from concurrent.futures import ProcessPoolExecutor
VARS={'tmmx':'daily_maximum_temperature','pr':'precipitation_amount','vpd':'daily_mean_vapor_pressure_deficit','vs':'daily_mean_wind_speed'}
BASE='https://tds-proxy.nkn.uidaho.edu/thredds/ncss/grid/agg_met_{v}_1979_CurrentYear_CONUS.nc'

def dates_check(p,y):
    with Dataset(p) as nc:
        t=nc.variables['day'];d=pd.to_datetime([str(x)[:10] for x in num2date(t[:],t.units)])
        assert d.equals(pd.date_range(f'{y}-01-01',f'{y}-12-31')),str(d[[0,-1]])

def fetch(code,y,s):
    old=R/f'data/raw/task2/gridmet_{code}_{y}.nc';p=old if old.exists() else RAW/f'gridmet_{code}_{y}.nc'
    if not p.exists():
        params=dict(var=VARS[code],north=42.1,south=32.4,west=-124.5,east=-114.1,horizStride=1,time_start=f'{y}-01-01T00:00:00Z',time_end=f'{y}-12-31T23:59:59Z',timeStride=1,accept='netcdf')
        for attempt in range(8):
            try:
                rr=s.get(BASE.format(v=code),params=params,timeout=(30,300));rr.raise_for_status();part=p.with_suffix('.part');part.write_bytes(rr.content);dates_check(part,y);part.replace(p);stamp(p,rr.url);break
            except Exception as e:
                print('retry',code,y,str(e)[:200],flush=True)
                if attempt==7:raise
                time.sleep(min(60,2**attempt))
    dates_check(p,y);return p

def work(code):
    done=D/f'climate_{code}.csv.gz'
    if done.exists():return str(done)
    grid=pd.read_csv(R/'data/processed/grid_4km.csv');s=requests.Session();manifest=[]
    for y in range(1981,2025):
        dest=D/f'daily_{code}_{y}.npy';p=fetch(code,y,s)
        with Dataset(p) as nc:
            lat=nc.variables['lat'][:];lon=nc.variables['lon'][:];v=nc.variables[VARS[code]]
            iy=np.abs(lat[:,None]-grid.center_lat.values[None,:]).argmin(axis=0);ix=np.abs(lon[:,None]-grid.center_lon.values[None,:]).argmin(axis=0)
            if not dest.exists():
                arr=np.ma.filled(v[:],np.nan).astype('float32')[:,iy,ix]
                if code=='tmmx':
                    assert v.units.lower() in ['k','kelvin'];arr-=273.15
                np.save(dest,arr)
            meta=json.loads(p.with_suffix('.meta.json').read_text());meta.update(code=code,year=y,units=v.units,path=str(p.relative_to(R)));manifest.append(meta)
            if y==1981:
                m=grid[['cell_id','center_lon','center_lat']].copy();m['gridmet_lon']=lon[ix];m['gridmet_lat']=lat[iy];m.to_csv(O/f'gridmet_mapping_{code}.csv',index=False)
        (LOG/f'climate_manifest_{code}.json').write_text(json.dumps(manifest,indent=2));print('CLIMATE',code,y,flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        # One cell chunk at a time; never hold all thirty years in RAM.
        arrays=[np.load(D/f'daily_{code}_{y}.npy',mmap_mode='r') for y in range(1981,2011)]
        p95=np.zeros(len(grid));mean=np.zeros(len(grid));sd=np.zeros(len(grid));annual=[]
        for k in range(0,len(grid),256):
            a=np.concatenate([a[:,k:k+256] for a in arrays]);p95[k:k+256]=np.nanquantile(a,.95,axis=0);mean[k:k+256]=np.nanmean(a,axis=0);sd[k:k+256]=np.nanstd(a,axis=0,ddof=1)
        for a in arrays:
            valid=np.isfinite(a).all(axis=0);tot=np.sum(a,axis=0);tot[~valid]=np.nan;annual.append(tot)
        annual=np.stack(annual);base=grid[['cell_id']].copy();base['daily_p95']=p95;base['daily_mean']=mean;base['daily_sd']=sd;base['annual_total_mean']=np.nanmean(annual,axis=0);base['annual_total_sd']=np.nanstd(annual,axis=0,ddof=1);base.to_csv(D/f'climatology_{code}_1981_2010.csv',index=False)
        rows=[]
        for y in range(2011,2025):
            a=np.load(D/f'daily_{code}_{y}.npy');valid=np.isfinite(a).all(axis=0);out=grid[['cell_id']].copy();out['climate_year']=y;out['prediction_year']=y+1;out['feature_max_date']=f'{y}-12-31'
            if code=='pr':
                out['annual_total_precipitation']=np.sum(a,axis=0);out['annual_precip_anomaly']=out.annual_total_precipitation-base.annual_total_mean
                run=np.zeros(len(grid),dtype=int);longest=run.copy()
                for daily in a:run=np.where(daily<1,run+1,0);longest=np.maximum(longest,run)
                out['max_consecutive_dry_days']=longest
            else:
                ordinary={'tmmx':'annual_mean_tmax','vpd':'annual_mean_vpd','vs':'annual_mean_wind_speed'}[code];extreme={'tmmx':'hot_days_p95','vpd':'high_vpd_days_p95','vs':'high_wind_days_p95'}[code]
                out[ordinary]=np.mean(a,axis=0);out[extreme]=(a>p95).sum(axis=0);valid &= np.isfinite(p95)
            out.loc[~valid,[c for c in out if c not in ['cell_id','climate_year','prediction_year','feature_max_date']]]=np.nan
            rows.append(out)
        pd.concat(rows).to_csv(done,index=False)
    print('CLIMATE FEATURES COMPLETE',code,flush=True);return str(done)

if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=4) as pool:print(list(pool.map(work,VARS)),flush=True)
    panel=None
    for code in VARS:
        d=pd.read_csv(D/f'climate_{code}.csv.gz');keys=['cell_id','climate_year','prediction_year','feature_max_date'];panel=d if panel is None else panel.merge(d,on=keys,validate='one_to_one')
    panel.to_csv(D/'climate_panel.csv.gz',index=False)
    coverage=panel.groupby('prediction_year').apply(lambda x:x.drop(columns=['cell_id','feature_max_date']).notna().all(axis=1).mean(),include_groups=False);coverage.to_csv(O/'climate_coverage_by_year.csv')
