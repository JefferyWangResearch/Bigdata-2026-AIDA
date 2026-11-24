from setup import *
from pyproj import Geod
GEO=Geod(ellps='WGS84')

def window_data(year,window):
    years=[year-1] if window=='annual' else [year-2,year-1];ss=[];tt=[];dd=[]
    for y in years:
        a=np.load(D/f'wind_sources_{y}.npz');ss.append(a['speed']);tt.append(a['direction_from']);dd.extend(a['dates'])
    dates=pd.to_datetime(dd);speed=np.concatenate(ss);direction=np.concatenate(tt)
    if window!='annual':
        start=f'{year-2}-'+('09-01' if window=='Sep-Jan' else '10-01');mask=(dates>=start)&(dates<=f'{year-1}-01-31');speed=speed[mask];direction=direction[mask];dates=dates[mask]
    assert dates.max()<pd.Timestamp(f'{year}-01-01')
    return a['cell_ids'],speed,direction,dates

def main():
    assert json.loads((R/'logs/wind_direction_validation.json').read_text())['status']=='PASS'
    pool=pd.read_csv(D/'candidate_pool.csv');sources=pd.read_csv(D/'historical_sources.csv');allout=[];coverage=[]
    for quality in ['primary','strict']:
      for year in range(2018,2026):
        c=pool.loc[pool.quality.eq(quality)&pool.year.eq(year)].reset_index(drop=True);s=sources.loc[sources.quality.eq(quality)&sources.year.eq(year)].reset_index(drop=True)
        assert s.first_year.lt(year).all()
        dist=np.sqrt((c.x.values[:,None]-s.x.values)**2+(c.y.values[:,None]-s.y.values)**2)/1000
        az,_,_=GEO.inv(np.broadcast_to(s.center_lon.values,dist.shape),np.broadcast_to(s.center_lat.values,dist.shape),np.broadcast_to(c.center_lon.values[:,None],dist.shape),np.broadcast_to(c.center_lat.values[:,None],dist.shape));bearing=np.deg2rad(az)
        for window in ['Sep-Jan','Oct-Jan','annual']:
          ids,speed,fr,dates=window_data(year,window);lookup={v:k for k,v in enumerate(ids)};ix=[lookup[v] for v in s.cell_id];speed=speed[:,ix];fr=fr[:,ix];ok=np.isfinite(speed)&np.isfinite(fr);valid=ok.mean(axis=0)>=.95
          coverage.append(dict(quality=quality,year=year,window=window,n_sources=len(s),n_valid_sources=int(valid.sum()),minimum_daily_coverage=float(ok.mean(axis=0).min()),start=str(dates.min().date()),end=str(dates.max().date())))
          theta=np.deg2rad((fr+180)%360);cs=np.nanmean(np.cos(theta),axis=0);sn=np.nanmean(np.sin(theta),axis=0)
          align=np.cos(bearing)*cs+np.sin(bearing)*sn
          perp=np.cos(bearing)*sn-np.sin(bearing)*cs
          pos=np.zeros(dist.shape);spweighted=np.zeros(dist.shape);highweighted=np.zeros(dist.shape);pressure=np.zeros(dist.shape);reversepressure=np.zeros(dist.shape)
          for k in np.where(valid)[0]:
            targets=np.where(dist[:,k]<=100)[0]
            if not len(targets):continue
            a=np.cos(theta[:,k,None]-bearing[targets,k]);v=speed[:,k,None];hi=v>=np.nanquantile(v,.9)
            pos[targets,k]=np.nanmean(np.maximum(a,0),axis=0)
            spweighted[targets,k]=np.nansum(v*a,axis=0)/np.nansum(v)
            highweighted[targets,k]=np.nansum(v*a*hi,axis=0)/np.nansum(v*hi)
            pressure[targets,k]=np.nanmean(v*np.maximum(a,0),axis=0)
            reversepressure[targets,k]=np.nanmean(v*np.maximum(-a,0),axis=0)
          configs=[(r,25) for r in [25,50,100]]+([(50,10),(50,50)] if window=='Sep-Jan' else [])
          for radius,L in configs:
            w=np.exp(-dist/L)*(dist<=radius);total=w.sum(axis=1);w=w*valid;den=w.sum(axis=1);cov=np.divide(den,total,out=np.zeros_like(den),where=total>0)
            eligible=(c.distance.values<=radius)&(cov>=.95)&(den>0);out=c.loc[eligible].copy();out['radius']=radius;out['L']=L;out['window']=window;out['source_weather_coverage']=cov[eligible]
            for name,arr in [('mean_alignment',align),('perpendicular_alignment',perp),('positive_alignment',pos),('wind_speed_weighted_alignment',spweighted),('high_wind_weighted_alignment',highweighted)]:
                val=np.nansum(w*arr,axis=1)/np.where(den>0,den,np.nan);out[name]=val[eligible]
            out['WSP']=np.nansum(w*pressure,axis=1)[eligible];out['reversed_WSP']=np.nansum(w*reversepressure,axis=1)[eligible];out['reversed_alignment']=-out.mean_alignment
            allout.append(out)
        print('exposure',quality,year,flush=True)
    pd.concat(allout).to_csv(D/'exposures.csv.gz',index=False);pd.DataFrame(coverage).to_csv(O/'weather_coverage.csv',index=False)
if __name__=='__main__':main()
