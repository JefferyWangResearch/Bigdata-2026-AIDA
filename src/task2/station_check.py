"""Independent FROM/TO sign check; not a local-wind accuracy validation."""
from setup import *
import requests,datetime
from netCDF4 import Dataset,num2date

def main():
    p=RAW/'iem_asos_july2017.csv'
    if not p.exists():
        params=dict(station=['SFO','SAC','LAX'],data=['drct','sknt'],year1=2017,month1=7,day1=1,year2=2017,month2=8,day2=1,tz='Etc/UTC',format='onlycomma',latlon='yes',elev='no',missing='M',trace='T',direct='no',report_type=3)
        r=requests.get('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py',params=params,timeout=120);r.raise_for_status();p.write_text(r.text)
        p.with_suffix('.meta.json').write_text(json.dumps(dict(url=r.url,retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
    d=pd.read_csv(p,na_values=['M']);d['valid']=pd.to_datetime(d.valid);d['date']=(d.valid-pd.Timedelta(hours=7)).dt.floor('D');d=d.loc[d.drct.notna()&d.sknt.gt(0)].copy();d['sin']=np.sin(np.deg2rad(d.drct))*d.sknt;d['cos']=np.cos(np.deg2rad(d.drct))*d.sknt
    out=[]
    with Dataset(RAW/'gridmet_th_2017.nc') as n:
        t=n.variables['day'];dates=pd.to_datetime([str(x)[:10] for x in num2date(t[:],t.units)]);lat=n.variables['lat'][:];lon=n.variables['lon'][:]
        for station,sub in d.groupby('station'):
            iy=np.abs(lat-sub.lat.iloc[0]).argmin();ix=np.abs(lon-sub.lon.iloc[0]).argmin();vals=np.ma.filled(n.variables['daily_mean_wind_direction'][:,iy,ix],np.nan)
            daily=sub.groupby('date')[['sin','cos']].mean();daily['station_from']=np.rad2deg(np.arctan2(daily['sin'],daily['cos']))%360;daily['gridmet_reported']=pd.Series(vals,index=dates).reindex(daily.index);daily=daily.dropna()
            agreement=np.cos(np.deg2rad(daily.station_from-daily.gridmet_reported));out.append(dict(station=station,n_days=len(daily),mean_cosine_if_gridmet_from=float(agreement.mean()),mean_cosine_if_gridmet_to=float(-agreement.mean())))
    result=pd.DataFrame(out);result.to_csv(O/'station_direction_crosscheck.csv',index=False)
    assert len(result)==3 and result.mean_cosine_if_gridmet_from.gt(0).all(),result
    path=R/'logs/wind_direction_validation.json';v=json.loads(path.read_text());v['empirical_station_check']=out;v['provider_metadata']='Degrees Clockwise from north; description Daily mean wind direction (10m). Metadata does not explicitly say FROM; independently crosschecked sign against ASOS FROM-direction observations.';v['station_check_status']='PASS';path.write_text(json.dumps(v,indent=2));print(result.to_string(index=False))
if __name__=='__main__':main()
