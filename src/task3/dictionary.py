from setup import *
from model import M0,ORD,EXT
spec={
'distance_nearest_source_km':('Minimum center-to-center EPSG3310 distance to any previously observed occupied grid cell','km'),
'source_count_25km':('Number of distinct previously occupied cells within25km','cells'),
'source_count_50km':('Number of distinct previously occupied cells within50km','cells'),
'source_pressure_L25':('Sum exp(-distance_km/25) over ALL previously occupied cells; not raw record counts','dimensionless'),
'log_previous_3yr_effort':('log1p(mean annual vascular-plant occurrence count during t-3,t-2,t-1); previous-year sensitivity substitutes t-1','log count'),
'annual_mean_tmax':('Mean daily maximum temperature during calendar t-1','degC'),
'annual_total_precipitation':('Sum daily precipitation during calendar t-1','mm'),
'annual_mean_vpd':('Mean daily vapor pressure deficit during calendar t-1','kPa'),
'annual_mean_wind_speed':('Mean daily nondirectional wind speed during calendar t-1','m/s'),
'hot_days_p95':('Days Tmax exceeds fixed cell-specific1981–2010 pooled daily95thpercentile','days'),
'high_vpd_days_p95':('Days VPD exceeds fixed cell-specific1981–2010 pooled daily95thpercentile','days'),
'max_consecutive_dry_days':('Longest within-calendar-year run of daily precipitation<1mm; resets Jan1','days'),
'annual_precip_anomaly':('Calendar t-1 precipitation total minus1981–2010 mean annual total','mm'),
'high_wind_days_p95':('Days nondirectional wind speed exceeds fixed cell-specific1981–2010 pooled daily95thpercentile','days')}
rows=[]
for f in M0+ORD+EXT:
 rows.append(dict(feature=f,feature_group='M0' if f in M0 else 'M1' if f in ORD else 'M2',definition=spec[f][0],time_window='all records event-year<t' if f in M0[:4] else 't-3 through t-1' if f==M0[-1] else 'calendar t-1; climatology1981–2010',units=spec[f][1],source='GBIF Salsola occupied4km cells' if f in M0[:4] else 'GBIF vascular-plant density tiles (uniform approximate effort proxy)' if f==M0[-1] else 'gridMET nearest native cell',leakage_rule='feature_max_date <= Dec31(t-1); baseline ends2010; imputation/scaling fitted only on past training rows'))
pd.DataFrame(rows).to_csv(O/'feature_dictionary.csv',index=False)
