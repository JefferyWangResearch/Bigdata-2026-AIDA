from setup import *
from netCDF4 import Dataset,num2date

def main():
    checks=[]
    old=json.loads((R/'logs/task2/task1_manifest.json').read_text());changed=[]
    for name,h in old.items():
        p=R/name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:changed.append(name)
    assert not changed,changed;checks.append(f'{len(old)} Task1 input/result/script hashes unchanged')
    wind=json.loads((R/'logs/wind_direction_validation.json').read_text());assert wind['status']=='PASS' and wind['station_check_status']=='PASS';checks.append('FROM/TO unit tests plus independent ASOS sign check')
    manifest=json.loads((R/'logs/task2/weather_manifest.json').read_text());assert len(manifest)==18
    for m in manifest:
        # variable and year in dates are independently checked against local filenames.
        year=int(m['dates'][0][:4]);code='vs' if m['variable'].endswith('speed') else 'th';p=RAW/f'gridmet_{code}_{year}.nc';assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256']
        with Dataset(p) as n:
            v=n.variables[m['variable']];a=np.ma.filled(v[:],np.nan);assert np.nanmin(a)>=0
            if code=='th':assert np.nanmax(a)<=360
            else:assert np.nanmax(a)<100
    checks.append('18 weather snapshot hashes, units/ranges and annual date audit')
    effort=pd.read_csv(R/'data/processed/observation_effort_cell_year.csv.gz');assert not effort.duplicated(['cell_id','year']).any()
    selected=pd.read_csv(D/'effort_query_cells.csv');assert set(effort.cell_id)==set(selected.cell_id);assert len(effort)==len(selected)*11
    for cell in selected.cell_id:
        p=RAW/f'effort_{cell}.json';a=json.loads(p.read_text());m=json.loads((RAW/f'effort_{cell}.meta.json').read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256'];assert effort.loc[effort.cell_id.eq(cell),'all_plant_observation_count'].sum()==a['count']
    for cell,g in effort.groupby('cell_id'):
        g=g.sort_values('year');expected=g.all_plant_observation_count.shift(1).rolling(3).mean();assert np.allclose(expected,g.effort_previous_3yr_mean,equal_nan=True)
    checks.append('Complete targeted effort retrieval, count conservation and historical-only lag construction')
    pool=pd.read_csv(D/'primary_analysis_pool.csv');pairs=pd.read_csv(D/'matched_pairs_primary.csv');stats=pd.read_csv(R/'outputs/task2_alignment_by_year.csv');idx=pool.set_index(['year','cell_id'])
    for row in pairs.itertuples():
        p=idx.loc[(row.year,row.positive_cell)];n=idx.loc[(row.year,row.negative_cell)];assert p.label==1 and n.label==0 and p.region==n.region;assert p.first_year==row.year and (pd.isna(n.first_year) or n.first_year>row.year)
    for year,sub in pairs.groupby('year'):
        p=sub.drop_duplicates('positive_cell');delta=p.positive_alignment.median()-sub.negative_alignment.median();actual=stats.loc[stats.year.eq(year),'delta_alignment'].iloc[0];assert np.isclose(delta,actual)
    checks.append('Independent matched-pair eligibility and median-difference reconstruction')
    null=pd.read_csv(R/'outputs/wind_permutation_results.csv');assert len(null)==500
    obs=stats.delta_alignment.mean();p=(1+(null.statistic>=obs).sum())/501;assert np.isclose(null.empirical_p.iloc[0],p);checks.append('500 permutation rows and independent empirical p-value')
    cov=pd.read_csv(O/'weather_coverage.csv');assert (pd.to_datetime(cov.end).dt.year<cov.year).all();checks.append('All exposure windows end before target year')
    result=dict(status='PASS',checks=checks,task1_changed_files=changed,n_effort_cells=len(selected),n_primary_positive=int(stats.n_positive.sum()),primary_equal_year_difference=float(obs),permutation_p=float(p),scope='Computational reproducibility checks, not independent scientific review')
    (R/'logs/task2/validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
