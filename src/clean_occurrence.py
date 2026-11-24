"""Deterministic occurrence audit; uncertainty alternatives remain available."""
from common import *
from shapely.geometry import Point

def main():
    rows=json.loads((ROOT/'data/raw/occurrences.json').read_text());d=pd.DataFrame(rows)
    for c in ['decimalLatitude','decimalLongitude','year','coordinateUncertaintyInMeters']:
        d[c]=pd.to_numeric(d.get(c,pd.Series(index=d.index,dtype=float)),errors='coerce')
    fields=['key','occurrenceID','scientificName','species','decimalLatitude','decimalLongitude','eventDate','year','month','day','coordinateUncertaintyInMeters','institutionCode','datasetKey','datasetName','basisOfRecord','issues','license','references','stateProvince','locality','verbatimLocality','geodeticDatum','modified','lastCrawled','firstIdentified','identificationVerificationStatus','occurrenceStatus','establishmentMeans','http://unknown.org/captive_cultivated']
    for c in fields:
        if c not in d:d[c]=None
    d=d[fields].copy();d['issues']=d.issues.apply(lambda x:'|'.join(x) if isinstance(x,list) else str(x or ''))
    audit=[];excluded=[]
    def keep(mask,step):
        nonlocal d
        mask=pd.Series(mask,index=d.index).fillna(False);n=len(d)
        e=d.loc[~mask].copy();e['exclusion_reason']=step;excluded.append(e)
        d=d.loc[mask].copy();audit.append(dict(step=step,n_before=n,n_removed=n-len(d),n_after=len(d)))
    keep(d.occurrenceStatus.ne('ABSENT'),'not_absent')
    keep(d['http://unknown.org/captive_cultivated'].fillna('').ne('captive'),'not_explicitly_captive')
    keep(d.decimalLatitude.between(-90,90)&d.decimalLongitude.between(-180,180),'valid_coordinates')
    keep(d.year.notna()&d.year.between(1,2026)&d.year.eq(d.year.round()),'valid_year')
    # Exclude unambiguously future dates and dates spanning multiple years.
    s=d.eventDate.fillna('').astype(str);start=s.str.split('/').str[0].str[:10]
    end=s.str.split('/').str[-1].str[:4]
    keep((~s.str.contains('/')|start.str[:4].eq(end))&((start<='2026-10-04')|start.eq('')),'unambiguous_nonfuture_date')
    keep(~d.issues.str.contains('ZERO_COORDINATE|COORDINATE_INVALID|COORDINATE_OUT_OF_RANGE|PRESUMED_NEGATED_LATITUDE|PRESUMED_NEGATED_LONGITUDE'),'coordinate_issue_filter')
    g,proj=boundary()
    keep([g.covers(Point(lon,lat)) for lon,lat in zip(d.decimalLongitude,d.decimalLatitude)],'within_california_boundary')
    centroid=INV.transform(proj.centroid.x,proj.centroid.y)
    # Only exact detectable state centroid and explicit centroid issue; no invented institution gazetteer.
    keep(~((d.decimalLongitude-centroid[0]).abs().lt(1e-5)&(d.decimalLatitude-centroid[1]).abs().lt(1e-5))&~d.issues.str.contains('COORDINATE_REPRESENTS_COUNTRY_CENTROID'),'detectable_centroid_filter')
    keep(~(d.occurrenceID.notna()&d.occurrenceID.ne('')&d.duplicated('occurrenceID')),'deduplicate_occurrence_id')
    d['date_key']=d.eventDate.fillna(d.year.astype(int).astype(str))
    keep(~d.duplicated(['species','date_key','decimalLatitude','decimalLongitude']),'deduplicate_species_date_coordinates')
    d['year']=d.year.astype(int)
    d['uncertainty_missing']=d.coordinateUncertaintyInMeters.isna()
    d['uncertainty_4km_ok']=d.coordinateUncertaintyInMeters.isna()|d.coordinateUncertaintyInMeters.between(0,4000)
    d['uncertainty_1km_known']=d.coordinateUncertaintyInMeters.between(0,1000)
    d['institution_locality_flag']=d.locality.fillna('').str.contains('herbarium|museum|botanical garden',case=False,regex=True)
    d.to_csv(ROOT/'data/interim/clean_unrestricted_precision.csv',index=False)
    keep(d.uncertainty_4km_ok,'primary_precision_unknown_or_le_4km')
    d.to_csv(ROOT/'data/processed/occurrences_clean.csv',index=False)
    pd.concat(excluded).to_csv(ROOT/'outputs/excluded_records.csv',index=False)
    pd.DataFrame(audit).to_csv(ROOT/'outputs/data_cleaning_summary.csv',index=False)
    d.groupby(['datasetKey','datasetName'],dropna=False).agg(records=('key','size'),first_year=('year','min'),last_year=('year','max')).reset_index().to_csv(ROOT/'outputs/source_summary.csv',index=False)
    d.groupby(['year','datasetName'],dropna=False).size().rename('n').reset_index().to_csv(ROOT/'outputs/source_by_year.csv',index=False)
    d.issues.str.split('|').explode().value_counts().rename_axis('issue').reset_index(name='n').to_csv(ROOT/'outputs/quality_issues.csv',index=False)
    print(pd.DataFrame(audit).to_string(index=False))
if __name__=='__main__':main()
