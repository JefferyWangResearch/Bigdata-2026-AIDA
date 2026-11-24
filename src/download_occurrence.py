"""Cache Task 1 public inputs and exact query provenance; reruns use the cache."""
from pathlib import Path
import json, datetime, hashlib, time
import requests
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'data/raw'; RAW.mkdir(parents=True,exist_ok=True)
S=requests.Session(); S.headers['User-Agent']='Salsola-feasibility-research/1.0'
manifest=[]
def fetch(name,url,params=None,allow_error=False):
    p=RAW/name
    if not p.exists():
        r=S.get(url,params=params,timeout=120)
        if not allow_error:r.raise_for_status()
        p.write_bytes(r.content)
        (RAW/(name+'.meta.json')).write_text(json.dumps(dict(url=r.url,status=r.status_code,retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
        time.sleep(.2)
    meta=json.loads((RAW/(name+'.meta.json')).read_text());meta.update(file=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest());manifest.append(meta)
    return p

def main():
    fetch('calflora_access.json','https://api.calflora.org/observations',dict(taxon='Salsola tragus',maxResults=1,georef='a'),True)
    tax=json.loads(fetch('gbif_taxon_match.json','https://api.gbif.org/v1/species/match',dict(name='Salsola tragus',strict='true')).read_text())
    assert tax['matchType']=='EXACT' and tax['rank']=='SPECIES'
    base=dict(taxonKey=tax['usageKey'],country='US')
    # Union state-labelled records (including missing coords) and georeferenced CA bounding box.
    queries={'state':dict(stateProvince='California'),'bbox':dict(decimalLatitude='32,42.1',decimalLongitude='-124.6,-114',hasCoordinate='true')}
    allrows=[]; counts={}
    for tag,extra in queries.items():
        offset=0; rows=[]
        while True:
            d=json.loads(fetch(f'gbif_{tag}_{offset:05d}.json','https://api.gbif.org/v1/occurrence/search',dict(base,**extra,limit=300,offset=offset)).read_text())
            counts[tag]=d['count'];rows+=d['results']
            print(tag,offset,d['count'],flush=True)
            if d['endOfRecords']:break
            offset+=300
            if offset>=9900:raise RuntimeError('Use authenticated GBIF download for larger query; do not truncate.')
        assert len(rows)==counts[tag],(tag,len(rows),counts[tag])
        allrows+=rows
    unique={r['key']:r for r in allrows}
    (RAW/'occurrences.json').write_text(json.dumps(list(unique.values())))
    (RAW/'query_summary.json').write_text(json.dumps(dict(query_counts=counts,union_unique_gbif_ids=len(unique),query_overlap=len(allrows)-len(unique)),indent=2))
    fetch('cb_2023_us_state_500k.zip','https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_500k.zip')
    (ROOT/'logs/download_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('UNION',len(unique))
if __name__=='__main__':main()
