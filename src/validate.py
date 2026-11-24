from common import *
from shapely.geometry import Point,shape
import hashlib

def main():
    d=pd.read_csv(ROOT/'data/processed/occurrences_with_cells.csv');a=pd.read_csv(ROOT/'outputs/observations_by_year.csv');l=pd.read_csv(ROOT/'data/processed/next_year_labels.csv.gz');g=pd.read_csv(ROOT/'data/processed/grid_4km.csv');audit=pd.read_csv(ROOT/'outputs/data_cleaning_summary.csv');first=d.groupby('cell_id').year.min()
    assert (audit.n_before-audit.n_removed==audit.n_after).all()
    assert audit.n_after.iloc[-1]==len(d)
    assert a.total_observations.sum()==len(d)
    assert a.newly_observed_cells.sum()==d.cell_id.nunique()
    assert not l.duplicated(['cell_id','prediction_year']).any()
    assert not l.first_observation_year.lt(l.prediction_year).any()
    assert (l.new_occurrence_label==l.first_observation_year.eq(l.prediction_year).astype(int)).all()
    assert l.new_occurrence_label.sum()==first.between(2000,2025).sum()
    assert (l.candidate_25km<=l.candidate_50km).all() and (l.candidate_50km<=l.candidate_100km).all()
    # Independently check geometry for a deterministic sample.
    feats=json.loads((ROOT/'data/processed/grid_4km.geojson').read_text())['features'];geom={f['properties']['cell_id']:shape(f['geometry']) for f in feats}
    for row in d.sample(min(100,len(d)),random_state=SEED).itertuples():assert geom[row.cell_id].buffer(1e-9).covers(Point(row.decimalLongitude,row.decimalLatitude))
    for y in range(2000,2026):
        sub=l.loc[l.prediction_year.eq(y)];assert len(sub)==len(g)-first.lt(y).sum()
    manifest=json.loads((ROOT/'logs/download_manifest.json').read_text())
    for m in manifest:assert hashlib.sha256((ROOT/'data/raw'/m['file']).read_bytes()).hexdigest()==m['sha256']
    result=dict(status='PASS',clean_records=len(d),observed_cells=d.cell_id.nunique(),grid_cells=len(g),candidate_rows=len(l),checks=['cleaning arithmetic','annual reconciliation','target eligibility','first-year labels','nested neighborhoods','independent grid geometry sample','year-specific candidate counts','raw-file checksums'],scope='Computational audit only; not independent scientific review')
    (ROOT/'logs/validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
