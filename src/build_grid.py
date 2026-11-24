from common import *
from shapely.geometry import box, mapping

def main():
    g,p=boundary();size=4000;minx,miny,maxx,maxy=p.bounds;rows=[];features=[]
    for gx in range(int(np.floor(minx/size)),int(np.floor(maxx/size))+1):
        for gy in range(int(np.floor(miny/size)),int(np.floor(maxy/size))+1):
            b=box(gx*size,gy*size,(gx+1)*size,(gy+1)*size)
            if not p.intersects(b):continue
            lon,lat=INV.transform(b.centroid.x,b.centroid.y)
            row=dict(cell_id=f'{gx}_{gy}',gx=gx,gy=gy,x=b.centroid.x,y=b.centroid.y,center_lon=lon,center_lat=lat,california_fraction=p.intersection(b).area/b.area)
            rows.append(row);features.append(dict(type='Feature',properties=row,geometry=mapping(transform(INV.transform,b))))
    pd.DataFrame(rows).to_csv(ROOT/'data/processed/grid_4km.csv',index=False)
    (ROOT/'data/processed/grid_4km.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=features)))
    d=assign(pd.read_csv(ROOT/'data/processed/occurrences_clean.csv'))
    assert set(d.cell_id)<=set(r['cell_id'] for r in rows)
    d.to_csv(ROOT/'data/processed/occurrences_with_cells.csv',index=False)
    print('Grid cells',len(rows),'observed cells',d.cell_id.nunique())
if __name__=='__main__':main()
