from pathlib import Path
import json, zipfile
import numpy as np
import pandas as pd
import shapefile
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer
ROOT=Path(__file__).resolve().parents[1]
SEED=20261004
np.random.seed(SEED)
FWD=Transformer.from_crs(4326,3310,always_xy=True)
INV=Transformer.from_crs(3310,4326,always_xy=True)
def boundary():
    p=ROOT/'data/interim/boundary';p.mkdir(exist_ok=True)
    with zipfile.ZipFile(ROOT/'data/raw/cb_2023_us_state_500k.zip') as z:z.extractall(p)
    r=shapefile.Reader(str(next(p.glob('*.shp'))))
    for sr in r.iterShapeRecords():
        if sr.record.as_dict()['STUSPS']=='CA':
            g=shape(sr.shape.__geo_interface__);return g,transform(FWD.transform,g)
def assign(df,size=4000):
    d=df.copy();x,y=FWD.transform(d.decimalLongitude.values,d.decimalLatitude.values)
    d['x']=x;d['y']=y;d['gx']=np.floor(x/size).astype(int);d['gy']=np.floor(y/size).astype(int)
    d['cell_id']=d.gx.astype(str)+'_'+d.gy.astype(str);return d
