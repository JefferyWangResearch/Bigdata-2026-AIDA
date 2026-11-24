"""Optional bounded download accelerator; identical fetch/date checks and cache paths.
Run only while baseline downloads are still before2000 so file requests do not overlap.
Does not construct features or inspect labels/results.
"""
from climate import *
from concurrent.futures import ProcessPoolExecutor

def recent(code):
    s=requests.Session()
    for year in range(2024,2010,-1):
        p=fetch(code,year,s);print('PREFETCH',code,year,p.stat().st_size,flush=True)
if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(recent,['vpd','vs']))
