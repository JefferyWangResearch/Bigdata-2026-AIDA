"""Reproduce Task2 from cached Task1; fail fast rather than invent missing outputs."""
from pathlib import Path
import subprocess,sys,os,datetime,json
R=Path(__file__).resolve().parents[2];ledger=[]
assert (R/'data/processed/task2/candidate_pool.csv').exists(), 'Initialize Task2 fixed pools first; never rerun Task1.'
for script in ['weather','station_check','effort','test_exposure_math','exposure','analyze','figures','validate','report']:
    started=datetime.datetime.now(datetime.timezone.utc).isoformat();env=os.environ.copy();env['MPLCONFIGDIR']=str(R/'logs/task2/matplotlib');env['PYTHONPYCACHEPREFIX']=str(R/'logs/task2/pycache');env['OPENBLAS_NUM_THREADS']='1'
    with (R/'logs/task2'/f'run_{script}.log').open('w') as f:
        p=subprocess.run([sys.executable,str(R/'src/task2'/f'{script}.py')],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
    ledger.append(dict(script=script,started_utc=started,returncode=p.returncode));(R/'logs/task2/run_ledger.json').write_text(json.dumps(ledger,indent=2));print(script,p.returncode,flush=True)
    if p.returncode:sys.exit(p.returncode)
