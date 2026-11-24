"""Run the bounded occurrence feasibility audit; existing downloads are reused."""
from pathlib import Path
import subprocess,sys,datetime,json
root=Path(__file__).resolve().parents[1]; ledger=[]
for script in ['download_occurrence','clean_occurrence','build_grid','build_labels','make_figures','write_report','validate']:
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (root/'logs'/f'{script}.log').open('w') as f:
        result=subprocess.run([sys.executable,str(root/'src'/f'{script}.py')],cwd=root,stdout=f,stderr=subprocess.STDOUT)
    ledger.append(dict(script=script,started_utc=started,returncode=result.returncode))
    (root/'logs/run_ledger.json').write_text(json.dumps(ledger,indent=2))
    print(script,result.returncode,flush=True)
    if result.returncode:sys.exit(result.returncode)
