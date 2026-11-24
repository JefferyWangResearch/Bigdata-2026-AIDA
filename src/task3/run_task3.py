from setup import *
import subprocess,time,os

def main():
    ledger=[];env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR=str(LOG/'matplotlib'))
    for script in ['spatial.py','effort.py','climate.py','model.py','radius_sensitivity.py','dictionary.py','figures.py','report.py','validate.py']:
        start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run([sys.executable,str(R/'src/task3'/script)],cwd=R,env=env);ledger.append(dict(script=script,start_utc=start,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=p.returncode));(LOG/'run_ledger.json').write_text(json.dumps(ledger,indent=2))
        if p.returncode:raise SystemExit(p.returncode)
if __name__=='__main__':main()
