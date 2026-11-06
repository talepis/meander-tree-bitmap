"""Conservative fresh resource gate. Does not start MEANDER or contact a server."""
import argparse,hashlib,json,sys
from pathlib import Path
from server_resources import Resources,GiB
from verify_bundle import main as verify
H=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('M37_MEMORY_PREFLIGHT.json'));a=p.parse_args()
    verify()
    ref=json.loads((H/'memory_evidence/REFERENCE_M37_COMPACT_MEMORY_PREFLIGHT.json').read_text())
    test=json.loads((H/'SELF_TEST_RESULT.json').read_text())
    digest=hashlib.sha256((H/'SHA256_MANIFEST.json').read_bytes()).hexdigest()
    assert test['status']=='PASS' and test['bundle_manifest_sha256']==digest
    import psutil
    active=[]
    for proc in psutil.process_iter(['pid','cmdline','status']):
        try:
            args=proc.info['cmdline'] or []
            if proc.info['status']==psutil.STATUS_ZOMBIE:continue
            for i,arg in enumerate(args[:-1]):
                if Path(arg).name=='server_meander.py' and args[i+1].isdigit() and int(args[i+1])>=30:
                    active.append(proc.info['pid']);break
        except (psutil.AccessDenied,psutil.NoSuchProcess):pass
    resources=Resources().snapshot();budget=max(0,resources['effective_available_bytes']-19*GiB)
    predicted=int(ref['predicted_peak_bytes']);decision='RUN' if not active and predicted+8*GiB<=budget else 'STOP'
    out=dict(n=37,decision=decision,predicted_peak_bytes=predicted,
        source_manifest_sha256=digest,guarded_budget_bytes=budget,resources=resources,active_MEANDER_worker_PIDs=active,
        prediction_is_proved_upper_bound=False,
        basis='Previous M36 full-run based conservative forecast retained without subtracting unmeasured M37 savings. Same coefficient arithmetic and capacities, fewer live workspace pages. Fresh 16 GiB reserve plus 3 GiB gate allowance and 8 GiB extra preflight margin. No matrix spill.')
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
    if decision!='RUN':raise SystemExit(75)
if __name__=='__main__':main()
