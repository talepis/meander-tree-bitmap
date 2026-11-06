"""One isolated complete computation at a time; warm-up is reported separately."""
import os,sys
os.environ.update(NUMBA_NUM_THREADS='24',NUMBA_THREADING_LAYER='omp',OMP_WAIT_POLICY='PASSIVE',KMP_BLOCKTIME='0',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,time,threading,gc,ctypes,hashlib
from pathlib import Path
import psutil
H=Path(__file__).resolve().parent;R=H.parents[1];O=R/'outputs/TREE_GEOMETRY_FACTOR_20261104'
if (H/'EXPERIMENT_PAUSE').exists():
 print('Paused for correction; no calculation started.',flush=True);raise SystemExit(77)
mode=sys.argv[1];requested_mode=mode;n=int(sys.argv[2]);reps=int(sys.argv[3]) if len(sys.argv)>3 else 3;tag=sys.argv[4] if len(sys.argv)>4 else mode
engine_root=H/'candidate'
sys.path.insert(0,str(engine_root))
from linear_support import calculate
flags=dict(minimize_grammar=True,split_maps=True,lifetime=True,geometry_mode='target',parallel_geometry=12,numeric_mode='parallel')
import target_geometry
target_geometry.MODE=mode
source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in engine_root.glob('*.py')};proc=psutil.Process();proc.cpu_affinity(list(range(24)))
sys.path.insert(0,str(R/"work/tree_all5_20261103"))
from process_qos import high_qos
from thread_placement import pin_master
qos=high_qos();placement=pin_master()
def power():
    class P(ctypes.Structure):_fields_=[('AC',ctypes.c_ubyte),('flags',ctypes.c_ubyte),('pct',ctypes.c_ubyte),('reserved',ctypes.c_ubyte),('life',ctypes.c_uint32),('full',ctypes.c_uint32)]
    p=P();ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(p));return dict(AC=int(p.AC),battery_percent=int(p.pct),energy_saver=int(p.reserved))
target=O/('M%d_%s.json'%(n,tag))
if target.exists():raise FileExistsError(target)
start=time.perf_counter();warm=calculate(12,12,timeout=1800,dense_cut=-125,**flags);warm_seconds=time.perf_counter()-start
assert warm['Mn']=='678390116'
refs=json.loads((R/'outputs/MN_TABLE/TABLE_DATA.json').read_text(encoding='utf-8'))['selected'];expected=next(str(x['answer_decimal']) for x in refs if int(x['n'])==n)
halt=threading.Event();peak=[0];rows=[]
def sample():
    while not halt.wait(.005):peak[0]=max(peak[0],proc.memory_info().rss)
thread=threading.Thread(target=sample,daemon=True);thread.start()
try:
    for rep in range(reps):
        gc.collect();peak[0]=proc.memory_info().rss;before=power();start=time.perf_counter();cpu_start=sum(proc.cpu_times()[:2])
        def progress(row):
            (O/'LIVE.json').write_text(json.dumps(dict(status='RUNNING',mode=mode,n=n,rep=rep,RSS_sampled_peak_bytes=peak[0],process_CPU_seconds=sum(proc.cpu_times()[:2])-cpu_start,**row),indent=2),encoding='utf-8')
        d=calculate(n,12,memory_mib=(psutil.virtual_memory().available-(6<<30))/(1<<20),timeout=300,
            progress=progress,dense_cut=-125,index_prealloc_bytes=2701940180 if n==30 else 0,**flags)
        elapsed=time.perf_counter()-start
        assert d['status']=='PASS' and d['Mn']==expected
        row=dict(status='PASS',n=n,mode=mode,requested_mode=requested_mode,rep=rep,kind='first_after_small_warmup' if rep==0 else 'warm',
            wall_seconds=elapsed,process_CPU_seconds=sum(proc.cpu_times()[:2])-cpu_start,RSS_sampled_peak_bytes=peak[0],Mn=d['Mn'],workers=12,
            power_before=before,power_after=power(),warmup_seconds=warm_seconds,source_hashes=source_hashes,thread_placement=placement,process_qos=qos,details=d)
        rows.append(row);target.write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(json.dumps({k:row[k] for k in ['status','n','mode','rep','Mn','wall_seconds','RSS_sampled_peak_bytes','power_before']}),flush=True)
    (O/'LIVE.json').write_text(json.dumps(dict(status='COMPLETE',mode=mode,n=n,Mn=d['Mn']),indent=2),encoding='utf-8')
finally:halt.set();thread.join()
