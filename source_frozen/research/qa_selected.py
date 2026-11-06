import os
os.environ.update(NUMBA_NUM_THREADS='24',NUMBA_THREADING_LAYER='omp',OMP_WAIT_POLICY='PASSIVE',OPENBLAS_NUM_THREADS='1')
import sys,json,time,hashlib,gc
from pathlib import Path
import numpy as np,psutil
H=Path(__file__).resolve().parent;R=H.parents[1];O=R/'outputs/TREE_GEOMETRY_FACTOR_20261104'
sys.path.insert(0,str(H/'candidate'))
from linear_support import calculate
import target_geometry as G
from geometry_checks import used_indices,compare_geometry
flags=dict(workers=12,timeout=1800,memory_mib=44000,dense_cut=-125,minimize_grammar=True,
    split_maps=True,lifetime=True,geometry_mode='target',parallel_geometry=12,numeric_mode='parallel')
counts=dict(coefficients=0,index_values=0,support_values=0,stages=0);cases=[];start=time.perf_counter()
for n in (26,28):
    if psutil.virtual_memory().available<(24<<30):raise MemoryError('RAM reserve for geometry oracle; no disk fallback')
    G.MODE='baseline';saved=[]
    def remember(stage,data,state):
        if data.nbytes+(4<<30)>psutil.virtual_memory().available:raise MemoryError('Oracle RAM budget')
        saved.append((data.copy(),state['byte_offsets'].copy(),state['widths'].copy(),state['meta'],
            state['indices'][:used_indices(state['meta'])].copy(),state['row_ids'],state['column_ids']))
    base=calculate(n,layer_check=remember,**flags);before=dict(counts)
    G.MODE='bitmap'
    def compare(stage,data,state):
        old,off,width,meta,ix,rr,cc=saved[stage-1]
        assert np.array_equal(off,state['byte_offsets']) and np.array_equal(width,state['widths'])
        assert np.array_equal(old,data),(n,stage,'coefficients')
        indexes,support=compare_geometry(meta,state['meta'],ix,state['indices'],rr,state['row_ids'],cc,state['column_ids'])
        counts['index_values']+=int(indexes);counts['support_values']+=int(support);counts['stages']+=1
        counts['coefficients']+=int(np.sum(state['dshape'][:,0]*state['dshape'][:,1]))
    result=calculate(n,layer_check=compare,**flags)
    assert result['Mn']==base['Mn']
    assert result['peak_coefficient_arena_bytes']==base['peak_coefficient_arena_bytes']
    case=dict(n=n,Mn=result['Mn'],status='EXACT_PASS',counts={k:counts[k]-before[k] for k in counts})
    cases.append(case);print(json.dumps(case),flush=True)
    del saved;gc.collect()
receipt=dict(status='PASS',counts=counts,cases=cases,seconds=time.perf_counter()-start,
    oracle_storage='RAM only; no matrix files',
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (H/'candidate').glob('*.py')})
(O/'SELECTED_CORRECTNESS.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(status='PASS',**counts)),flush=True)
