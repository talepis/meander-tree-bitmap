import os
os.environ.update(NUMBA_NUM_THREADS='24',NUMBA_THREADING_LAYER='omp',OMP_WAIT_POLICY='PASSIVE',OPENBLAS_NUM_THREADS='1')
import sys,json,time,hashlib,random,gc
from pathlib import Path
import numpy as np
from numba import njit
H=Path(__file__).resolve().parent;R=H.parents[1];O=R/'outputs/TREE_GEOMETRY_FACTOR_20261104'
sys.path.insert(0,str(H/'candidate'))
from linear_support import calculate
import target_geometry as G
import bitmap_geometry as B

@njit(cache=True)
def used_indices(meta):
    used=0
    for m in meta:
        if m[6]:
            if not m[11]&1:used=max(used,m[9]+m[3]-m[2])
            if not m[11]&2:used=max(used,m[10]+m[5]-m[4])
    return used

@njit(cache=True)
def compare_geometry(oldmeta,newmeta,oldix,newix,oldR,newR,oldC,newC):
    if oldmeta.shape!=newmeta.shape:raise AssertionError('Metadata shape')
    count=0;support_count=0
    for i in range(len(oldmeta)):
        a=oldmeta[i];b=newmeta[i]
        for k in range(14):
            if k not in (9,10) and a[k]!=b[k]:raise AssertionError('Metadata semantics')
        if a[6]:
            for axis in range(2):
                if a[11]&(1<<axis):
                    if a[9+axis]!=b[9+axis]:raise AssertionError('Immutable map offset')
                else:
                    length=a[3]-a[2] if axis==0 else a[5]-a[4]
                    for k in range(length):
                        if oldix[a[9+axis]+k]!=newix[b[9+axis]+k]:raise AssertionError('Translated index')
                    count+=length
    for d in range(len(oldR)):
        if not np.array_equal(oldR[d],newR[d]) or not np.array_equal(oldC[d],newC[d]):raise AssertionError('Axis support')
        support_count+=len(oldR[d])+len(oldC[d])
    return count,support_count

refs={int(x['n']):str(x['answer_decimal']) for x in json.loads((R/'outputs/MN_TABLE/TABLE_DATA.json').read_text(encoding='utf-8'))['selected']}
flags=dict(workers=12,timeout=1800,memory_mib=44000,dense_cut=-125,minimize_grammar=True,split_maps=True,lifetime=True,geometry_mode='target',parallel_geometry=12,numeric_mode='parallel')
counts=dict(coefficients=0,index_values=0,support_values=0,stages=0);cases=[];start=time.perf_counter()
for n in [*range(1,13),20,24]:
    G.MODE='baseline';saved=[]
    def remember(stage,data,state):
        saved.append((data.copy(),state['byte_offsets'].copy(),state['widths'].copy(),state['meta'],state['indices'][:used_indices(state['meta'])].copy(),state['row_ids'],state['column_ids']))
    base=calculate(n,layer_check=remember,**flags)
    for mode in (['bitmap','bitmap_nocache'] if n<=12 else ['bitmap']):
        G.MODE=mode
        def compare(stage,data,state):
            old,off,width,meta,ix,rr,cc=saved[stage-1]
            assert np.array_equal(off,state['byte_offsets']) and np.array_equal(width,state['widths'])
            assert np.array_equal(old,data),(n,stage,'coefficients')
            c,s=compare_geometry(meta,state['meta'],ix,state['indices'],rr,state['row_ids'],cc,state['column_ids'])
            counts['index_values']+=int(c);counts['support_values']+=int(s);counts['stages']+=1
            counts['coefficients']+=int(np.sum(state['dshape'][:,0]*state['dshape'][:,1]))
        new=calculate(n,layer_check=compare,**flags)
        assert new['Mn']==base['Mn']==refs[n]
        cases.append(dict(n=n,mode=mode,Mn=new['Mn'],status='EXACT_PASS'))
    print(json.dumps(dict(n=n,status='EXACT_PASS')),flush=True)
    del saved;gc.collect()
    (O/'CORRECTNESS.json').write_text(json.dumps(dict(status='RUNNING',counts=counts,cases=cases),indent=2)+'\n')
receipt=dict(status='PASS',counts=counts,cases=cases,seconds=time.perf_counter()-start,
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (H/'candidate').glob('*.py')})
(O/'CORRECTNESS.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(status='PASS',**counts)),flush=True)
