"""Force independent-target parallel writes with a Python-integer oracle."""
import numpy as np
from numba import types
from numba.typed import List

def run_checks():
    import numeric_kernels as K
    G=(__import__('math').comb(72,36)//37)**2;mask=(1<<64)-1;checks=0
    def pack(values):
        a=np.zeros(len(values)*16,np.uint8);z=a.view(np.uint64)
        for k,v in enumerate(values):z[2*k]=v&mask;z[2*k+1]=v>>64
        return a
    for full_bound in (False,True):
      x=G//(17*16)
      av=[G,G-1,1<<126,(1<<64)-1] if full_bound else [x,2*x,3*x,4*x]
      bv=[x]*4;weight=1 if full_bound else 17;nd=4
      for overwrite in (False,True):
        blocks=List.empty_list(types.Array(types.uint8,1,'C'));blocks.append(pack(av));blocks.append(pack(bv))
        original=[a.copy() for a in blocks]
        ops=[];offsets=[0];expected=[]
        for d in range(nd):
            first=np.zeros(14,np.int64);first[1]=d;first[2:6]=[0,2,0,2];first[12]=weight;first[13]=1;ops.append(first)
            e=[v*weight for v in av]
            if not full_bound:
                second=first.copy();second[0]=1;second[6]=1;second[10]=2;second[11]=d;second[13]=2;ops.append(second)
                e[0]+=2*x*weight;e[2]+=2*x*weight
            expected.append(e);offsets.append(len(ops))
        meta=np.asarray(ops,np.int64);offsets=np.asarray(offsets,np.int64)
        order=np.arange(nd,dtype=np.int64);sizes=np.full(nd,64,np.int64);dw=np.full(nd,16,np.uint8);sw=np.full(2,16,np.uint8)
        ssh=np.asarray([[2,2],[2,2]],np.int64);dsh=np.asarray([[2,2]]*nd,np.int64)
        ix=np.asarray([0,1,0,0],np.int32);maps=ix.copy();eligible=K.eligibility(meta,offsets,dsh)
        targets=List.empty_list(types.Array(types.uint8,1,'C'))
        for _ in order:targets.append(np.empty(0,np.uint8))
        K.allocate(targets,order,0,nd,sizes,dw,dsh,eligible,overwrite)
        lo=np.zeros(nd,np.uint64);hi=lo.copy()
        K.compute_wave(blocks,targets,order,0,nd,sw,dw,ssh,dsh,offsets,meta,ix,maps,eligible,overwrite,True,lo,hi)
        for d in order:
            z=targets[d].view(np.uint64);got=[int(z[2*k])+(int(z[2*k+1])<<64) for k in range(4)]
            assert got==expected[d] and int(lo[d])+(int(hi[d])<<64)==max(got)
            checks+=len(got)
        assert all(np.array_equal(a,b) for a,b in zip(blocks,original))
        # Reclamation occurs only after the complete wave has joined.
        heads=np.full(nd,-1,np.int64);heads[-1]=0;nxt=np.asarray([1,-1],np.int64)
        K.release_sources(blocks,heads,nxt,0,nd,np.empty(0,np.uint8))
        assert all(len(a)==0 for a in blocks)
    return dict(status='PASS',parallel_destination_coefficient_checks=checks,
        shared_sources_unchanged=True,release_after_wave=True,M36_bound_edge_test=True)

if __name__=='__main__':
    import os,sys,json
    from pathlib import Path
    os.environ.setdefault('NUMBA_NUM_THREADS','4')
    root=Path(__file__).resolve().parent
    sys.path.insert(0,str(root/('engine' if (root/'engine').is_dir() else 'candidate')))
    print(json.dumps(run_checks()))
