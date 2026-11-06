"""Disjoint grammar destinations, nogil preparation, bounded private scratch."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import psutil
from numba import njit
from numba.typed import List
import split_geometry as Q

@njit(cache=True,nogil=True)
def bound(keys,mx,offsets,targets,heights,tab):
    need=1
    for i in range(len(keys)):
        if not mx[i]:continue
        gid,a,h=keys[i]
        for e in range(offsets[gid],offsets[gid+1]):
            b=heights[targets[e]]
            for aa in (a-1,a+1):
                if aa<0 or aa>=len(tab):continue
                for hh in (h-1,h+1):
                    if hh<0 or hh>=tab.shape[1]:continue
                    need=max(need,tab[aa,hh],tab[b,hh])
    return need

class Geometry:
    def __init__(self,g,workers,budget=256<<20):
        self.workers=workers;self.budget=budget;self.pool=ThreadPoolExecutor(max_workers=workers)
        self.graphs=[];self.stats=dict(parallel_stages=0,serial_fallback_stages=0,max_private_scratch_bytes=0,budget_bytes=budget)
        for part in range(workers):
            offsets=[0];ix=[]
            for v in range(g['nodes']):
                ix.extend(e for e in range(int(g['offsets'][v]),int(g['offsets'][v+1])) if int(g['targets'][e])%workers==part)
                offsets.append(len(ix))
            ix=np.asarray(ix,np.int64)
            self.graphs.append((np.asarray(offsets,np.uint32),g['targets'][ix],g['labels'][ix],g['weights'][ix]))

    def prepare(self,*args):
        # Position convention is exactly split_geometry.prepare_split.
        n,step,keys,mx,rs,cs=args[:6];heights=args[10];tab=args[11]
        sizes=[int(bound(keys,mx,off,tgt,heights,tab)) for off,tgt,lab,w in self.graphs]
        scratch=24*sum(sizes)
        total_elements=sum(len(x) for x in rs)+sum(len(x) for x in cs)
        if scratch>self.budget or total_elements<40000 or scratch+(3<<30)>psutil.virtual_memory().available:
            self.stats['serial_fallback_stages']+=1
            return Q.prepare_split(*args)
        self.stats['parallel_stages']+=1;self.stats['max_private_scratch_bytes']=max(self.stats['max_private_scratch_bytes'],scratch)
        def run(part):
            local=list(args);local[6:10]=self.graphs[part];m=sizes[part]
            local[16]=np.empty(m,np.int32);local[17]=np.empty(m,np.int32)
            local[18]=np.zeros(m,np.uint32);local[19]=np.zeros(m,np.uint32)
            local[20]=np.empty(m,np.int32);local[21]=np.empty(m,np.int32)
            local[22]=0;local[24]=np.empty(0,np.int32)
            return Q.prepare_split(*local)
        results=list(self.pool.map(run,range(self.workers)))
        keys=[];shapes=[];metas=[];indices=[];outR=List.empty_list(Q.ARR);outC=List.empty_list(Q.ARR);dbase=ibase=0
        for kk,ss,mm,ii,rr,cc,epoch in results:
            mm=mm.copy();mm[:,1]+=dbase
            for op in mm:
                if op[6]:
                    if not op[11]&1:op[9]+=ibase
                    if not op[11]&2:op[10]+=ibase
            keys.append(kk);shapes.append(ss);metas.append(mm);indices.append(ii)
            for x in rr:outR.append(x)
            for x in cc:outC.append(x)
            dbase+=len(kk);ibase+=len(ii)
        return np.concatenate(keys),np.concatenate(shapes),np.concatenate(metas),np.concatenate(indices),outR,outC,args[22]

    def close(self):self.pool.shutdown(wait=True)
