import numpy as np
from numba import njit,types
from numba.typed import Dict,List
import reference_geometry as R
IARR=types.Array(types.int64,1,'C')
ARR=types.Array(types.int64,1,'C')
KEY2=types.UniTuple(types.int64,2)

@njit(cache=True,nogil=True)
def rank_at(axis,value):
    if axis[-1]-axis[0]+1==len(axis):return value-axis[0]
    k=R.lower(axis,value)
    if k==len(axis) or axis[k]!=value:raise ArithmeticError('Missing factored endpoint rank')
    return k

@njit(cache=True,nogil=True)
def prepare(keys,heads,descs,skeys,rs,cs,heights,tab,terms,dr,dc,axes,rowids,colids):
    metas=np.empty((len(descs),14),np.int64);shape=np.empty((len(keys),2),np.int64)
    plans=Dict.empty(KEY2,types.int64);z=np.int64(0);pids=[z];pids.pop();paxes=[z];paxes.pop();lengths=[z];lengths.pop()
    op=0;raw=np.int64(0)
    for di in range(len(keys)):
        rr=axes[rowids[di]];cc=axes[colids[di]];nr=len(rr);nc=len(cc);shape[di,0]=nr;shape[di,1]=nc
        da=keys[di,1];dh=keys[di,2];db=heights[keys[di,0]];j=heads[di]
        while j!=-1:
            v=descs[j];si=v[0];ri=rs[si];ci=cs[si];sa=skeys[si,1];sh=skeys[si,2];sb=heights[skeys[si,0]]
            fullsource=len(ri)==tab[sa,sh] and len(ci)==tab[sb,sh]
            kind=2 if v[6]==2 else 1;rb=cb=rstart=cstart=np.int64(0);mask=np.int64(0)
            if nr==tab[da,dh] and nc==tab[db,dh] and fullsource:
                if v[6]==0:kind=0;rb=v[2]+v[7];cb=v[4]+v[8]
                elif v[6]==1:cstart=v[11];mask=3
                else:rstart=v[11];mask=3
            else:
                rcontig=ccontig=False;rfirst=cfirst=np.int64(0)
                if v[6]!=2:
                    rfirst=rank_at(rr,np.int64(ri[v[2]])+v[7]);rlast=rank_at(rr,np.int64(ri[v[3]-1])+v[7]);rcontig=rlast-rfirst+1==v[3]-v[2]
                if v[6]!=1:
                    cfirst=rank_at(cc,np.int64(ci[v[4]])+v[8]);clast=rank_at(cc,np.int64(ci[v[5]-1])+v[8]);ccontig=clast-cfirst+1==v[5]-v[4]
                if v[6]==0 and rcontig and ccontig:kind=0;rb=rfirst;cb=cfirst
                else:
                    if rcontig:rstart=rfirst;mask|=1
                    else:
                        tid=dr[j];key=(rowids[di],tid);length=terms[tid,2]-terms[tid,1];raw+=length
                        if key not in plans:plans[key]=len(pids);pids.append(tid);paxes.append(rowids[di]);lengths.append(length)
                        rstart=plans[key]
                    if ccontig:cstart=cfirst;mask|=2
                    else:
                        tid=dc[j];key=(colids[di],tid);length=terms[tid,2]-terms[tid,1];raw+=length
                        if key not in plans:plans[key]=len(pids);pids.append(tid);paxes.append(colids[di]);lengths.append(length)
                        cstart=plans[key]
            row=(v[0],di,v[2],v[3],v[4],v[5],kind,rb,cb,rstart,cstart,mask,v[12],v[13])
            for k in range(14):metas[op,k]=row[k]
            op+=1;j=v[14]
    offsets=np.zeros(len(pids)+1,np.int64)
    for k in range(len(pids)):
        if lengths[k]>1152921504606846975-offsets[k]:raise OverflowError('Factored index arena')
        offsets[k+1]=offsets[k]+lengths[k]
    lists=List.empty_list(IARR);counts=np.zeros(len(axes),np.int64)
    for k in range(len(paxes)):counts[paxes[k]]+=1
    for c in range(len(axes)):lists.append(np.empty(counts[c],np.int64))
    counts[:]=0
    for pid in range(len(paxes)):
        cid=paxes[pid];lists[cid][counts[cid]]=pid;counts[cid]+=1
    for m in metas:
        if m[6]!=0:
            if not m[11]&1:m[9]=offsets[m[9]]
            if not m[11]&2:m[10]=offsets[m[10]]
    outR=List.empty_list(ARR);outC=List.empty_list(ARR)
    for d in range(len(keys)):outR.append(axes[rowids[d]]);outC.append(axes[colids[d]])
    return shape,metas,offsets,np.asarray(pids,np.int64),lists,outR,outC,raw
