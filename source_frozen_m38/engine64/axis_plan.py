"""Exact structural classes for axis unions and rank translations."""
import numpy as np
from numba import njit,types
from numba.typed import Dict,List
import reference_geometry as R
ARR=types.Array(types.int64,1,'C')
IARR=types.Array(types.int64,1,'C')
KEY6=types.UniTuple(types.int64,6)
KEY2=types.UniTuple(types.int64,2)
KEY3=types.UniTuple(types.int64,3)

@njit(cache=True,nogil=True)
def make_plan(keys,heads,descs,rowids,colids,heights,tab,dense_cut):
    dims,cost=R.plan(keys,heads,descs,heights,tab,dense_cut)
    terms=Dict.empty(KEY6,types.int64);classes=Dict.empty(KEY2,types.int64);trie=Dict.empty(KEY2,types.int64)
    z=np.int64(0);tkeys=[(z,z,z,z,z,z)];tkeys.pop();reps=[(z,z)];reps.pop()
    class_terms=List.empty_list(IARR);class_dims=[z];class_dims.pop();identities=[z];identities.pop()
    dr=np.empty(len(descs),np.int64);dc=np.empty(len(descs),np.int64)
    for j in range(len(descs)):
        v=descs[j];si=v[0]
        for axis in range(2):
            sid=rowids[si] if axis==0 else colids[si]
            lo=v[2] if axis==0 else v[4];hi=v[3] if axis==0 else v[5]
            mapped=v[6]==(2 if axis==0 else 1)
            shift=(v[9] if axis==0 else v[10]) if mapped else (v[7] if axis==0 else v[8])
            start=v[11] if mapped else np.int64(0)
            key=(sid,lo,hi,np.int64(mapped),shift,start)
            if key not in terms:
                terms[key]=len(tkeys);tkeys.append(key);reps.append((np.int64(axis),si))
            if axis==0:dr[j]=terms[key]
            else:dc[j]=terms[key]
    rowclass=np.empty(len(keys),np.int64);colclass=rowclass.copy();raw_work=np.int64(0)
    for di in range(len(keys)):
        full=dims[di,0]==0 and dims[di,1]==0
        for axis in range(2):
            dimension=tab[keys[di,1],keys[di,2]] if axis==0 else tab[heights[keys[di,0]],keys[di,2]]
            node=np.int64(0)
            if full:
                ckey=(np.int64(0),dimension);union=np.empty(0,np.int64)
            else:
                count=0;j=heads[di]
                while j!=-1:count+=1;j=descs[j,14]
                union=np.empty(count,np.int64);at=0;j=heads[di]
                while j!=-1:
                    tid=dr[j] if axis==0 else dc[j];union[at]=tid;at+=1
                    raw_work+=tkeys[tid][2]-tkeys[tid][1];j=descs[j,14]
                union=np.unique(union)
                for tid in union:
                    pair=(node,tid)
                    if pair not in trie:trie[pair]=len(trie)+1
                    node=trie[pair]
                ckey=(np.int64(1),node)
            if ckey not in classes:
                classes[ckey]=len(class_terms);class_terms.append(union);class_dims.append(dimension);identities.append(np.int64(full))
            cid=classes[ckey];class_dims[cid]=max(class_dims[cid],dimension)
            if axis==0:rowclass[di]=cid
            else:colclass[di]=cid
    tk=np.empty((len(tkeys),6),np.int64);rep=np.empty((len(reps),2),np.int64)
    for i in range(len(tkeys)):
        for k in range(6):tk[i,k]=tkeys[i][k]
        rep[i,0]=reps[i][0];rep[i,1]=reps[i][1]
    unique_work=np.int64(0)
    for c in range(len(class_terms)):
        for tid in class_terms[c]:unique_work+=tk[tid,2]-tk[tid,1]
    return tk,rep,dr,dc,rowclass,colclass,class_terms,np.asarray(class_dims,np.int64),np.asarray(identities,np.uint8),raw_work,unique_work

@njit(cache=True,nogil=True)
def canonicalize(class_axes,rowclass,colclass,mapdata):
    ids=Dict.empty(KEY3,types.int64);axes=List.empty_list(ARR);canonical=np.empty(len(class_axes),np.int64)
    for c in range(len(class_axes)):
        a=class_axes[c];linear=a[-1]-a[0]+1==len(a)
        key=(np.int64(0),np.int64(a[0]),np.int64(len(a))) if linear else (np.int64(1),np.int64(c),np.int64(0))
        if key not in ids:
            ids[key]=len(axes);axes.append(mapdata[int(a[0]):int(a[0])+len(a)] if linear and int(a[0])+len(a)<=len(mapdata) else a)
        canonical[c]=ids[key]
    return axes,canonical[rowclass],canonical[colclass],canonical

@njit(cache=True,nogil=True)
def diagnostic_outputs(outR,outC,rowclass,colclass,tk,dr,dc,heads,descs,meta,mapdata):
    count=max(rowclass.max(),colclass.max())+1;class_axes=List.empty_list(ARR)
    for _ in range(count):class_axes.append(np.empty(0,np.int64))
    raw_axes=np.int64(0)
    for di in range(len(rowclass)):
        for axis in range(2):
            cid=rowclass[di] if axis==0 else colclass[di];a=outR[di] if axis==0 else outC[di]
            raw_axes+=len(a)
            if not len(class_axes[cid]):class_axes[cid]=a
            elif not np.array_equal(class_axes[cid],a):raise ArithmeticError('Structural support class mismatch')
    axes,rids,cids,canonical=canonicalize(class_axes,rowclass,colclass,mapdata)
    used=Dict.empty(KEY2,types.int64);raw=unique=np.int64(0);op=0
    for di in range(len(rowclass)):
        j=heads[di]
        while j!=-1:
            m=meta[op]
            if m[6]!=0:
                for axis in range(2):
                    if not m[11]&(1<<axis):
                        tid=dr[j] if axis==0 else dc[j];cid=canonical[rowclass[di] if axis==0 else colclass[di]]
                        length=tk[tid,2]-tk[tid,1];raw+=length;key=(cid,tid)
                        if key not in used:used[key]=len(used);unique+=length
            op+=1;j=descs[j,14]
    unique_axes=np.int64(0)
    for a in axes:unique_axes+=len(a)
    return axes,rids,cids,raw,unique,raw_axes,unique_axes
