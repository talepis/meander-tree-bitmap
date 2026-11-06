import numpy as np
from numba import njit,types
from numba.typed import List
ARR=types.Array(types.int64,1,'C')

@njit(cache=True,nogil=True)
def costs(terms,reps,class_terms,dimensions,identities,rs,cs):
    scratch=dimensions.copy();work=np.zeros(len(dimensions),np.int64);pure=np.zeros(len(dimensions),np.uint8)
    mapped=interval=nonlinear=np.int64(0)
    for c in range(len(dimensions)):
        if identities[c]:scratch[c]=0;pure[c]=1;continue
        all_interval=True
        for tid in class_terms[c]:
            t=terms[tid];axis,si=reps[tid];a=rs[si] if axis==0 else cs[si];n=t[2]-t[1]
            is_interval=t[3]==0 and a[t[2]-1]-a[t[1]]+1==n
            if t[3]:mapped+=n
            elif is_interval:interval+=n
            else:nonlinear+=n
            if not is_interval:all_interval=False
            work[c]+=n
        if all_interval:scratch[c]=0;pure[c]=1;work[c]=len(class_terms[c])*16
    return scratch,work,pure,mapped,interval,nonlinear

@njit(cache=True,nogil=True)
def build_axis(c,terms,reps,class_terms,dimension,identity,pure,rs,cs,mapdata,tags,buf,epoch):
    if identity:return mapdata[:dimension]
    tids=class_terms[c]
    # Sorted interval merge: cardinality/contiguity without enumerating ranks.
    if pure:
        starts=np.empty(len(tids),np.int64);ends=starts.copy()
        for k in range(len(tids)):
            tid=tids[k];t=terms[tid];axis,si=reps[tid];a=rs[si] if axis==0 else cs[si]
            starts[k]=np.int64(a[t[1]])+t[4];ends[k]=np.int64(a[t[2]-1])+t[4]+1
        order=np.argsort(starts);starts=starts[order];ends=ends[order]
        count=0;at=0;left=starts[0];right=ends[0]
        for k in range(1,len(starts)):
            if starts[k]<=right:right=max(right,ends[k])
            else:starts[at]=left;ends[at]=right;at+=1;count+=right-left;left=starts[k];right=ends[k]
        starts[at]=left;ends[at]=right;at+=1;count+=right-left
        if at==1:return mapdata[int(starts[0]):int(ends[0])]
        out=np.empty(count,np.int64);pos=0
        for k in range(at):
            for value in range(starts[k],ends[k]):out[pos]=value;pos+=1
        return out
    nr=0;minimum=4294967295;maximum=0
    for tid in tids:
        t=terms[tid];axis,si=reps[tid];a=rs[si] if axis==0 else cs[si]
        for k in range(t[1],t[2]):
            value=np.int64(mapdata[t[5]+a[k]-t[4]]) if t[3] else np.int64(a[k])+t[4]
            if tags[value]!=epoch:
                tags[value]=epoch;buf[nr]=value;nr+=1;minimum=min(minimum,value);maximum=max(maximum,value)
    if maximum-minimum+1==nr:return mapdata[int(minimum):int(maximum)+1]
    if maximum-minimum+1<=8*nr:
        out=np.empty(nr,np.int64);at=0
        for value in range(minimum,maximum+1):
            if tags[value]==epoch:out[at]=value;at+=1
        if at!=nr:raise ArithmeticError('Factored support scan mismatch')
        return out
    return np.sort(buf[:nr].copy())

@njit(cache=True,nogil=True)
def build_group(group,terms,reps,class_terms,dimensions,identities,pure,rs,cs,mapdata,tags,buf,epoch):
    result=List.empty_list(ARR)
    for c in group:
        epoch+=1;result.append(build_axis(c,terms,reps,class_terms,dimensions[c],identities[c],pure[c],rs,cs,mapdata,tags,buf,epoch))
    return result
