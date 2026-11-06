"""Disjoint-target arithmetic, proven full-coverage overwrite, and hot maxima."""
import numpy as np
from numba import njit,prange
import stream_packed as P
import split_wide as W
U=np.uint64
MASK=U(0xffffffff)

@njit(cache=True,nogil=True)
def eligibility(meta,offsets,dshape):
    out=np.zeros(len(dshape),np.uint8)
    for d in range(len(dshape)):
        if offsets[d]==offsets[d+1]:continue
        op=meta[offsets[d]]
        out[d]=op[6]==0 and op[7]==0 and op[8]==0 and op[3]-op[2]==dshape[d,0] and op[5]-op[4]==dshape[d,1]
    return out

@njit(cache=True,nogil=True)
def allocate(targets,order,begin,end,sizes,widths,shape,eligible,overwrite):
    skipped=np.int64(0)
    for p in range(begin,end):
        d=order[p]
        if overwrite and eligible[d]:
            dest=np.empty(sizes[d],np.uint8)
            logical=shape[d,0]*shape[d,1]*widths[d]
            dest[logical:]=0  # Preserve deterministic padding, at most seven bytes.
            skipped+=logical
        else:dest=np.zeros(sizes[d],np.uint8)
        targets[d]=dest
    return skipped

@njit(cache=True,nogil=True)
def copy_scaled(T,V,weight):
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):T[i,j]=U(V[i,j])*U(weight)

@njit(cache=True,nogil=True,parallel=True)
def copy_scaled_parallel(T,V,weight):
    for i in prange(V.shape[0]):
        for j in range(V.shape[1]):T[i,j]=U(V[i,j])*U(weight)

@njit(cache=True,nogil=True)
def copy_to_target(V,dest,dw,nr,nc,weight,threshold):
    count=nr*nc
    if dw==1:
        T1=dest[:count].reshape(nr,nc)
        if count>=threshold:copy_scaled_parallel(T1,V,weight)
        else:copy_scaled(T1,V,weight)
    elif dw==2:
        T2=dest.view(np.uint16)[:count].reshape(nr,nc)
        if count>=threshold:copy_scaled_parallel(T2,V,weight)
        else:copy_scaled(T2,V,weight)
    elif dw==4:
        T4=dest.view(np.uint32)[:count].reshape(nr,nc)
        if count>=threshold:copy_scaled_parallel(T4,V,weight)
        else:copy_scaled(T4,V,weight)
    else:
        T8=dest.view(np.uint64)[:count].reshape(nr,nc)
        if count>=threshold:copy_scaled_parallel(T8,V,weight)
        else:copy_scaled(T8,V,weight)

@njit(inline='always')
def wide_copy_cell(src,T64,sw,snc,nc,op,i,j):
    lo,hi=W.load(src,0,sw,(op[2]+i)*snc+op[4]+j);weight=U(op[12])
    p0=(lo&MASK)*weight;p1=(lo>>U(32))*weight+(p0>>U(32))
    at=2*(i*nc+j);T64[at]=(p0&MASK)|((p1&MASK)<<U(32));T64[at+1]=hi*weight+(p1>>U(32))

@njit(cache=True,nogil=True,parallel=True)
def wide_copy_parallel(src,T64,sw,snc,nr,nc,op):
    for ii in prange(nr):
        i=np.int64(ii)
        for j in range(nc):wide_copy_cell(src,T64,sw,snc,nc,op,i,j)

@njit(cache=True,nogil=True)
def overwrite_first(src,dest,sw,dw,sshape,dshape,op,threshold):
    nr=dshape[0];nc=dshape[1];count=sshape[0]*sshape[1]
    if dw==16:
        T64=dest.view(np.uint64)
        if nr*nc>=threshold:wide_copy_parallel(src,T64,sw,sshape[1],nr,nc,op)
        else:
            for i in range(nr):
                for j in range(nc):wide_copy_cell(src,T64,sw,sshape[1],nc,op,i,j)
    elif sw==1:
        V1=src[:count].reshape(sshape[0],sshape[1])[op[2]:op[3],op[4]:op[5]]
        copy_to_target(V1,dest,dw,nr,nc,op[12],threshold)
    elif sw==2:
        V2=src.view(np.uint16)[:count].reshape(sshape[0],sshape[1])[op[2]:op[3],op[4]:op[5]]
        copy_to_target(V2,dest,dw,nr,nc,op[12],threshold)
    elif sw==4:
        V4=src.view(np.uint32)[:count].reshape(sshape[0],sshape[1])[op[2]:op[3],op[4]:op[5]]
        copy_to_target(V4,dest,dw,nr,nc,op[12],threshold)
    elif sw==8:
        V8=src.view(np.uint64)[:count].reshape(sshape[0],sshape[1])[op[2]:op[3],op[4]:op[5]]
        copy_to_target(V8,dest,dw,nr,nc,op[12],threshold)
    else:
        # A conservative old cap can give a wide source with small actual values.
        for i in range(nr):
            for j in range(nc):
                lo,hi=W.load(src,0,sw,(op[2]+i)*sshape[1]+op[4]+j)
                lo,hi=W.mul_small(lo,hi,U(op[12]))
                if hi:raise ArithmeticError('Narrow overwrite exceeds certified width')
                W.store(dest,0,dw,i*nc+j,lo,hi)

@njit(cache=True,nogil=True)
def maximum(block,w,nr,nc):
    count=nr*nc;hi=U(0)
    if w==1:return U(block[:count].max()),hi
    if w==2:return U(block.view(np.uint16)[:count].max()),hi
    if w==4:return U(block.view(np.uint32)[:count].max()),hi
    if w==8:return block.view(np.uint64)[:count].max(),hi
    z=block.view(np.uint64);lo=U(0)
    for k in range(count):
        a=z[2*k];b=z[2*k+1]
        if b>hi or (b==hi and a>lo):lo=a;hi=b
    return lo,hi

@njit(cache=True,nogil=True)
def one(blocks,dest,d,sw,dw,sshape,dshape,ops,indices,maps,threshold,overwrite,eligible):
    begin=0
    if overwrite and eligible:
        first=ops[0];s=first[0]
        overwrite_first(blocks[s],dest,sw[s],dw[d],sshape[s],dshape[d],first,threshold);begin=1
    remaining=ops[begin:];narrow=dw[d]<=8
    if narrow:
        for op in remaining:
            if sw[op[0]]>8:narrow=False;break
    if narrow:P.execute(blocks,dest,sw,dw,sshape,dshape,remaining,indices,maps,threshold)
    else:
        for op in remaining:
            s=op[0];src=blocks[s];count=(op[3]-op[2])*(op[5]-op[4])
            if count<threshold or dw[d]!=16:W.serial(src,dest,0,0,sw[s],dw[d],sshape[s,1],dshape[d,1],op,indices,maps)
            elif op[6]==2:W.parallel_cols(src,dest.view(np.uint64),0,0,sw[s],dw[d],sshape[s,1],dshape[d,1],op,indices,maps)
            else:W.parallel_rows(src,dest.view(np.uint64),0,0,sw[s],dw[d],sshape[s,1],dshape[d,1],op,indices,maps)

@njit(cache=True,nogil=True,parallel=True)
def compute_wave(blocks,targets,order,begin,end,sw,dw,sshape,dshape,offsets,meta,indices,maps,eligible,overwrite,hotmax,lows,highs):
    # No nested OpenMP. Every destination owns its array; source lists are read-only.
    for pp in prange(begin,end):
        p=np.int64(pp);d=order[p];dest=targets[d];ops=meta[offsets[d]:offsets[d+1]]
        one(blocks,dest,d,sw,dw,sshape,dshape,ops,indices,maps,2**62,overwrite,eligible[d])
        if hotmax:lows[d],highs[d]=maximum(dest,dw[d],dshape[d,0],dshape[d,1])

@njit(cache=True,nogil=True)
def compute_serial(blocks,targets,order,begin,end,sw,dw,sshape,dshape,offsets,meta,indices,maps,eligible,overwrite,hotmax,lows,highs,threshold):
    for p in range(begin,end):
        d=order[p];dest=targets[d];ops=meta[offsets[d]:offsets[d+1]]
        one(blocks,dest,d,sw,dw,sshape,dshape,ops,indices,maps,threshold,overwrite,eligible[d])
        if hotmax:lows[d],highs[d]=maximum(dest,dw[d],dshape[d,0],dshape[d,1])

@njit(cache=True,nogil=True)
def release_sources(blocks,heads,nxt,begin,end,empty):
    for p in range(begin,end):
        s=heads[p]
        while s!=-1:blocks[s]=empty;s=nxt[s]

@njit(cache=True,nogil=True)
def wave_work(order,begin,end,offsets,meta):
    count=np.int64(0)
    for p in range(begin,end):
        d=order[p]
        for j in range(offsets[d],offsets[d+1]):count+=(meta[j,3]-meta[j,2])*(meta[j,5]-meta[j,4])
    return count

@njit(cache=True,nogil=True)
def wave_decision(order,begin,end,offsets,meta,sw,dw,workers,threshold):
    total=0.;maximum_work=0.;inner_estimate=0.
    for p in range(begin,end):
        d=order[p];work=0.;wide_source=False
        for j in range(offsets[d],offsets[d+1]):
            if sw[meta[j,0]]>8:wide_source=True
        for j in range(offsets[d],offsets[d+1]):
            op=meta[j];nr=op[3]-op[2];nc=op[5]-op[4];cells=nr*nc
            axis=nc if op[6]==2 else nr
            parallel=cells>=threshold and (not wide_source or dw[d]==16)
            lanes=min(workers,axis) if parallel else 1
            work+=cells;inner_estimate+=cells/max(1,lanes)
        maximum_work=max(maximum_work,work);total+=work
    return end-begin>1 and total>=65536 and maximum_work<0.9*inner_estimate
