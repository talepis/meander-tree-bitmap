"""Exact last-consumer scheduling and independently reclaimable blocks."""
import heapq
import time
import numpy as np
import psutil
from numba import njit,prange
from numba.typed import List
from numba import types
import packed_kernels as P
import split_packed as SP
import wide_kernels as W
import split_wide as SW

def plan(meta,source_sizes,target_sizes):
    fanin=[set() for _ in target_sizes];consumers=[set() for _ in source_sizes]
    for op in meta:
        s,d=map(int,op[:2]);fanin[d].add(s);consumers[s].add(d)
    source_sizes=list(map(int,source_sizes));target_sizes=list(map(int,target_sizes))
    initial=sum(source_sizes[s] for s,c in enumerate(consumers) if c)
    def simulate(order):
        remain=[len(x) for x in consumers];current=initial;peak=current
        for d in order:
            current+=target_sizes[d];peak=max(peak,current)
            for s in fanin[d]:
                remain[s]-=1
                if remain[s]==0:current-=source_sizes[s]
        if current!=sum(target_sizes):raise ArithmeticError('Liveness accounting')
        return peak
    remain=[set(x) for x in consumers];freed=[0]*len(fanin);version=[0]*len(fanin);done=set();heap=[]
    for s,c in enumerate(remain):
        if len(c)==1:freed[next(iter(c))]+=source_sizes[s]
    def push(d):heapq.heappush(heap, (target_sizes[d]-freed[d],target_sizes[d],d,version[d]))
    for d in range(len(fanin)):push(d)
    order=[]
    while heap:
        _,_,d,v=heapq.heappop(heap)
        if d in done or version[d]!=v:continue
        done.add(d);order.append(d)
        for s in fanin[d]:
            remain[s].remove(d)
            if len(remain[s])==1:
                nxt=next(iter(remain[s]));freed[nxt]+=source_sizes[s];version[nxt]+=1;push(nxt)
    orders=[list(range(len(fanin))),order,sorted(range(len(fanin)),key=lambda d:target_sizes[d])]
    scores=[simulate(x) for x in orders];best=min(range(len(scores)),key=scores.__getitem__)
    return orders[best],fanin,consumers,dict(full_two_arena_bytes=sum(source_sizes)+sum(target_sizes),
        original_order_stream_peak_bytes=scores[0],greedy_stream_peak_bytes=scores[1],
        small_first_stream_peak_bytes=scores[2],selected_stream_peak_bytes=scores[best],selected_order=best,
        coefficient_lower_bound_bytes=max(initial,sum(target_sizes)))

@njit(cache=True,nogil=True,parallel=True)
def maxima(blocks,widths,shape):
    lows=np.zeros(len(widths),np.uint64);highs=lows.copy()
    for j0 in prange(len(widths)):
        j=np.int64(j0);block=blocks[j];count=shape[j,0]*shape[j,1];w=widths[j]
        if w==1:lows[j]=np.uint64(block[:count].max())
        elif w==2:lows[j]=np.uint64(block.view(np.uint16)[:count].max())
        elif w==4:lows[j]=np.uint64(block.view(np.uint32)[:count].max())
        elif w==8:lows[j]=block.view(np.uint64)[:count].max()
        else:
            a=block.view(np.uint64);lo=hi=np.uint64(0)
            for k in range(count):
                vl=a[2*k];vh=a[2*k+1]
                if vh>hi or (vh==hi and vl>lo):lo=vl;hi=vh
            lows[j]=lo;highs[j]=hi
    return lows,highs

@njit(cache=True,nogil=True)
def static_plan(meta,source_sizes,target_sizes):
    ns=len(source_sizes);nd=len(target_sizes);best_peak=np.int64(9223372036854775807)
    natural=np.arange(nd);reverse=natural[::-1].copy();small=np.argsort(target_sizes)
    best_order=natural;best_last=np.empty(ns,np.int64);peaks=np.empty(3,np.int64)
    for variant in range(3):
        order=natural if variant==0 else (reverse if variant==1 else small)
        rank=np.empty(nd,np.int64)
        for k in range(nd):rank[order[k]]=k
        last=np.full(ns,-1,np.int64)
        for op in meta:last[op[0]]=max(last[op[0]],rank[op[1]])
        release=np.zeros(nd,np.int64);current=np.int64(0)
        for k in range(ns):
            if last[k]>=0:release[last[k]]+=source_sizes[k];current+=source_sizes[k]
        peak=current
        for k in range(nd):
            current+=target_sizes[order[k]];peak=max(peak,current);current-=release[k]
        peaks[variant]=peak
        if peak<best_peak:best_peak=peak;best_order=order;best_last=last
    return best_order,best_last,best_peak,peaks


def execute(blocks,soff,doff,sw,dw,sshape,dshape,meta,indices,maps,threshold,split,workspace_limit,prepared=None):
    if not split:raise ValueError('Streaming v3 requires split immutable maps')
    import stream_packed as SP3
    import profile_stream_chunk as SC
    srcbytes=np.diff(soff);dstbytes=np.diff(doff)
    order,last,peak,peaks=static_plan(meta,srcbytes,dstbytes) if prepared is None else prepared
    if peak>workspace_limit:raise MemoryError('Streaming coefficient budget')
    heads=np.full(len(order),-1,np.int64);nxt=np.full(len(last),-1,np.int64)
    empty=np.empty(0,np.uint8)
    for s,k in enumerate(last):
        if k>=0:nxt[s]=heads[k];heads[k]=s
        else:blocks[s]=empty
    targets=List.empty_list(types.Array(types.uint8,1,'C'))
    for _ in order:targets.append(empty)
    if len(meta)>1 and np.any(meta[1:,1]<meta[:-1,1]):meta=meta[np.argsort(meta[:,1],kind='stable')]
    offsets=np.zeros(len(order)+1,np.int64);offsets[1:]=np.cumsum(np.bincount(meta[:,1],minlength=len(order)))
    checked_available=psutil.virtual_memory().available;last_check=time.monotonic();charged_since_check=0;guard_reads=1;position=0
    alloc=transfer=release=0.;eligible=eligible_bytes=oneop_bytes=0
    while position<len(order):
        end=position;need=0
        while end<len(order) and end-position<256:
            size=int(dstbytes[order[end]])
            if end>position and need+size>16<<20:break
            need+=size;end+=1
        now=time.monotonic()
        if charged_since_check+need>=32<<20 or now-last_check>=.05 or need+(3<<30)>checked_available-charged_since_check:
            checked_available=psutil.virtual_memory().available;charged_since_check=0;last_check=now;guard_reads+=1
        if need+(3<<30)>checked_available-charged_since_check:raise MemoryError('Fresh streaming RAM guard')
        charged_since_check+=need
        vals=SC.execute(blocks,targets,order,position,end,dstbytes,heads,nxt,empty,sw,dw,sshape,dshape,offsets,meta,indices,maps,threshold)
        alloc+=vals[0];transfer+=vals[1];release+=vals[2];eligible+=vals[3];eligible_bytes+=vals[4];oneop_bytes+=vals[5]
        position=end
    return targets,dict(full_two_arena_bytes=int(srcbytes.sum()+dstbytes.sum()),actual_coefficient_live_peak_bytes=int(peak),
        allocation_seconds=alloc,transfer_seconds=transfer,release_seconds=release,overwrite_eligible_blocks=eligible,overwrite_eligible_bytes=eligible_bytes,overwrite_single_op_bytes=oneop_bytes,selected_stream_peak_bytes=int(peak),original_order_stream_peak_bytes=int(peaks[0]),reverse_stream_peak_bytes=int(peaks[1]),
        small_first_stream_peak_bytes=int(peaks[2]),memory_guard_queries=guard_reads,scheduler='Compiled static exact last-use; best of forward/reverse/small-first')
