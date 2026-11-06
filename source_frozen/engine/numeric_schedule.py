"""Exact liveness scheduling and RAM-bounded parallel destination waves."""
import heapq
import numpy as np
from numba import njit
import lifetime as original

@njit(cache=True,nogil=True)
def evaluate(order,meta,source_sizes,target_sizes):
    ns=len(source_sizes);nd=len(target_sizes);rank=np.empty(nd,np.int64)
    for k in range(nd):rank[order[k]]=k
    last=np.full(ns,-1,np.int64)
    for op in meta:last[op[0]]=max(last[op[0]],rank[op[1]])
    release=np.zeros(nd,np.int64);current=np.int64(0)
    for s in range(ns):
        if last[s]>=0:release[last[s]]+=source_sizes[s];current+=source_sizes[s]
    peak=current
    for k in range(nd):
        current+=target_sizes[order[k]];peak=max(peak,current);current-=release[k]
    return last,peak

@njit(cache=True,nogil=True)
def greedy_order(meta,source_sizes,target_sizes):
    ns=len(source_sizes);nd=len(target_sizes)
    if ns>=2**32 or nd>=2**32:raise OverflowError('Scheduler key domain')
    keys=np.empty(len(meta),np.uint64)
    for j in range(len(meta)):keys[j]=(np.uint64(meta[j,0])<<np.uint64(32))|np.uint64(meta[j,1])
    keys.sort();unique=0
    for j in range(len(keys)):
        if j==0 or keys[j]!=keys[j-1]:unique+=1
    sources=np.empty(unique,np.int64);dests=np.empty(unique,np.int64);at=0
    for j in range(len(keys)):
        if j==0 or keys[j]!=keys[j-1]:
            sources[at]=np.int64(keys[j]>>np.uint64(32));dests[at]=np.int64(keys[j]&np.uint64(0xffffffff));at+=1
    scount=np.zeros(ns,np.int64);dcount=np.zeros(nd,np.int64)
    for j in range(unique):scount[sources[j]]+=1;dcount[dests[j]]+=1
    soff=np.zeros(ns+1,np.int64);doff=np.zeros(nd+1,np.int64)
    for j in range(ns):soff[j+1]=soff[j]+scount[j]
    for j in range(nd):doff[j+1]=doff[j]+dcount[j]
    fanin=np.empty(unique,np.int64);cursor=doff[:-1].copy()
    for j in range(unique):d=dests[j];fanin[cursor[d]]=sources[j];cursor[d]+=1
    remaining=scount.copy();freed=np.zeros(nd,np.int64);version=np.zeros(nd,np.int64);done=np.zeros(nd,np.uint8)
    for s in range(ns):
        if remaining[s]==1:freed[dests[soff[s]]]+=source_sizes[s]
    z=np.int64(0);heap=[(z,z,z,z)];heap.pop()
    for d in range(nd):heapq.heappush(heap,(target_sizes[d]-freed[d],target_sizes[d],np.int64(d),version[d]))
    order=np.empty(nd,np.int64);at=0
    while heap:
        net,size,d,v=heapq.heappop(heap)
        if done[d] or version[d]!=v:continue
        done[d]=1;order[at]=d;at+=1
        for j in range(doff[d],doff[d+1]):
            s=fanin[j];remaining[s]-=1
            if remaining[s]==1:
                nxt=np.int64(-1)
                for k in range(soff[s],soff[s+1]):
                    if not done[dests[k]]:nxt=dests[k];break
                if nxt<0:raise ArithmeticError('Missing last consumer')
                freed[nxt]+=source_sizes[s];version[nxt]+=1
                heapq.heappush(heap,(target_sizes[nxt]-freed[nxt],target_sizes[nxt],nxt,version[nxt]))
    if at!=nd:raise ArithmeticError('Incomplete liveness order')
    return order

@njit(cache=True,nogil=True)
def improved_plan(meta,source_sizes,target_sizes):
    order,last,peak,base_peaks=original.static_plan(meta,source_sizes,target_sizes)
    candidate=greedy_order(meta,source_sizes,target_sizes)
    clast,cpeak=evaluate(candidate,meta,source_sizes,target_sizes)
    # Never trade a larger predicted coefficient peak for a new order.
    if cpeak<peak:return candidate,clast,cpeak,base_peaks,True,peak
    return order,last,peak,base_peaks,False,peak

@njit(cache=True,nogil=True)
def bounded_waves(order,last,source_sizes,target_sizes,peak_limit,workers,byte_limit):
    nd=len(order);release=np.zeros(nd,np.int64);current=np.int64(0)
    for s in range(len(last)):
        if last[s]>=0:release[last[s]]+=source_sizes[s];current+=source_sizes[s]
    waves=np.empty((nd,2),np.int64);count=0;position=0;actual_peak=current
    while position<nd:
        end=position;need=np.int64(0)
        while end<nd and end-position<workers:
            size=target_sizes[order[end]]
            if end>position and (need+size>byte_limit or current+need+size>peak_limit):break
            if current+need+size>peak_limit:raise MemoryError('Wave exceeds certified liveness peak')
            need+=size;end+=1
        if end==position:raise MemoryError('Empty arithmetic wave')
        current+=need;actual_peak=max(actual_peak,current)
        for k in range(position,end):current-=release[k]
        waves[count,0]=position;waves[count,1]=end;count+=1;position=end
    if current!=np.sum(target_sizes):raise ArithmeticError('Wave release accounting')
    return waves[:count].copy(),actual_peak
