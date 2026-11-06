"""Isolated arithmetic experiments; no more live coefficient RAM than the plan."""
import time
import numpy as np
import psutil
from numba import types
from numba.typed import List
import lifetime as original
import numeric_schedule as schedule
import numeric_kernels as K

CONFIG={};LAST_MAXIMA=None
def configure(mode,workers):
    global CONFIG,LAST_MAXIMA
    modes={
        'control':(False,False,False,False),
        'overwrite':(True,False,False,False),
        'hotmax':(False,True,False,False),
        'parallel':(False,False,True,False),
        'schedule':(False,False,False,True),
        'parallel_overwrite':(True,False,True,False),
        'parallel_hotmax':(False,True,True,False),
        'combined':(True,True,True,True),
        'combined_coldmax':(True,False,True,True),
    }
    if mode not in modes:raise ValueError('Unknown arithmetic experiment')
    overwrite,hotmax,parallel,greedy=modes[mode]
    CONFIG=dict(mode=mode,workers=int(workers),overwrite=overwrite,hotmax=hotmax,parallel=parallel,greedy=greedy)
    LAST_MAXIMA=None

def static_plan(meta,source_sizes,target_sizes):
    if CONFIG['greedy']:return schedule.improved_plan(meta,source_sizes,target_sizes)
    p=original.static_plan(meta,source_sizes,target_sizes)
    return (*p,False,p[2])

def maxima(blocks,widths,shape):
    global LAST_MAXIMA
    if CONFIG['hotmax']:
        result=LAST_MAXIMA;LAST_MAXIMA=None
        if result is None:raise ArithmeticError('Missing per-destination maxima')
        return result
    return original.maxima(blocks,widths,shape)

def execute(blocks,soff,doff,sw,dw,sshape,dshape,meta,indices,maps,threshold,split,workspace_limit,prepared=None):
    global LAST_MAXIMA
    if not split:raise ValueError('Numeric experiments require split maps')
    srcbytes=np.diff(soff);dstbytes=np.diff(doff)
    p=static_plan(meta,srcbytes,dstbytes) if prepared is None else prepared
    order,last,peak,peaks=p[:4]
    if peak>workspace_limit:raise MemoryError('Certified coefficient workspace budget')
    t=time.perf_counter()
    waves,actual_peak=schedule.bounded_waves(order,last,srcbytes,dstbytes,int(peak),
        CONFIG['workers'] if CONFIG['parallel'] else 256,(128 if CONFIG['parallel'] else 16)<<20)
    wave_plan_seconds=time.perf_counter()-t
    heads=np.full(len(order),-1,np.int64);nxt=np.full(len(last),-1,np.int64);empty=np.empty(0,np.uint8)
    for s,k in enumerate(last):
        if k>=0:nxt[s]=heads[k];heads[k]=s
        else:blocks[s]=empty
    targets=List.empty_list(types.Array(types.uint8,1,'C'))
    for _ in order:targets.append(empty)
    if len(meta)>1 and np.any(meta[1:,1]<meta[:-1,1]):meta=meta[np.argsort(meta[:,1],kind='stable')]
    offsets=np.zeros(len(order)+1,np.int64);offsets[1:]=np.cumsum(np.bincount(meta[:,1],minlength=len(order)))
    eligible=K.eligibility(meta,offsets,dshape)
    hot=CONFIG['hotmax'];overwrite=CONFIG['overwrite']
    lo=np.zeros(len(order) if hot else 0,np.uint64);hi=np.zeros(len(order) if hot else 0,np.uint64)
    checked_available=psutil.virtual_memory().available;last_check=time.monotonic();charged=0;guard_reads=1
    allocation_seconds=compute_seconds=release_seconds=0.;skipped=0;parallel_waves=0;parallel_targets=0
    for begin,end in waves:
        begin=int(begin);end=int(end);need=sum(int(dstbytes[order[k]]) for k in range(begin,end))
        now=time.monotonic()
        if charged+need>=32<<20 or now-last_check>=.05 or need+(3<<30)>checked_available-charged:
            checked_available=psutil.virtual_memory().available;charged=0;last_check=now;guard_reads+=1
        if need+(3<<30)>checked_available-charged:raise MemoryError('Fresh numeric wave memory guard')
        charged+=need;t=time.perf_counter()
        skipped+=K.allocate(targets,order,begin,end,dstbytes,dw,dshape,eligible,overwrite)
        allocation_seconds+=time.perf_counter()-t;t=time.perf_counter()
        # Small waves stay on the established path; large single blocks retain inner parallelism.
        use_parallel=CONFIG['parallel'] and K.wave_decision(order,begin,end,offsets,meta,sw,dw,CONFIG['workers'],threshold)
        args=(blocks,targets,order,begin,end,sw,dw,sshape,dshape,offsets,meta,indices,maps,eligible,overwrite,hot,lo,hi)
        if use_parallel:
            K.compute_wave(*args);parallel_waves+=1;parallel_targets+=end-begin
        else:K.compute_serial(*args,threshold)
        compute_seconds+=time.perf_counter()-t;t=time.perf_counter()
        K.release_sources(blocks,heads,nxt,begin,end,empty)
        release_seconds+=time.perf_counter()-t
    if hot:LAST_MAXIMA=(lo,hi)
    return targets,dict(full_two_arena_bytes=int(srcbytes.sum()+dstbytes.sum()),
        actual_coefficient_live_peak_bytes=int(actual_peak),selected_stream_peak_bytes=int(peak),
        baseline_selected_stream_peak_bytes=int(p[5]),greedy_selected=bool(p[4]),
        original_order_stream_peak_bytes=int(peaks[0]),reverse_stream_peak_bytes=int(peaks[1]),
        small_first_stream_peak_bytes=int(peaks[2]),memory_guard_queries=guard_reads,
        wave_count=len(waves),parallel_waves=parallel_waves,parallel_targets=parallel_targets,
        overwrite_eligible_blocks=int(eligible.sum()),zero_fill_bytes_avoided=int(skipped),
        allocation_seconds=allocation_seconds,transfer_seconds=compute_seconds,
        release_seconds=release_seconds,wave_planning_seconds=wave_plan_seconds,
        scheduler='Exact liveness order; bounded disjoint-target waves',numeric_mode=CONFIG['mode'])
