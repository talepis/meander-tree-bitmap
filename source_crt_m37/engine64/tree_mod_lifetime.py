"""Exact last-consumer streaming for the isolated modular tree runs."""
import time
import numpy as np
import psutil
from numba import types
from numba.typed import List

import lifetime as original
import numeric_kernels as K
import numeric_schedule as schedule
import tree_mod_numeric as N

CONFIG = {}


def configure(modulus, workers):
    if modulus not in ((1 << 64), (1 << 64) - 1, 7):
        raise ValueError('Unsupported CRT ring')
    CONFIG.clear()
    CONFIG.update(modulus=modulus, mode=1 if modulus == (1 << 64) - 1 else 2 if modulus == 7 else 0, workers=int(workers))


def static_plan(meta, source_sizes, target_sizes):
    return schedule.improved_plan(meta, source_sizes, target_sizes)


def maxima(blocks, widths, shape):
    return original.maxima(blocks, widths, shape)


def execute(blocks, soff, doff, sw, dw, sshape, dshape, meta, indices, maps,
            threshold, split, workspace_limit, prepared=None):
    if not CONFIG or not split or np.any(sw != 8) or np.any(dw != 8):
        raise ValueError('Modular streaming requires configured split maps and fixed widths')
    srcbytes = np.diff(soff)
    dstbytes = np.diff(doff)
    plan = static_plan(meta, srcbytes, dstbytes) if prepared is None else prepared
    order, last, peak, peaks = plan[:4]
    if peak > workspace_limit:
        raise MemoryError('Modular streaming coefficient budget')
    t = time.perf_counter()
    waves, actual_peak = schedule.bounded_waves(
        order, last, srcbytes, dstbytes, int(peak), CONFIG['workers'], 128 << 20)
    wave_plan_seconds = time.perf_counter() - t
    heads = np.full(len(order), -1, np.int64)
    nxt = np.full(len(last), -1, np.int64)
    empty = np.empty(0, np.uint8)
    for s, k in enumerate(last):
        if k >= 0:
            nxt[s] = heads[k]
            heads[k] = s
        else:
            blocks[s] = empty
    targets = List.empty_list(types.Array(types.uint8, 1, 'C'))
    for _ in order:
        targets.append(empty)
    if len(meta) > 1 and np.any(meta[1:, 1] < meta[:-1, 1]):
        meta = meta[np.argsort(meta[:, 1], kind='stable')]
    offsets = np.zeros(len(order) + 1, np.int64)
    offsets[1:] = np.cumsum(np.bincount(meta[:, 1], minlength=len(order)))
    eligible = np.zeros(len(order), np.uint8)
    checked_available = psutil.virtual_memory().available
    last_check = time.monotonic()
    charged = 0
    guard_reads = 1
    allocation_seconds = compute_seconds = release_seconds = 0.0
    parallel_waves = inner_parallel_waves = 0
    for begin, end in waves:
        begin, end = int(begin), int(end)
        need = sum(int(dstbytes[order[k]]) for k in range(begin, end))
        now = time.monotonic()
        if charged + need >= 32 << 20 or now - last_check >= .05 or need + (3 << 30) > checked_available - charged:
            checked_available = psutil.virtual_memory().available
            charged = 0
            last_check = now
            guard_reads += 1
        if need + (3 << 30) > checked_available - charged:
            raise MemoryError('Fresh modular numeric wave RAM guard')
        charged += need
        t = time.perf_counter()
        K.allocate(targets, order, begin, end, dstbytes, dw, dshape, eligible, False)
        allocation_seconds += time.perf_counter() - t
        t = time.perf_counter()
        args = (blocks, targets, order, begin, end, sshape, dshape,
                offsets, meta, indices, maps, CONFIG['mode'])
        if end - begin == 1:
            N.compute_single(blocks, targets, int(order[begin]), sshape, dshape,
                             offsets, meta, indices, maps, CONFIG['mode'], threshold)
            inner_parallel_waves += 1
        elif need >= 1 << 16:
            N.compute_wave(*args)
            parallel_waves += 1
        else:
            N.compute_serial(*args)
        compute_seconds += time.perf_counter() - t
        t = time.perf_counter()
        K.release_sources(blocks, heads, nxt, begin, end, empty)
        release_seconds += time.perf_counter() - t
    return targets, dict(full_two_arena_bytes=int(srcbytes.sum() + dstbytes.sum()),
        actual_coefficient_live_peak_bytes=int(actual_peak),
        selected_stream_peak_bytes=int(peak),
        baseline_selected_stream_peak_bytes=int(plan[5]),
        greedy_selected=bool(plan[4]),
        original_order_stream_peak_bytes=int(peaks[0]),
        reverse_stream_peak_bytes=int(peaks[1]),
        small_first_stream_peak_bytes=int(peaks[2]),
        memory_guard_queries=guard_reads, wave_count=len(waves),
        parallel_waves=parallel_waves, inner_parallel_waves=inner_parallel_waves,
        allocation_seconds=allocation_seconds, transfer_seconds=compute_seconds,
        release_seconds=release_seconds, wave_planning_seconds=wave_plan_seconds,
        scheduler='Modular exact liveness with bounded disjoint-target waves')
