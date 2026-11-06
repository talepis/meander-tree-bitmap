"""Whole-computation A/B benchmark, one job at a time. No server/network operations."""
import argparse
import gc
import json
import statistics
import threading
import time
from pathlib import Path
from server_meander import HERE, configure, JobLock, save
from server_resources import Resources, GiB


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('n', type=int)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    r = Resources()
    if not 1 <= a.n <= 36 or not 1 <= a.threads <= r.cpu_limit or not 1 <= a.repetitions <= 20:
        p.error('Invalid order, allocated thread count or repetition count')
    if a.output.exists():
        raise FileExistsError(a.output)
    from verify_bundle import main as verify
    verify()
    a.output.mkdir(parents=True)
    configure(a.threads, HERE / '.numba_cache')
    import psutil
    import target_geometry as G
    from linear_support import calculate
    proc = psutil.Process()
    # Same fresh memory reserve and arithmetic in every call, no matrix spill.
    r.install_guard(3 * GiB)
    flags = dict(workers=a.threads, timeout=24*3600, dense_cut=-125,
                 minimize_grammar=True, split_maps=True, lifetime=True,
                 geometry_mode='target', parallel_geometry=a.threads, numeric_mode='parallel')
    warm = {}
    for mode in ('baseline', 'bitmap'):
        G.MODE = mode
        start = time.perf_counter()
        d = calculate(12, **flags)
        assert d['Mn'] == '678390116'
        warm[mode] = time.perf_counter() - start
    rows = []
    expected = None
    for rep in range(a.repetitions):
        # Reverse order on alternating repetitions to expose clock/drift effects.
        for mode in (('baseline', 'bitmap') if rep % 2 == 0 else ('bitmap', 'baseline')):
            G.MODE = mode
            gc.collect()
            peak = [proc.memory_info().rss]
            stop = threading.Event()
            def sample():
                while not stop.wait(.005):
                    peak[0] = max(peak[0], proc.memory_info().rss)
            worker = threading.Thread(target=sample, daemon=True)
            worker.start()
            try:
                start = time.perf_counter()
                result = calculate(a.n, memory_mib=max(1,(r.snapshot()['effective_available_bytes']-6*GiB)/(1<<20)),
                                   index_prealloc_bytes=2701940180 if a.n==30 else 0, **flags)
                elapsed = time.perf_counter()-start
            finally:
                stop.set(); worker.join()
            if expected is None:
                expected = result['Mn']
            assert result['status'] == 'PASS' and result['Mn'] == expected
            row = dict(rep=rep, mode=mode, n=a.n, wall_seconds=elapsed,
                       RSS_sampled_peak_bytes=peak[0], Mn=result['Mn'], details=result)
            rows.append(row)
            save(a.output / ('%s_%d.json' % (mode,rep)),row)
            print(json.dumps({k:v for k,v in row.items() if k!='details'}),flush=True)
    med = {m:statistics.median(x['wall_seconds'] for x in rows if x['mode']==m) for m in warm}
    save(a.output / 'SUMMARY.json',dict(status='PASS', n=a.n, Mn=expected, threads=a.threads,
         warmup_seconds=warm, median_wall_seconds=med,
         time_reduction_percent=100*(1-med['bitmap']/med['baseline']),
         independent_CRT=False, RSS_note='5 ms sampled RSS, not coefficient RAM',
         compilation_seconds=None, compilation_note='Warm-up separate; remaining lazy JIT/cache loading stays in measured call'))


if __name__ == '__main__':
    lock = JobLock(HERE / '.NUMERICAL_JOB.lock')
    try:
        main()
    finally:
        lock.close()
