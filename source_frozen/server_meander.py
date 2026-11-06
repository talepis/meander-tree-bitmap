#!/usr/bin/env python3
"""Single-job launcher for exact cropped-bitmap tree geometry, audited through M36."""
import argparse
import hashlib
import json
import math
import os
import platform
import sys
import threading
import time
import traceback
from pathlib import Path
from server_resources import Resources, GiB

HERE = Path(__file__).resolve().parent


def save(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def configure(threads, cache):
    os.environ.update(NUMBA_NUM_THREADS=str(threads), NUMBA_THREADING_LAYER='omp',
        OMP_WAIT_POLICY='PASSIVE', KMP_BLOCKTIME='0', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_CACHE_DIR=str(Path(os.environ.get('MEANDER_CACHE_DIR',str(cache))).resolve()))
    sys.path.insert(0, str(HERE / 'engine'))


def dimensions(n):
    def B(m, h):
        if h < 0 or h > m or (m - h) & 1:
            return 0
        k = (m - h) // 2
        return math.comb(m, k) - (math.comb(m, k - 1) if k else 0)
    C = math.comb(2 * n, n) // (n + 1)
    maxB = max(B(m, h) for m in range(n + 1) for h in range(m + 1))
    maps = maxB + sum(B(m, h) for m in range(1, n) for h in range(2, m + 1))
    # map data, duplicate initial index prefix, growing index arena and scratch.
    fixed = 24 * maxB + 4 * maps  # Actual fixed arrays of selected split-map profile.
    if (n < 37 and C * C >= 1 << 128) or maxB >= 1 << 32 or maps >= 1 << 63:
        raise OverflowError('Audited value/index domain exceeded')
    return dict(n=n, catalan=str(C), proved_global_bound_decimal=str(C*C),
        bound_bits=(C*C).bit_length(), max_ballot_rank_count=maxB,
        closemap_entries_including_identity=maps, closemap_bytes=maps*4,
        fixed_support_preflight_bytes=fixed,pooled_scratch_entries=2*maxB,
        pooled_scratch_bytes=24*maxB,global_pool_offsets_dtype='int64',
        maximum_centroid_branch_families=(n+1)//2,
        transfer_stages=2*n-min(8, 2*n),
        rank_value_dtype='uint32', map_offset_dtype='int64',
        total_peak_RAM_prediction_bytes=None,
        total_peak_RAM_prediction_note='Fixed support is NOT total peak. Coefficients and stage metadata are additional; fresh gates remain active.')


class JobLock:
    def __init__(self, path):
        self.f = open(path, 'a+b'); self.f.seek(0)
        self.f.write(b'0'); self.f.flush(); self.f.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def close(self):
        self.f.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('n', type=int, nargs='?', default=36)
    p.add_argument('--inspect', action='store_true', help='Only read resources and exact dimensions; no large arrays')
    p.add_argument('--threads', default='auto')
    p.add_argument('--cache-dir',type=Path,help='Optional short writable JIT-cache path, useful for long Windows paths')
    p.add_argument('--geometry-engine', choices=['auto','baseline','bitmap'], default='auto')
    p.add_argument('--memory-policy',choices=['auto','baseline','eager','full'],default='auto')
    p.add_argument('--numeric-mode',choices=['parallel','baseline'],default='parallel')
    p.add_argument('--profile', choices=['parallel'],default='parallel')
    p.add_argument('--memory-gib', type=float, help='Optional extra process RSS budget, never overrides real allocation')
    p.add_argument('--reserve-gib', type=float, default=16)
    p.add_argument('--timeout-hours', type=float, default=24)
    p.add_argument('--memory-preflight', type=Path, help='Required audited memory decision for a full M37 run')
    p.add_argument('--output', type=Path, default=Path('runs/M36'))
    a = p.parse_args()
    if not 1 <= a.n <= 37:
        p.error('Supported orders are 1..37')
    if a.reserve_gib < 3 or not math.isfinite(a.reserve_gib):
        p.error('Reserve must be finite and at least 3 GiB')
    if a.memory_gib is not None and (not math.isfinite(a.memory_gib) or a.memory_gib <= 0):
        p.error('Invalid memory limit')
    if not math.isfinite(a.timeout_hours) or a.timeout_hours <= 0:
        p.error('Invalid time limit')
    if sys.maxsize < 1 << 32:
        raise RuntimeError('64-bit Python required')
    resources = Resources(a.memory_gib); initial = resources.snapshot()
    threads = resources.default_threads if a.threads == 'auto' else int(a.threads)
    if not 1 <= threads <= resources.cpu_limit:
        p.error('Threads exceed the CPUs actually allocated to this process')
    dims = dimensions(a.n); reserve = int(a.reserve_gib * GiB)
    selected_geometry = ('bitmap' if a.n >= 28 else 'baseline') if a.geometry_engine == 'auto' else a.geometry_engine
    selected_memory = ('full' if sys.platform.startswith('linux') else 'baseline') if a.memory_policy=='auto' else a.memory_policy
    if selected_geometry!='bitmap':
        if a.memory_policy not in ('auto','baseline'):p.error('This memory policy requires bitmap geometry')
        selected_memory='baseline'
    flags=dict(minimize_grammar=True,split_maps=True,lifetime=True,geometry_mode='target',parallel_geometry=threads,numeric_mode=a.numeric_mode)
    report = dict(selected_memory_policy=selected_memory,selected_geometry_engine=selected_geometry,geometry_selection_policy='auto: baseline below 28, bitmap at 28..37; measured full-order gain at M32',selected_numeric_mode=a.numeric_mode,selected_profile=a.profile,resources=initial, dimensions=dims, selected_threads=threads,
        reserve_bytes=reserve, full_M37_previously_run=False,
        independent_CRT=False, platform=platform.platform(), python=sys.version)
    if a.inspect:
        print(json.dumps(report, indent=2)); return
    if a.n == 37:
        if a.memory_preflight is None or not a.memory_preflight.is_file():
            raise RuntimeError('M37 requires a completed memory preflight based on a full bitmap M36')
        decision=json.loads(a.memory_preflight.read_text(encoding='utf-8'))
        predicted=decision.get('predicted_peak_bytes')
        budget=initial['effective_available_bytes']-reserve-(3*GiB)
        if (decision.get('n')!=37 or decision.get('decision')!='RUN' or
                decision.get('source_manifest_sha256')!=hashlib.sha256((HERE/'SHA256_MANIFEST.json').read_bytes()).hexdigest() or
                not isinstance(predicted,int) or predicted<=0 or predicted>budget or
                not decision.get('basis')):
            raise RuntimeError('M37 memory preflight is stale, incomplete, or exceeds fresh guarded RAM')
        report['memory_preflight']=decision
    from verify_bundle import main as verify_sources
    verify_sources()
    receipt_path=HERE/'SELF_TEST_RESULT.json'
    if not receipt_path.is_file():raise RuntimeError('Run self_test.py before a numerical job')
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    manifest_digest=hashlib.sha256((HERE/'SHA256_MANIFEST.json').read_bytes()).hexdigest()
    if receipt.get('status')!='PASS' or receipt.get('bundle_manifest_sha256')!=manifest_digest:
        raise RuntimeError('A passing self-test for these exact sources is required')
    report['self_test_receipt']=str(receipt_path)
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    # One numerical job per package directory, including different output paths.
    lock = JobLock(HERE / '.NUMERICAL_JOB.lock')
    if (out / 'STATE.json').exists() or (out / 'RESULT.json').exists():
        raise FileExistsError('Output already contains a run; choose a fresh output directory')
    run_start = time.perf_counter(); stop = threading.Event(); peak = [initial['process_RSS_bytes']]
    save(out / 'PREFLIGHT.json', report)
    state = dict(status='PREPARING', n=a.n, pid=os.getpid(), output=str(out))
    save(out / 'STATE.json', state)
    arena = initial['effective_available_bytes'] - reserve - dims['fixed_support_preflight_bytes'] - GiB
    if arena <= 0:
        save(out / 'STATE.json', dict(state, status='STOPPED_RESOURCE', reason='Insufficient RAM even for fixed support and reserve'))
        raise MemoryError('Insufficient actual allocation for fixed support; no numerical job started')
    def watch():
        last = 0
        try:
            while not stop.wait(0.005):
                import psutil
                peak[0] = max(peak[0], psutil.Process().memory_info().rss)
                now = time.monotonic()
                if now-last < 0.02:
                    continue
                last = now; snap = resources.snapshot()
                if snap['effective_available_bytes'] < min(8*GiB, reserve//2):
                    save(out/'RESOURCE_STOP.json', dict(status='STOPPED_RESOURCE', resources=snap,
                        reason='Fresh available memory crossed emergency reserve; only this worker exits'))
                    save(out/'STATE.json',dict(state,status='STOPPED_RESOURCE',reason='Fresh memory emergency reserve'))
                    os._exit(75)
        except BaseException as e:
            save(out/'RESOURCE_STOP.json', dict(status='GUARD_ERROR', reason=repr(e)))
            os._exit(76)
    try:
        configure(threads, a.cache_dir.resolve() if a.cache_dir else HERE / '.numba_cache')
        resources.install_guard(reserve)
        import numba, numpy, psutil
        from linear_support import calculate
        import target_geometry
        target_geometry.MODE = selected_geometry
        import memory_release
        memory_release.configure(selected_memory)
        imports = time.perf_counter()-run_start
        watcher = threading.Thread(target=watch, daemon=True); watcher.start()
        latest_peak = [peak[0]]
        with (out / 'STAGES.jsonl').open('a', encoding='utf-8') as log:
            def progress(row):
                snap = resources.snapshot(); latest_peak[0] = peak[0]
                record = dict(row, process_RSS_bytes=snap['process_RSS_bytes'],
                    process_RSS_run_sampled_peak_bytes=peak[0],
                    effective_available_bytes=snap['effective_available_bytes'],
                    cgroups=snap['cgroups'], matrix_read_bytes=0, matrix_write_bytes=0)
                log.write(json.dumps(record)+'\n'); log.flush()
                save(out/'STATE.json', dict(state, status='RUNNING', stage=record))
                print(json.dumps(dict(stage=row['stage'], elapsed=row['elapsed'],
                    RSS_GiB=round(snap['process_RSS_bytes']/GiB, 3))), flush=True)
            result = calculate(a.n, threads, tail=8, memory_mib=arena/(1<<20),
                timeout=a.timeout_hours*3600, dense_cut=-125, progress=progress,
                index_prealloc_bytes=2701940180 if a.n==30 else 0,**flags)
        if a.n == 37:
            result['variant']='SERVER_M37_BITMAP_UINT32_CERTIFIED128_INDEX_POOL_RELEASE'
            result['arithmetic_certification']='Exact source-max cap at every stage, checked before numeric execution; terminal checked 128-bit accumulation'
            result['memory_change']='Completed-stage translation index pool is released when retained capacity is at least twice the needed stage index size; numerical kernels and guards unchanged'
        result.update(selected_memory_policy=selected_memory,selected_geometry_engine=selected_geometry,selected_numeric_mode=a.numeric_mode,selected_profile=a.profile,process_RSS_sampled_peak_bytes=peak[0], RSS_sampling_period_seconds=0.005,
            launcher_import_seconds=imports, launcher_total_seconds=time.perf_counter()-run_start,
            numba=numba.__version__, numpy=numpy.__version__, psutil=psutil.__version__,
            threading_backend=numba.threading_layer(), numerical_job_excludes_independent_CRT=True,
            cache_policy='Persistent local JIT cache; cold compilation is included in first-run phase/total times',
            compilation_seconds=None,
            compilation_note='No fabricated compilation subtraction: first-run JIT is included in the recorded phases. Warm-up separately for a warm benchmark.')
        try:
            import resource
            result['process_RSS_OS_high_water_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except ImportError:
            result['process_RSS_OS_high_water_bytes'] = None
        save(out/'RESULT.json', result)
        save(out/'STATE.json', dict(state, status='COMPLETE', Mn=result['Mn']))
        print(json.dumps({k:v for k,v in result.items() if k!='steps'}, indent=2), flush=True)
    except BaseException as e:
        save(out/'ERROR.json', dict(error=repr(e), traceback=traceback.format_exc()))
        save(out/'STATE.json', dict(state, status='ERROR', reason=repr(e)))
        raise
    finally:
        stop.set(); lock.close()


if __name__ == '__main__':
    main()
