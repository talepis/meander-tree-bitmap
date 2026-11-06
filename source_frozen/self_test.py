#!/usr/bin/env python3
"""Run on the destination BEFORE M36. No full Catalan arrays or matrix files."""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from server_meander import HERE, configure, dimensions, save, JobLock
from server_resources import Resources, cgroup_paths

KNOWN = [0, 1, 2, 8, 42, 262, 1828, 13820, 110954, 933458,
         8152860, 73424650, 678390116]


def brute(n):
    pairs = []
    def rec(i, stack, matching):
        if i == 2*n:
            pairs.append(tuple(matching)); return
        if len(stack) < 2*n-i:
            rec(i+1, stack+[i], matching)
        if stack:
            j = stack[-1]; m = matching.copy(); m[i] = j; m[j] = i
            rec(i+1, stack[:-1], m)
    rec(0, [], [-1]*(2*n))
    answer = 0
    for low in pairs:
        for up in pairs:
            v = 0; k = 0
            while True:
                v = up[low[v]]; k += 1
                if v == 0:
                    break
            answer += k == n
    return answer, len(pairs)**2


def main():
    if not __debug__:
        raise RuntimeError('Tests must not run with Python -O')
    p = argparse.ArgumentParser(); p.add_argument('--threads', type=int, default=2)
    p.add_argument('--extended', action='store_true', help='Also run a full M26')
    p.add_argument('--output', type=Path, default=Path('SELF_TEST_RESULT.json'))
    a = p.parse_args(); start = time.perf_counter(); resource = Resources()
    if not 1 <= a.threads <= resource.cpu_limit:
        raise ValueError('Requested test threads exceed allocation')
    configure(a.threads, HERE/'.numba_cache')
    import memory_release
    memory_release.configure("full")
    from memory_smoke import run_checks as memory_checks
    memory_test=memory_checks()
    import numpy as np
    import wide_kernels as W
    import split_wide as SW
    from linear_support import calculate
    import target_geometry
    target_geometry.MODE = 'bitmap'
    from geometry_smoke import run_checks as geometry_checks
    bitmap_checks = geometry_checks()
    from deterministic import Compiler
    from weighted import Operators
    rng = random.Random(20261103); mask = (1<<64)-1
    split = lambda x: (np.uint64(x & mask), np.uint64(x>>64))
    checks = 0
    for n in (1, 26, 30, 34, 35, 36):
        C = math.comb(2*n, n)//(n+1); G = C*C; gl, gh = split(G)
        for j in range(512):
            x = rng.randrange(G+1); y = rng.randrange(G+1)
            if j < 6:
                x = [0, 1, min(G, mask), min(G, 1<<64), G-1, G][j]
            w = rng.randrange(1<<32); xl, xh = split(x); yl, yh = split(y)
            lo, hi = W.mul_capped(xl, xh, np.uint64(w), gl, gh)
            assert int(lo)+(int(hi)<<64) == min(G, x*w)
            lo, hi = W.plus_capped(xl, xh, yl, yh, gl, gh)
            assert int(lo)+(int(hi)<<64) == min(G, x+y)
            checks += 2
        # Two individually safe terms whose sum and product bounds overflow
        # 128 bits. Only the bounds may saturate, never actual coefficients.
        meta = np.zeros((2,14),np.int64); meta[:,12] = 0xffffffff; meta[:,13] = 1
        lo, hi = W.caps(np.array([gl]), np.array([gh]), meta, 1, gl, gh)
        assert int(lo[0])+(int(hi[0])<<64) == G
        badl, badh = split(G+1)
        try:
            W.caps(np.array([badl]), np.array([badh]), meta, 1, gl, gh)
        except ArithmeticError:
            pass
        else:
            raise AssertionError('Source above global bound was accepted')
        checks += 2
    # Exercise the explicit machine-overflow branches as well.
    G = (1<<128)-1; gl, gh = split(G)
    for x,y in [(G,G),(G,1),(1<<127,1<<127)]:
        lo,hi = W.plus_capped(*split(x),*split(y),gl,gh)
        assert int(lo)+(int(hi)<<64)==min(G,x+y); checks += 1
    lo,hi = W.mul_capped(gl,gh,np.uint64(0xffffffff),gl,gh)
    assert int(lo)+(int(hi)<<64)==G; checks += 1
    # M37's Catalan-square bound exceeds 128 bits. Its separate cap routine
    # must certify the actual source-max based bounds before parallel writes.
    meta=np.zeros((1,14),np.int64);meta[0,12]=16;meta[0,13]=1
    x=1<<120;lo,hi=SW.caps_exact_128(np.array([np.uint64(x&mask)]),
        np.array([np.uint64(x>>64)]),meta,1)
    assert int(lo[0])+(int(hi[0])<<64)==x*16;checks+=1
    meta[0,12]=2;x=1<<127
    try:SW.caps_exact_128(np.array([np.uint64(x&mask)]),
        np.array([np.uint64(x>>64)]),meta,1)
    except OverflowError:pass
    else:raise AssertionError('M37 128-bit product overflow was accepted')
    checks+=1
    meta=np.zeros((2,14),np.int64);meta[:,12]=1;meta[:,13]=1;x=1<<127
    try:SW.caps_exact_128(np.array([np.uint64(x&mask)]),
        np.array([np.uint64(x>>64)]),meta,1)
    except OverflowError:pass
    else:raise AssertionError('M37 128-bit sum overflow was accepted')
    checks+=1
    # Synthetic ranks across 2**31, without allocating a multi-billion array.
    from factor_metadata import rank_at
    from reference_geometry import lower
    rank_axis=np.asarray([2**31-1,2**31+1,2**31+5],np.uint32)
    assert lower(rank_axis,np.int64(2**31+1))==1
    assert rank_at(rank_axis,np.int64(2**31+5))==2
    assert rank_at(rank_axis,np.int64(2**31+1))==1
    checks+=3
    from audit_m36 import run_audit
    domain_audit=run_audit()
    from numeric_smoke import run_checks
    numeric_smoke=run_checks()
    dims = dimensions(36)
    dims37=dimensions(37)
    assert dims37['bound_bits']==131 and dims37['max_ballot_rank_count']==3511574910
    ops = Operators(8); compiler = Compiler(36, range(1,19), ops)
    coverage = compiler.lower_count((0,0,0,0,0,0))
    assert coverage == int(dims['catalan'])
    grammar = compiler.compile()
    assert grammar['nodes'] < 1 << (64-48-1)
    # Every reachable lower prefix has a completion and positive weights.
    pending = [(0,0,0,0,0,0)]; seen = set(pending)
    for state in pending:
        assert compiler.lower_count(state) > 0
        edges = compiler.choices(state)
        assert len({opened for _,opened,_ in edges}) == len(edges)
        for after, opened, weight in edges:
            assert weight >= 1
            if after not in seen:
                seen.add(after); pending.append(after)
    assert cgroup_paths('0::/job/sub\n', '1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n') == [
        ('cgroup2',str(Path('/sys/fs/cgroup/job/sub')),''), ('cgroup2',str(Path('/sys/fs/cgroup/job')),''),
        ('cgroup2',str(Path('/sys/fs/cgroup')),'')]
    rows = []; brute_pairs = 0
    for n in range(1,13):
        d = calculate(n,a.threads,timeout=1800,dense_cut=-125,minimize_grammar=True,split_maps=True,lifetime=True,geometry_mode="target",parallel_geometry=a.threads,numeric_mode="parallel")
        assert int(d['Mn']) == KNOWN[n]
        if n <= 6:
            ref, pairs = brute(n); brute_pairs += pairs
            assert int(d['Mn']) == ref
        rows.append(dict(n=n,Mn=d['Mn'],seconds=d['seconds']))
        print(json.dumps(rows[-1]),flush=True)
    if a.extended:
        d = calculate(26,a.threads,timeout=1800,dense_cut=-125,minimize_grammar=True,split_maps=True,lifetime=True,geometry_mode="target",parallel_geometry=a.threads,numeric_mode="parallel")
        assert d['Mn']=='91412898898828176826244'
        rows.append(dict(n=26,Mn=d['Mn'],seconds=d['seconds']))
    receipt = dict(memory_policy=memory_release.POLICY,memory_page_test=memory_test,selected_geometry_engine='bitmap',bitmap_checks=bitmap_checks,status='PASS',platform=sys.platform,python=sys.version,
        bundle_manifest_sha256=hashlib.sha256((HERE/'SHA256_MANIFEST.json').read_bytes()).hexdigest(),domain_audit=domain_audit,numeric_smoke=numeric_smoke,
        arithmetic_checks=checks,brute_pairs=brute_pairs,whole_order_tests=rows,
        M37_dimensions=dims37,
        M36_metadata=dict(grammar_nodes=grammar['nodes'],reachable_complete_prefix_states=len(seen),
            weighted_lower_coverage=str(coverage),**dims),
        threads=a.threads,seconds=time.perf_counter()-start,
        full_M36_executed=False,independent_CRT=False)
    save(a.output,receipt); print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    lock=JobLock(HERE/'.NUMERICAL_JOB.lock')
    try:main()
    finally:lock.close()
