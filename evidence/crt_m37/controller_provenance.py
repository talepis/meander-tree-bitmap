"""Single sequential M37 CRT controller on the already authorised CPU pod."""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import psutil
import subprocess
import sys
import time
import traceback

ROOT = Path('/opt/meander-tree-mod-crt-m37-20261105-001')
PKG = ROOT / 'MEANDER_TREE_MOD_CRT_M37'
PYTHON = '/opt/meander-tree-m35-20261103-001/.venv/bin/python'
POD_UID = 'cc20182f-f1a1-48eb-9a02-ec52e0c4cde9'
SOURCE_DIGEST = 'f4b8f59d5026089e59b1974d7239f30dfff8c556552535d05b197f2d29501c66'
EXPECTED_M37 = '26370652969449510879209096913242792'
M0, M1, M2 = 1 << 64, (1 << 64) - 1, 7
PRODUCT2 = M0 * M1
PRODUCT3 = PRODUCT2 * M2
GIB = 1 << 30
N = 37
GROUPS = [list(range(1, (N + 1) // 2 + 1))[j::4] for j in range(4)]
RINGS = [('pow2', M0), ('ones', M1), ('seven', M2)]


def save(path: Path, value: object) -> None:
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def status(phase: str, **fields) -> None:
    save(ROOT / 'CONTROLLER_STATUS.json', dict(phase=phase, epoch=time.time(), **fields))


def preflight(label: str) -> dict:
    cap = int(Path('/sys/fs/cgroup/memory.max').read_text())
    current = int(Path('/sys/fs/cgroup/memory.current').read_text())
    available = psutil.virtual_memory().available
    if cap != 1900 * GIB or current > 4 * GIB or available < 1700 * GIB:
        raise MemoryError(f'{label}: fresh pod/host memory preflight failed')
    large = []
    for proc in psutil.process_iter(['pid', 'memory_info', 'status']):
        try:
            if proc.pid != os.getpid() and proc.info['status'] != psutil.STATUS_ZOMBIE:
                rss = proc.info['memory_info'].rss if proc.info['memory_info'] else 0
                if rss > GIB:
                    large.append((proc.pid, rss))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if large:
        raise RuntimeError(f'{label}: another large process is active: {large}')
    record = dict(label=label, epoch=time.time(), pod_uid=POD_UID,
                  pod_limit_bytes=cap, cgroup_current_bytes=current,
                  host_available_bytes=available, reserve_bytes=16 * GIB,
                  matrix_files_allowed=False, concurrent_numerical_jobs_allowed=False)
    save(ROOT / f'PREFLIGHT_{label}.json', record)
    return record


def check_source() -> None:
    result = subprocess.run([PYTHON, str(PKG / 'verify_sources.py')],
                            cwd=PKG, capture_output=True, text=True, check=True)
    actual = subprocess.check_output([PYTHON, '-c',
        "import sys;sys.path.insert(0,%r);from common import source_digest;print(source_digest())" % str(PKG)],
        cwd=PKG, text=True).strip()
    if actual != SOURCE_DIGEST:
        raise RuntimeError('Frozen numerical source digest mismatch')
    save(ROOT / 'SOURCE_CHECK.json', dict(status='PASS', digest=actual,
         verification=result.stdout.strip(), pod_uid=POD_UID))


def run_small_checks(env: dict) -> None:
    marker = ROOT / 'SERVER_SMALL_CHECKS.json'
    if marker.exists():
        old = json.loads(marker.read_text())
        if old.get('status') == 'PASS' and old.get('source_digest') == SOURCE_DIGEST:
            return
        raise RuntimeError('Existing small-check marker differs from frozen source')
    status('SMALL_CHECKS')
    results = {}
    for name in ('test_mod_rings.py', 'test_small_full.py'):
        command = [PYTHON, str(PKG / 'tests' / name)]
        result = subprocess.run(command, cwd=PKG, env=env,
                                capture_output=True, text=True, timeout=1800)
        (ROOT / (name + '.log')).write_text(result.stdout + result.stderr)
        if result.returncode or '_PASS:' not in result.stdout:
            raise RuntimeError(f'{name} failed on Linux')
        results[name] = result.stdout.strip()
    save(marker, dict(status='PASS', source_digest=SOURCE_DIGEST, results=results))


def validate_pass(group_index: int, lower_sizes: list[int], ring: str, modulus: int) -> dict:
    name = f'M37_G{group_index}_{ring.upper()}'
    directory = ROOT / 'runs' / name
    result_file = directory / 'RESULT.json'
    state_file = directory / 'STATE.json'
    record = json.loads(result_file.read_text())
    state = json.loads(state_file.read_text())
    if record.get('status') != 'PASS' or state.get('status') != 'PASS':
        raise ArithmeticError(name + ': pass status failed')
    if state.get('result_sha256') != hashlib.sha256(result_file.read_bytes()).hexdigest():
        raise ArithmeticError(name + ': result hash mismatch')
    if record.get('n') != N or record.get('lower_sizes') != lower_sizes or record.get('ring') != ring:
        raise ArithmeticError(name + ': order/group/ring identity mismatch')
    if record.get('source_digest') != SOURCE_DIGEST or record.get('modulus_decimal') != str(modulus):
        raise ArithmeticError(name + ': source or modulus mismatch')
    if len(record.get('steps', [])) != 2 * N - 8:
        raise ArithmeticError(name + ': incomplete stage history')
    if record.get('matrix_read_bytes') != 0 or record.get('matrix_write_bytes') != 0:
        raise ArithmeticError(name + ': matrix disk I/O was used')
    residue = int(record['Mn'])
    if not 0 <= residue < modulus:
        raise ArithmeticError(name + ': invalid residue')
    return dict(group=group_index, lower_sizes=lower_sizes, ring=ring,
                modulus_decimal=str(modulus), residue_decimal=str(residue),
                lower_coverage_decimal=str(record['weighted_lower_coverage']),
                seconds=record['seconds'], sampled_peak_RSS_bytes=record['process_RSS_sampled_peak_bytes'],
                stage_count=len(record['steps']), matrix_read_bytes=0, matrix_write_bytes=0,
                result_sha256=hashlib.sha256(result_file.read_bytes()).hexdigest())


def run_pass(group_index: int, lower_sizes: list[int], ring: str, modulus: int,
             env: dict, pass_number: int, pass_total: int) -> dict:
    name = f'M37_G{group_index}_{ring.upper()}'
    directory = ROOT / 'runs' / name
    if (directory / 'RESULT.json').exists():
        return validate_pass(group_index, lower_sizes, ring, modulus)
    if directory.exists():
        raise RuntimeError(name + ': existing incomplete run; refusing duplicate computation')
    preflight(name)
    directory.mkdir(parents=True)
    config = dict(n=N, lower_sizes=lower_sizes, ring=ring, modulus_decimal=str(modulus),
                  backend='64', threads=16, reserve_bytes=16 * GIB,
                  timeout_seconds=24 * 3600, output=str(directory),
                  source_digest=SOURCE_DIGEST, matrix_files_allowed=False,
                  cache_dir=str(ROOT / 'numba_cache'))
    save(directory / 'CONFIG.json', config)
    stdout = (directory / 'stdout.log').open('wb')
    stderr = (directory / 'stderr.log').open('wb')
    try:
        proc = subprocess.Popen([PYTHON, str(PKG / 'group_worker.py'), str(directory / 'CONFIG.json')],
                                cwd=PKG, env=env, stdin=subprocess.DEVNULL,
                                stdout=stdout, stderr=stderr, start_new_session=True)
    finally:
        stdout.close(); stderr.close()
    start_ticks = int((Path('/proc') / str(proc.pid) / 'stat').read_text().split()[21])
    save(ROOT / (name + '_START.json'), dict(status='STARTED', pod_uid=POD_UID,
         pid=proc.pid, start_ticks=start_ticks, epoch=time.time(),
         group=group_index, ring=ring, source_digest=SOURCE_DIGEST))
    status('RUNNING', n=N, group=group_index, ring=ring,
           pass_number=pass_number, pass_total=pass_total, worker_pid=proc.pid)
    code = proc.wait()
    if code:
        raise RuntimeError(f'{name}: worker exited {code}; see immutable pass logs')
    return validate_pass(group_index, lower_sizes, ring, modulus)


def reconstruct_group(group_index: int, records: list[dict], catalan: int) -> dict:
    coverage = {int(row['lower_coverage_decimal']) for row in records}
    if len(coverage) != 1:
        raise ArithmeticError('Inconsistent group coverage between residue rings')
    bound = catalan * coverage.pop()
    residues = {row['ring']: int(row['residue_decimal']) for row in records}
    a, b = residues['pow2'], residues['ones']
    first = a + M0 * ((b - a) % M1)
    if bound < PRODUCT2:
        if len(records) != 2:
            raise ArithmeticError('Unneeded third pass for a two-ring-bounded group')
        value = first
        product = PRODUCT2
    else:
        if len(records) != 3 or math.gcd(PRODUCT2, M2) != 1:
            raise ArithmeticError('Third coprime residue required for this group')
        multiplier = ((residues['seven'] - first) * pow(PRODUCT2, -1, M2)) % M2
        value = first + multiplier * PRODUCT2
        product = PRODUCT3
    if not 0 <= value <= bound or product <= bound:
        raise ArithmeticError('CRT value outside the proved family bound')
    result = dict(status='CRT_RECONSTRUCTION_MATCH', group=group_index,
                  lower_sizes=GROUPS[group_index - 1], value_decimal=str(value),
                  lower_coverage_decimal=str(bound // catalan),
                  proved_family_bound_decimal=str(bound), CRT_product_decimal=str(product),
                  residues=records, source_digest=SOURCE_DIGEST)
    save(ROOT / f'M37_GROUP_{group_index}_RECONSTRUCTION.json', result)
    return result


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'CONTROLLER.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if (ROOT / 'M37_CRT_RECONSTRUCTION.json').exists():
                raise RuntimeError('M37 CRT already complete; refusing a duplicate series')
            check_source()
            catalan = math.comb(2 * N, N) // (N + 1)
            if PRODUCT2 >= catalan * catalan or PRODUCT3 <= catalan * catalan:
                raise ArithmeticError('Expected two/three-ring uniqueness bounds changed')
            env = os.environ.copy()
            env['MEANDER_CACHE_DIR'] = str(ROOT / 'numba_cache')
            (ROOT / 'numba_cache').mkdir(exist_ok=True)
            run_small_checks(env)
            preflight('SERIES_START')
            reconstructed = []
            pass_number = 0
            for i, lower_sizes in enumerate(GROUPS, start=1):
                from sys import path
                if str(PKG / 'engine64') not in path:
                    path.insert(0, str(PKG / 'engine64'))
                from deterministic import Compiler
                from weighted import Operators
                coverage = Compiler(N, lower_sizes, Operators(8)).lower_count((0, 0, 0, 0, 0, 0))
                bound = coverage * catalan
                rings = RINGS if bound >= PRODUCT2 else RINGS[:2]
                records = []
                for ring, modulus in rings:
                    pass_number += 1
                    records.append(run_pass(i, lower_sizes, ring, modulus, env, pass_number, 11))
                    status('PASS_COMPLETE', n=N, group=i, ring=ring,
                           completed_passes=pass_number, pass_total=11)
                if int(records[0]['lower_coverage_decimal']) != coverage:
                    raise ArithmeticError('Compiler lower coverage mismatch')
                reconstructed.append(reconstruct_group(i, records, catalan))
            if sum(int(x['lower_coverage_decimal']) for x in reconstructed) != catalan:
                raise ArithmeticError('The groups do not cover all lower matchings exactly once')
            if pass_number != 11:
                raise ArithmeticError('Unexpected modular pass schedule')
            value = sum(int(x['value_decimal']) for x in reconstructed)
            if str(value) != EXPECTED_M37:
                raise ArithmeticError('M37 CRT differs from the previous exact result')
            result = dict(status='CRT_RECONSTRUCTION_MATCH', n=N, Mn_decimal=str(value),
                          groups=reconstructed, completed_passes=pass_number,
                          moduli_decimal=[str(M0), str(M1), str(M2)],
                          global_bound_decimal=str(catalan * catalan),
                          source_digest=SOURCE_DIGEST, independent_of_tree_grammar=False,
                          scope='Fresh modular family passes share tree grammar and bitmap geometry with the exact algorithm',
                          matrix_read_bytes=0, matrix_write_bytes=0)
            save(ROOT / 'M37_CRT_RECONSTRUCTION.json', result)
            status('COMPLETE', n=N, Mn=str(value), completed_passes=pass_number)
        except BaseException as exc:
            save(ROOT / 'CONTROLLER_ERROR.json', dict(error=repr(exc), traceback=traceback.format_exc()))
            status('ERROR', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
