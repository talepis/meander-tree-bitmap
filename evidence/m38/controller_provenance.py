"""Run the frozen M38 family package after the isolated M34 CRT is complete."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path('/opt/meander-m38-families-20261105-001')
PKG = ROOT / 'MEANDER_M38_FAMILIES'
CRT34 = Path('/opt/meander-tree-mod-crt-m34-20261105-001')
PYTHON = '/opt/meander-tree-m35-20261103-001/.venv/bin/python'
SOURCE_DIGEST = 'efda251fc59d0866158215ee2e82658ed9a0a127050514f37265f0ba3eec43e4'
POD_UID = 'cc20182f-f1a1-48eb-9a02-ec52e0c4cde9'
EXPECTED_M32 = '154967080111946539652848747626'
EXPECTED_M34 = '19007633817297768906172279813820'
GIB = 1 << 30


def save(name, value):
    path = ROOT / name
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def status(phase, **fields):
    save('CONTROLLER_STATUS.json', dict(phase=phase, epoch=time.time(), **fields))


def command(label, argv, timeout=None, env=None):
    with (ROOT / (label + '.log')).open('ab') as log:
        p = subprocess.Popen(argv, cwd=PKG, env=env, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(label + '_START.json', dict(pid=p.pid, argv=argv, epoch=time.time()))
        try:
            code = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.terminate()
            p.wait()
            raise TimeoutError(label + ' exceeded its preflight deadline')
    if code:
        raise RuntimeError(label + ' exited with code ' + str(code))


def wait_for_crt():
    while True:
        p = CRT34 / 'CONTROLLER_STATUS.json'
        if p.exists():
            state = json.loads(p.read_text())
            if state.get('phase') == 'ERROR':
                raise RuntimeError('Predecessor M34 CRT failed')
            if state.get('phase') == 'COMPLETE':
                record = json.loads((CRT34 / 'M34_CRT_RECONSTRUCTION.json').read_text())
                if record.get('status') != 'CRT_RECONSTRUCTION_MATCH' or record.get('Mn_decimal') != EXPECTED_M34:
                    raise ArithmeticError('Predecessor CRT record mismatch')
                if len(record.get('passes', [])) != 2:
                    raise ArithmeticError('Predecessor CRT is incomplete')
                save('PREDECESSOR_CRT_GATE.json', dict(status='PASS', predecessor=str(CRT34),
                    n=34, Mn=EXPECTED_M34, passes=2, epoch=time.time()))
                return
        status('WAITING_FOR_M34_CRT')
        time.sleep(30)


def check_source():
    code = "import sys;sys.path.insert(0,%r);from common import source_digest;print(source_digest())" % str(PKG)
    actual = subprocess.check_output([PYTHON, '-c', code], cwd=PKG, text=True).strip()
    if actual != SOURCE_DIGEST:
        raise RuntimeError('Frozen source digest mismatch')
    save('SOURCE_CHECK.json', dict(status='PASS', digest=actual, pod_uid=POD_UID))


def preflight():
    import psutil
    cap = int(Path('/sys/fs/cgroup/memory.max').read_text())
    used = int(Path('/sys/fs/cgroup/memory.current').read_text())
    available = psutil.virtual_memory().available
    if cap != 1900 * GIB or used > 4 * GIB:
        raise MemoryError('Fresh 1900 GiB pod preflight failed')
    large = []
    for proc in psutil.process_iter(['pid', 'memory_info', 'status']):
        try:
            if proc.pid != os.getpid() and proc.info['status'] != psutil.STATUS_ZOMBIE:
                rss = proc.info['memory_info'].rss if proc.info['memory_info'] else 0
                if rss > GIB:
                    large.append(dict(pid=proc.pid, rss_bytes=rss))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if large:
        raise RuntimeError('Another large process is active in this pod: ' + repr(large))
    result = dict(pod_uid=POD_UID, cgroup_limit_bytes=cap, cgroup_current_bytes=used,
                  host_available_bytes=available, reserve_bytes=16 * GIB,
                  matrix_files_allowed=False, epoch=time.time())
    save('FRESH_RESOURCE_PREFLIGHT.json', result)
    return result


def inspect_m38():
    raw = subprocess.check_output([PYTHON, str(PKG / 'server_meander.py'), '38',
        '--groups', '4', '--threads', '16', '--reserve-gib', '16', '--inspect'],
        cwd=PKG, text=True)
    plan = json.loads(raw)
    if plan.get('initial_admission') != 'READY_FOR_GUARDED_GROUPS':
        raise MemoryError('M38 fixed support cannot fit')
    if plan['dimensions']['fixed_support_bytes'] != 642422174744:
        raise ArithmeticError('M38 static dimension mismatch')
    if plan['dimensions']['global_bound_bits'] != 135:
        raise ArithmeticError('M38 upper-bound bit count mismatch')
    if plan['threads'] != 16 or len(plan['groups']) != 4:
        raise RuntimeError('Unexpected CPU or family partition')
    save('M38_INSPECT.json', plan)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'CONTROLLER.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if (ROOT / 'M38_FINAL_VALIDATION.json').exists():
                raise RuntimeError('M38 was already completed; refusing a duplicate run')
            check_source()
            wait_for_crt()
            preflight()
            env = os.environ.copy()
            env['MEANDER_CACHE_DIR'] = str(ROOT / 'numba_cache')
            (ROOT / 'numba_cache').mkdir(exist_ok=True)
            status('LINUX_TESTS')
            command('UNIT_TESTS', [PYTHON, '-m', 'unittest', 'discover', '-s', str(PKG / 'tests'), '-v'],
                    timeout=600, env=env)
            command('LINUX_SELF_TEST', [PYTHON, str(PKG / 'self_test.py'), '--threads', '16'],
                    timeout=1800, env=env)
            receipt = json.loads((PKG / 'SELF_TEST_RESULT.json').read_text())
            if receipt.get('status') != 'PASS' or receipt.get('source_digest') != SOURCE_DIGEST or receipt.get('rank64_boundary_status') != 'PASS':
                raise RuntimeError('Linux source or rank64 self-test failed')
            save('LINUX_TEST_RECEIPT.json', dict(status='PASS', source_digest=SOURCE_DIGEST,
                rank64_boundary_status='PASS', whole_small_orders=len(receipt['whole_small_orders'])))
            preflight()
            status('M32_FULL_GATE')
            m32 = ROOT / 'runs' / 'M32_G4'
            command('M32_FULL_GATE', [PYTHON, str(PKG / 'server_meander.py'), '32', '--groups', '4',
                '--threads', '16', '--reserve-gib', '16', '--timeout-hours', '4', '--resume',
                '--output', str(m32)], timeout=15000, env=env)
            gate = json.loads((m32 / 'RESULT.json').read_text())
            if gate.get('status') != 'PASS' or gate.get('Mn') != EXPECTED_M32 or gate.get('groups_completed') != 4 or gate.get('matrix_read_bytes') or gate.get('matrix_write_bytes'):
                raise ArithmeticError('Linux M32 full result mismatch')
            save('M32_GATE_VALIDATION.json', dict(status='PASS', Mn=EXPECTED_M32,
                source_digest=SOURCE_DIGEST, groups=4, seconds=gate['controller_session_seconds']))
            preflight()
            inspect_m38()
            status('RUNNING_M38', groups=4, completed_groups=0)
            m38 = ROOT / 'runs' / 'M38_G4'
            command('M38_FULL', [PYTHON, str(PKG / 'server_meander.py'), '38', '--groups', '4',
                '--threads', '16', '--reserve-gib', '16', '--timeout-hours', '168', '--resume',
                '--output', str(m38)], env=env)
            result = json.loads((m38 / 'RESULT.json').read_text())
            catalan = math.comb(76, 38) // 39
            if result.get('status') != 'PASS' or result.get('n') != 38 or result.get('groups_completed') != 4:
                raise ArithmeticError('Incomplete M38 result')
            if result.get('source_digest') != SOURCE_DIGEST or int(result.get('weighted_lower_coverage', 0)) != catalan:
                raise ArithmeticError('M38 source or Catalan coverage mismatch')
            if result.get('total_stages') != 272 or result.get('matrix_read_bytes') or result.get('matrix_write_bytes'):
                raise ArithmeticError('M38 stages or matrix I/O mismatch')
            save('M38_FINAL_VALIDATION.json', dict(status='PASS', n=38, Mn=result['Mn'],
                source_digest=SOURCE_DIGEST, groups=4, stages=272,
                sampled_peak_RSS_bytes=result['process_RSS_sampled_peak_bytes'],
                controller_session_seconds=result['controller_session_seconds'],
                matrix_read_bytes=0, matrix_write_bytes=0, independent_CRT=False))
            status('COMPLETE', n=38, Mn=result['Mn'])
        except BaseException as exc:
            save('CONTROLLER_ERROR.json', dict(error=repr(exc), traceback=traceback.format_exc()))
            status('ERROR', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
