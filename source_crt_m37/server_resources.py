"""Read-only resource discovery. Never changes cgroups, affinity or system settings."""
import math
import os
import re
from pathlib import Path
import psutil

GiB = 1 << 30
REAL_VM = psutil.virtual_memory


def integer(path):
    try:
        s = Path(path).read_text().strip()
        return None if s == 'max' else int(s)
    except (OSError, ValueError):
        return None


def unescape(s):
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), s)


def cgroup_paths(members=None, mounts=None):
    """Return visible self-to-mount ancestors, including parent constraints."""
    if members is None:
        try:
            members = Path('/proc/self/cgroup').read_text()
            mounts = Path('/proc/self/mountinfo').read_text()
        except OSError:
            return []
    groups = []
    for row in members.splitlines():
        _, controllers, member = row.split(':', 2)
        for line in mounts.splitlines():
            fields = line.split()
            if '-' not in fields:
                continue
            j = fields.index('-'); kind = fields[j + 1]
            if kind == 'cgroup2' and controllers:
                continue
            if kind == 'cgroup' and not (set(controllers.split(',')) & set(fields[j + 3].split(','))):
                continue
            if kind not in ('cgroup', 'cgroup2'):
                continue
            root, mount = unescape(fields[3]), Path(unescape(fields[4]))
            if member == '/':
                relative = ''  # cgroup namespace root
            elif root == '/':
                relative = member.lstrip('/')
            elif member == root:
                relative = ''
            elif member.startswith(root.rstrip('/') + '/'):
                relative = member[len(root):].lstrip('/')
            else:
                continue
            here = mount / relative
            while True:
                groups.append((kind, str(here), controllers))
                if here == mount:
                    break
                if mount not in here.parents:
                    raise RuntimeError('Invalid cgroup ancestry')
                here = here.parent
    return list(dict.fromkeys(groups))


class Resources:
    def __init__(self, memory_gib=None):
        self.paths = cgroup_paths()
        self.memory_bytes = None if memory_gib is None else int(memory_gib * GiB)
        try:
            self.affinity = sorted(os.sched_getaffinity(0))
        except AttributeError:
            try:
                self.affinity = psutil.Process().cpu_affinity()
            except (AttributeError, psutil.Error):
                self.affinity = list(range(os.cpu_count() or 1))
        limits = [len(self.affinity)]
        self.cpu_quotas = []
        for kind, path, ctr in self.paths:
            p = Path(path)
            if kind == 'cgroup2':
                try:
                    quota, period = (p / 'cpu.max').read_text().split()
                    if quota != 'max':
                        self.cpu_quotas.append(int(quota) / int(period))
                except (OSError, ValueError):
                    pass
            elif 'cpu' in ctr.split(','):
                quota = integer(p / 'cpu.cfs_quota_us'); period = integer(p / 'cpu.cfs_period_us')
                if quota and quota > 0 and period and period > 0:
                    self.cpu_quotas.append(quota / period)
        if self.cpu_quotas:
            limits.append(max(1, math.floor(min(self.cpu_quotas))))
        if os.environ.get('SLURM_CPUS_PER_TASK', '').isdigit():
            limits.append(max(1, int(os.environ['SLURM_CPUS_PER_TASK'])))
        self.cpu_limit = min(limits)
        # Physical count is a conservative default, explicit --threads can use
        # any logical CPUs actually allocated by affinity/quota/scheduler.
        physical = psutil.cpu_count(logical=False) or self.cpu_limit
        self.default_threads = min(physical, self.cpu_limit)

    def snapshot(self):
        vm = REAL_VM(); proc = psutil.Process(); mem = proc.memory_info()
        available = [vm.available]; totals = [vm.total]; records = []
        for kind, path, ctr in self.paths:
            p = Path(path)
            if kind == 'cgroup2':
                cap = integer(p / 'memory.max'); used = integer(p / 'memory.current')
            elif 'memory' in ctr.split(','):
                cap = integer(p / 'memory.limit_in_bytes'); used = integer(p / 'memory.usage_in_bytes')
            else:
                continue
            if cap is not None and 0 < cap < (1 << 60):
                if used is None:
                    raise RuntimeError('Cannot read usage for a finite cgroup memory limit')
                available.append(max(0, cap - used)); totals.append(cap)
                records.append(dict(path=path, hard_limit_bytes=cap, current_bytes=used))
        if self.memory_bytes is not None:
            available.append(max(0, self.memory_bytes - mem.rss)); totals.append(self.memory_bytes)
        try:
            import resource
            soft, _ = resource.getrlimit(resource.RLIMIT_AS)
            if soft != resource.RLIM_INFINITY:
                available.append(max(0, soft - mem.vms))
        except ImportError:
            pass
        return dict(host_total_bytes=vm.total, host_available_bytes=vm.available,
                    effective_total_bytes=min(totals), effective_available_bytes=min(available),
                    process_RSS_bytes=mem.rss, process_VMS_bytes=mem.vms,
                    cgroups=records, affinity=self.affinity, cpu_quotas=self.cpu_quotas,
                    cpu_limit=self.cpu_limit, default_threads=self.default_threads,
                    explicit_process_limit_bytes=self.memory_bytes)

    def install_guard(self, reserve):
        # The original engine already adds 3 GiB at its gates. Adjust the
        # fresh available value so that those gates enforce the larger reserve.
        def guarded_vm():
            snap = self.snapshot()
            return REAL_VM()._replace(total=snap['effective_total_bytes'],
                available=max(0, snap['effective_available_bytes'] - max(0, reserve - 3 * GiB)))
        psutil.virtual_memory = guarded_vm
