"""Private-array destructive-discard smoke check, before any large job."""
import time,hashlib
import numpy as np,psutil
import memory_release as M
def run_checks():
    policy,minimum=M.POLICY,M.MIN_BYTES
    M.configure('full',0)
    try:
        a=np.full((64<<20)+137,41,np.uint8);lo=(4<<20)+39;hi=a.nbytes-8193
        head=hashlib.sha256(a[:lo].tobytes()).digest();tail=hashlib.sha256(a[hi:].tobytes()).digest()
        before=psutil.Process().memory_info().rss;t=time.perf_counter()
        n=M.discard_dead(a,lo,hi);elapsed=time.perf_counter()-t
        after=psutil.Process().memory_info().rss
        assert hashlib.sha256(a[:lo].tobytes()).digest()==head
        assert hashlib.sha256(a[hi:].tobytes()).digest()==tail
        a[lo:hi]=47
        assert np.all(a[:lo]==41) and np.all(a[lo:hi]==47) and np.all(a[hi:]==41)
        assert n>32<<20
        return dict(status='PASS',backend=M.BACKEND,advice_bytes=n,RSS_drop=max(0,before-after),
            seconds=elapsed,live_prefix_suffix_preserved=True,rewritten_bytes_checked=True)
    finally:M.configure(policy,minimum)
