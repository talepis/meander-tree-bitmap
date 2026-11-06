"""Discard only dead workspace pages, retaining the same virtual allocations.

Destructive: callers prove each discarded byte is rewritten before its next read.
Coefficients and immutable maps are excluded. Only complete interior pages of
writable numeric ndarrays are touched. No files or swap requests are created.
"""
import ctypes
import mmap
import os
import sys
import time

POLICY=os.environ.get('MEANDER_MEMORY_POLICY','full' if sys.platform.startswith('linux') else 'eager')
MIN_BYTES=16<<20
if sys.platform.startswith('linux'):
    _lib=ctypes.CDLL(None,use_errno=True)
    _discard=_lib.madvise
    _discard.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]
    _discard.restype=ctypes.c_int
    BACKEND='linux_madvise_DONTNEED'
elif os.name=='nt':
    _lib=ctypes.WinDLL('kernel32',use_last_error=True)
    _discard=_lib.DiscardVirtualMemory
    _discard.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    _discard.restype=ctypes.c_uint32
    BACKEND='windows_DiscardVirtualMemory'
else:
    _discard=None
    BACKEND='unsupported'


def configure(policy='full',min_bytes=16<<20):
    global POLICY,MIN_BYTES
    if policy not in ('baseline','eager','trim','scratch','full'):
        raise ValueError('Unknown memory policy')
    if type(min_bytes)!=int or min_bytes<0:
        raise ValueError('Invalid page-release threshold')
    if policy!='baseline' and _discard is None:
        raise RuntimeError('Dead-page release requires a supported Linux/Windows backend')
    POLICY=policy;MIN_BYTES=min_bytes


def active():return POLICY!='baseline'
def trims_indices():return POLICY in ('trim','full')


def aligned_range(address,total_bytes,begin,end,page_size=mmap.PAGESIZE):
    if not 0<=begin<=end<=total_bytes:
        raise ValueError('Discard byte range is outside the allocation')
    if address<0 or total_bytes<0 or address+total_bytes>1<<64:
        raise OverflowError('Discard address domain')
    lo=((address+begin+page_size-1)//page_size)*page_size
    hi=((address+end)//page_size)*page_size
    return lo,max(0,hi-lo)


def discard_dead(array,begin=0,end=None):
    if not array.flags.c_contiguous or not array.flags.writeable or array.dtype.hasobject:
        raise ValueError('Discard requires a writable contiguous numeric ndarray')
    end=array.nbytes if end is None else int(end)
    address,size=aligned_range(int(array.ctypes.data),int(array.nbytes),int(begin),end)
    if size<MIN_BYTES or not size:return 0
    if _discard is None:raise RuntimeError('Page release unsupported')
    if sys.platform.startswith('linux'):
        result=_discard(ctypes.c_void_p(address),ctypes.c_size_t(size),4)
        error=ctypes.get_errno() if result else 0
    else:
        error=int(_discard(ctypes.c_void_p(address),ctypes.c_size_t(size)))
    if error:raise OSError(error,'Failed to discard dead workspace pages')
    return size


def after_geometry(scratch,current_bank,cache_bytes,work_bytes,index_pool,index_bytes,bounds,index_live,reuse):
    start=time.perf_counter();requested=0;calls=0
    if POLICY in ('scratch','full'):
        # Retain prefixes at the current scale. Only shrink dead tails: growing
        # stages keep their working pages and avoid repeated page faults.
        bounds[current_bank]=max(bounds[current_bank],int(cache_bytes))
        bounds[2]=max(bounds[2],int(work_bytes))
        for k,begin in ((0,int(cache_bytes)),(1,int(cache_bytes)),(2,int(work_bytes))):
            end=max(begin,bounds[k])
            n=discard_dead(scratch[k],begin,end);requested+=n;calls+=bool(n)
            if n:bounds[k]=min(scratch[k].nbytes,begin+2*mmap.PAGESIZE)
    if trims_indices():
        index_live=max(int(index_bytes),index_live) if reuse else int(index_bytes)
        n=discard_dead(index_pool,int(index_bytes),index_live);requested+=n;calls+=bool(n)
        if n:index_live=min(index_pool.nbytes,int(index_bytes)+2*mmap.PAGESIZE)
    return dict(memory_policy=POLICY,page_release_backend=BACKEND,
        dead_page_advice_bytes=requested,page_release_calls=calls,
        page_release_seconds=time.perf_counter()-start),bounds,index_live
