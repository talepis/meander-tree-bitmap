import os
os.environ.update(NUMBA_NUM_THREADS='24',NUMBA_THREADING_LAYER='omp',OMP_WAIT_POLICY='PASSIVE',OPENBLAS_NUM_THREADS='1')
import sys,json,time,hashlib,random,gc
from pathlib import Path
import numpy as np
from numba import njit
H=Path(__file__).resolve().parent;R=H.parents[1];O=R/'outputs/TREE_GEOMETRY_FACTOR_20261104'
sys.path.insert(0,str(H/'candidate'))
from linear_support import calculate
import target_geometry as G
import bitmap_geometry as B

@njit(cache=True)
def used_indices(meta):
    used=0
    for m in meta:
        if m[6]:
            if not m[11]&1:used=max(used,m[9]+m[3]-m[2])
            if not m[11]&2:used=max(used,m[10]+m[5]-m[4])
    return used

@njit(cache=True)
def compare_geometry(oldmeta,newmeta,oldix,newix,oldR,newR,oldC,newC):
    if oldmeta.shape!=newmeta.shape:raise AssertionError('Metadata shape')
    count=0;support_count=0
    for i in range(len(oldmeta)):
        a=oldmeta[i];b=newmeta[i]
        for k in range(14):
            if k not in (9,10) and a[k]!=b[k]:raise AssertionError('Metadata semantics')
        if a[6]:
            for axis in range(2):
                if a[11]&(1<<axis):
                    if a[9+axis]!=b[9+axis]:raise AssertionError('Immutable map offset')
                else:
                    length=a[3]-a[2] if axis==0 else a[5]-a[4]
                    for k in range(length):
                        if oldix[a[9+axis]+k]!=newix[b[9+axis]+k]:raise AssertionError('Translated index')
                    count+=length
    for d in range(len(oldR)):
        if not np.array_equal(oldR[d],newR[d]) or not np.array_equal(oldC[d],newC[d]):raise AssertionError('Axis support')
        support_count+=len(oldR[d])+len(oldC[d])
    return count,support_count

