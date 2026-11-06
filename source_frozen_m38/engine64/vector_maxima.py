"""Exact block maxima; typed contiguous reductions for narrow integers."""
import numpy as np
from numba import njit,prange

@njit(cache=True,nogil=True,parallel=True)
def maxima(data,offsets,widths,shape):
    low=np.zeros(len(widths),np.uint64);high=low.copy()
    for j0 in prange(len(widths)):
        j=np.int64(j0);at=offsets[j];count=shape[j,0]*shape[j,1];w=widths[j]
        if w==1:low[j]=np.uint64(data[at:at+count].max())
        elif w==2:low[j]=np.uint64(data.view(np.uint16)[at//2:at//2+count].max())
        elif w==4:low[j]=np.uint64(data.view(np.uint32)[at//4:at//4+count].max())
        elif w==8:low[j]=data.view(np.uint64)[at//8:at//8+count].max()
        else:
            values=data.view(np.uint64);lo=hi=np.uint64(0)
            for k in range(count):
                vl=values[at//8+2*k];vh=values[at//8+2*k+1]
                if vh>hi or (vh==hi and vl>lo):lo=vl;hi=vh
            low[j]=lo;high[j]=hi
    return low,high
