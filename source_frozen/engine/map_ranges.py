"""Exact min/max of a ballot closing map on a rank interval, without listing it."""
import numpy as np
from numba import njit

@njit(cache=True,nogil=True)
def tables(n,tab):
    low=np.full((n+1,n+2,n+2),9223372036854775807,np.int64)
    high=np.full((n+1,n+2,n+2),-1,np.int64)
    for m in range(2,n+1):
        for h in range(2,m+1):
            if not tab[m,h]:continue
            up=tab[m-1,h-1];down=tab[m-1,h+1];shift=tab[m-1,h-3] if h>=3 else 0
            for target in range(2,h+1):
                lo=np.int64(9223372036854775807);hi=np.int64(-1)
                if up:
                    if h==target:lo=shift;hi=shift+up-1
                    else:lo=low[m-1,h-1,target];hi=high[m-1,h-1,target]
                if down:
                    lo=min(lo,shift+low[m-1,h+1,target]);hi=max(hi,shift+high[m-1,h+1,target])
                low[m,h,target]=lo;high[m,h,target]=hi
    return low,high

@njit(cache=True,nogil=True)
def query(m,target,left,right,tab,low,high):
    if left<0 or right>tab[m,target] or left>=right:raise ArithmeticError('Closing-map interval domain')
    stack=np.empty((m+2,4),np.int64);stack[0]=np.asarray([m,target,0,0],np.int64);depth=1
    result_low=np.int64(9223372036854775807);result_high=np.int64(-1)
    while depth:
        depth-=1;mm=stack[depth,0];h=stack[depth,1];ib=stack[depth,2];ob=stack[depth,3];count=tab[mm,h]
        if right<=ib or left>=ib+count:continue
        if left<=ib and ib+count<=right:
            result_low=min(result_low,ob+low[mm,h,target]);result_high=max(result_high,ob+high[mm,h,target]);continue
        up=tab[mm-1,h-1];down=tab[mm-1,h+1];shift=tab[mm-1,h-3] if h>=3 else 0
        if up and left<ib+up:
            if h==target:
                l=max(left,ib);r=min(right,ib+up)
                if l<r:result_low=min(result_low,ob+shift+l-ib);result_high=max(result_high,ob+shift+r-ib-1)
            else:
                stack[depth,0]=mm-1;stack[depth,1]=h-1;stack[depth,2]=ib;stack[depth,3]=ob;depth+=1
        if down and right>ib+up:
            stack[depth,0]=mm-1;stack[depth,1]=h+1;stack[depth,2]=ib+up;stack[depth,3]=ob+shift;depth+=1
    if result_low>result_high:raise ArithmeticError('Empty closing-map interval image')
    return result_low,result_high

@njit(cache=True,nogil=True)
def axis_bounds(terms,reps,class_terms,dimensions,identities,skeys,heights,rs,cs,tab,low,high):
    tlo=np.empty(len(terms),np.int64);thi=tlo.copy()
    for tid in range(len(terms)):
        t=terms[tid];axis,si=reps[tid];a=rs[si] if axis==0 else cs[si]
        if not t[3]:tlo[tid]=np.int64(a[t[1]])+t[4];thi[tid]=np.int64(a[t[2]-1])+t[4]
        else:
            m=skeys[si,1]-1 if axis==0 else np.int64(heights[skeys[si,0]])-1;target=skeys[si,2]+1
            tlo[tid],thi[tid]=query(m,target,np.int64(a[t[1]])-t[4],np.int64(a[t[2]-1])+1-t[4],tab,low,high)
    bases=np.zeros(len(dimensions),np.int64);spans=dimensions.copy()
    for c in range(len(dimensions)):
        if identities[c]:continue
        lo=np.int64(9223372036854775807);hi=np.int64(-1)
        for tid in class_terms[c]:lo=min(lo,tlo[tid]);hi=max(hi,thi[tid])
        if lo<0 or hi>=dimensions[c]:raise ArithmeticError('Closing-map support bound')
        bases[c]=lo;spans[c]=hi-lo+1
    return bases,spans
