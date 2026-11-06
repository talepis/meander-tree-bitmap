"""Exact affine runs built directly, without first allocating the full maps."""
import functools,math
import numpy as np
from numba import njit
import dense_geometry as G

@njit(cache=True,nogil=True)
def build(n,tab,maxB,capacity):
    off=np.zeros((n+1,n+2),np.int64);lens=off.copy();mult=off.copy()
    starts=np.empty(capacity+1,np.int64);shifts=np.empty(capacity,np.int64)
    starts[0]=0;shifts[0]=0;used=1;at=maxB;last_end=maxB
    for m in range(1,n):
        for target in range(2,m+1):
            count=tab[m,target]
            if not count:continue
            off[m,target]=at;lens[m,target]=count;mult[m,target]=(m-target)//2+1
            stack=np.empty((m+2,4),np.int64);stack[0,:]=[m,target,0,0];depth=1
            while depth:
                depth-=1;mm=stack[depth,0];h=stack[depth,1];ib=stack[depth,2];ob=stack[depth,3]
                up=tab[mm-1,h-1];down=tab[mm-1,h+1];shift=tab[mm-1,h-3] if h>=3 else 0
                if down:
                    stack[depth,:]=[mm-1,h+1,ib+up,ob+shift];depth+=1
                if up:
                    if h==target:
                        begin=at+ib;delta=ob+shift-begin
                        if begin!=last_end:raise ArithmeticError('Affine run coverage')
                        if delta!=shifts[used-1]:
                            if used>=capacity:raise ArithmeticError('Affine run capacity proof')
                            starts[used]=begin;shifts[used]=delta;used+=1
                        last_end=begin+up
                    else:
                        stack[depth,:]=[mm-1,h-1,ib,ob];depth+=1
            at+=count
    if last_end!=at:raise ArithmeticError('Affine map total')
    starts[used]=at
    return off,lens,mult,starts[:used+1],shifts[:used]

@functools.lru_cache(maxsize=8)
def maps(n):
    maxB=max(G.B(m,h) for m in range(n+1) for h in range(m+1));capacity=1
    cats=[math.comb(2*k,k)//(k+1) for k in range(n//2+1)]
    for m in range(1,n):
        for h in range(2,m+1):
            if G.B(m,h):capacity+=sum(cats[:(m-h)//2+1])
    off,lens,mult,starts,shifts=build(n,G.TAB,maxB,capacity)
    return off,lens,mult,np.arange(maxB,dtype=np.int32),starts,shifts

@njit(inline='always')
def lookup(identity,starts,shifts,at):
    if at<len(identity):return np.int64(at)
    lo=0;hi=len(shifts)
    while lo+1<hi:
        mid=(lo+hi)//2
        if starts[mid]<=at:lo=mid
        else:hi=mid
    return at+shifts[lo]

@njit(cache=True,nogil=True)
def decode(identity,starts,shifts,at,count):
    if at+count<=len(identity):return identity[at:at+count]
    lo=0;hi=len(shifts)
    while lo+1<hi:
        mid=(lo+hi)//2
        if starts[mid]<=at:lo=mid
        else:hi=mid
    out=np.empty(count,np.int32);stop=at+count;k=0
    while at<stop:
        end=min(stop,starts[lo+1]);delta=shifts[lo]
        for val in range(at,end):out[k]=val+delta;k+=1
        at=end;lo+=1
    return out
