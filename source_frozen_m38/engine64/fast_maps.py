"""Exact piecewise-affine ballot closing maps, written once into one arena."""
import functools,time
import numpy as np
from numba import njit,prange
import dense_geometry as G

@njit(cache=True,nogil=True,parallel=True)
def fill(ms,hs,starts,lengths,maxB,tab,total):
 out=np.empty(total,np.int64)
 for k in prange(maxB):out[k]=k
 for j in prange(len(ms)):
  m=ms[j];target=hs[j];base=starts[j]
  stack=np.empty((m+2,4),np.int64);stack[0,0]=m;stack[0,1]=target;stack[0,2]=0;stack[0,3]=0;depth=1
  while depth:
   depth-=1;mm=stack[depth,0];h=stack[depth,1];ib=stack[depth,2];ob=stack[depth,3]
   up=tab[mm-1,h-1];down=tab[mm-1,h+1];shift=tab[mm-1,h-3] if h>=3 else 0
   if down:
    stack[depth,0]=mm-1;stack[depth,1]=h+1;stack[depth,2]=ib+up;stack[depth,3]=ob+shift;depth+=1
   if up:
    if h==target:
     for k in range(up):out[base+ib+k]=ob+shift+k
    else:
     stack[depth,0]=mm-1;stack[depth,1]=h-1;stack[depth,2]=ib;stack[depth,3]=ob;depth+=1
 return out

@functools.lru_cache(maxsize=8)
def maps(n):
 offsets=np.zeros((n+1,n+2),np.int64);lens=np.zeros_like(offsets);mult=np.zeros_like(offsets)
 maxB=max(G.B(m,h) for m in range(n+1) for h in range(m+1));ms=[];hs=[];starts=[];lengths=[];at=maxB
 for m in range(1,n):
  for h in range(2,m+1):
   count=G.B(m,h)
   if not count:continue
   offsets[m,h]=at;lens[m,h]=count;mult[m,h]=(m-h)//2+1
   ms.append(m);hs.append(h);starts.append(at);lengths.append(count);at+=count
 data=fill(np.asarray(ms,np.int64),np.asarray(hs,np.int64),np.asarray(starts,np.int64),np.asarray(lengths,np.int64),maxB,G.TAB,at)
 return offsets,lens,mult,data

@njit(cache=True,nogil=True)
def value_at(m,target,rank,tab):
 if target<2 or target>m or rank<0 or rank>=tab[m,target]:raise ValueError('Closing rank domain')
 h=target;mm=m;ob=np.int64(0);k=np.int64(rank)
 while mm>0:
  up=tab[mm-1,h-1];shift=tab[mm-1,h-3] if h>=3 else 0
  if k<up:
   if h==target:return ob+shift+k
   h-=1;mm-=1
  else:k-=up;ob+=shift;h+=1;mm-=1
 raise ArithmeticError('Incomplete closing rank')
