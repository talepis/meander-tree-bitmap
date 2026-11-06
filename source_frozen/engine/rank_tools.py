import numpy as np
from numba import njit
U=np.uint64
@njit(cache=True)
def ways_table(n):
 m=2*n;ways=np.zeros((m+1,m+2),np.uint64);ways[0,0]=1
 for r in range(1,m+1):
  for h in range(r+1):
   if h:ways[r,h]+=ways[r-1,h-1]
   ways[r,h]+=ways[r-1,h+1]
 return ways
@njit(cache=True)
def unrank(n,rank,ways):
 word=U(0);h=0
 for i in range(2*n):
  r=2*n-i-1;closed=ways[r,h-1] if h else U(0)
  if rank<closed:h-=1
  else:rank-=closed;word|=U(1)<<U(i);h+=1
 if h or rank:raise ValueError('Invalid Catalan rank')
 return word
