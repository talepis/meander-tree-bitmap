"""Exact colex ballot geometry; no coefficient hash table or full Catalan list."""
import math
from functools import lru_cache
import numpy as np
from numba import njit,prange
@lru_cache(None)
def B(m,h):
 if m<0 or h<0 or h>m or (m-h)&1:return 0
 k=(m-h)//2;return math.comb(m,k)-(math.comb(m,k-1) if k else 0)
TAB=np.asarray([[B(m,h) for h in range(54)] for m in range(53)],np.int64)
@njit(inline='always')
def unrank(m,h,rank,table):
 word=np.uint64(0)
 for pos in range(m-1,-1,-1):
  up=table[pos,h-1] if h else 0
  if rank<up:word|=np.uint64(1)<<np.uint64(pos);h-=1
  else:rank-=up;h+=1
 return word
@njit(cache=True)
def build_close(m,h,table):
 out=np.empty(table[m,h],np.uint32);stack=np.empty(m,np.int32)
 for rank in range(len(out)):
  word=unrank(m,h,rank,table);depth=0
  for j in range(m):
   if word>>np.uint64(j)&np.uint64(1):stack[depth]=j;depth+=1
   else:depth-=1
  word^=np.uint64(1)<<np.uint64(stack[depth-1]);level=h-2;result=0
  for mm in range(m,0,-1):
   if word>>np.uint64(mm-1)&np.uint64(1):level-=1
   else:result+=table[mm-1,level-1] if level else 0;level+=1
  out[rank]=result
 return out
@lru_cache(maxsize=256)
def close_map(m,h):
 out=build_close(m,h,TAB);mult=int(np.bincount(out.astype(np.int64)).max(initial=0));out.flags.writeable=False;return out,mult
@njit(cache=True)
def geometry(a,b,h,lookup,table):
 out=np.empty((table[a,h],table[b,h]),np.uint16)
 for p in range(out.shape[0]):
  P=unrank(a,h,p,table)
  for q in range(out.shape[1]):
   Q=unrank(b,h,q,table);rev=np.uint64(0)
   for j in range(b):rev|=(np.uint64(1)-((Q>>np.uint64(j))&np.uint64(1)))<<np.uint64(b-1-j)
   key=(np.uint64(1)<<np.uint64(a+b))|P|(rev<<np.uint64(a));out[p,q]=lookup[key]
 return out
@njit(cache=True,nogil=True)
def add(T,V,r0,c0,w):
 for i in range(V.shape[0]):
  for j in range(V.shape[1]):T[r0+i,c0+j]+=np.uint64(V[i,j])*np.uint64(w)
@njit(cache=True,nogil=True,parallel=True)
def add_parallel(T,V,r0,c0,w):
 for i in prange(V.shape[0]):
  for j in range(V.shape[1]):T[r0+i,c0+j]+=np.uint64(V[i,j])*np.uint64(w)
@njit(cache=True,nogil=True)
def columns(T,V,maps,w):
 for i in range(V.shape[0]):
  for j in range(V.shape[1]):T[i,maps[j]]+=np.uint64(V[i,j])*np.uint64(w)
@njit(cache=True,nogil=True,parallel=True)
def columns_parallel(T,V,maps,w):
 for i in prange(V.shape[0]):
  for j in range(V.shape[1]):T[i,maps[j]]+=np.uint64(V[i,j])*np.uint64(w)
@njit(cache=True,nogil=True)
def rows(T,V,maps,w):
 for j in range(V.shape[1]):
  for i in range(V.shape[0]):T[maps[i],j]+=np.uint64(V[i,j])*np.uint64(w)
@njit(cache=True,nogil=True,parallel=True)
def rows_parallel(T,V,maps,w):
 for j in prange(V.shape[1]):
  for i in range(V.shape[0]):T[maps[i],j]+=np.uint64(V[i,j])*np.uint64(w)
@njit(cache=True,nogil=True)
def dot96(A,indices,weights,blo,bhi):
 from_dummy=0
 # Imported primitives are inlined, retaining the proved range check.
 lo=hi=np.uint64(0)
 for i in range(A.shape[0]):
  for j in range(A.shape[1]):
   factor=np.uint64(weights[indices[i,j]]);x=np.uint64(A[i,j]);p0=(x&np.uint64(4294967295))*factor;p1=(x>>np.uint64(32))*factor+(p0>>np.uint64(32));vl=(p0&np.uint64(4294967295))|((p1&np.uint64(4294967295))<<np.uint64(32));vh=p1>>np.uint64(32);nl=lo+vl;nh=hi+vh+np.uint64(nl<lo)
   if nh>bhi or (nh==bhi and nl>blo):raise OverflowError('Dense terminal dot bound')
   lo=nl;hi=nh
 return lo,hi
