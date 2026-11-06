"""Isolated noncrossing frontier primitives, copied without modifying the originals."""
import numpy as np
from numba import njit
U=np.uint64

@njit(inline="always")
def length(key):
 m=0
 if key>>np.uint64(32):m+=32;key>>=np.uint64(32)
 if key>>np.uint64(16):m+=16;key>>=np.uint64(16)
 if key>>np.uint64(8):m+=8;key>>=np.uint64(8)
 if key>>np.uint64(4):m+=4;key>>=np.uint64(4)
 if key>>np.uint64(2):m+=2;key>>=np.uint64(2)
 if key>>np.uint64(1):m+=1
 return m

@njit(inline="always")
def splice(key,a):
 u=a-1;left=(key>>np.uint64(u))&np.uint64(1);right=(key>>np.uint64(a))&np.uint64(1)
 if left and not right:return np.uint64(0)
 adjusted=key
 if left and right:
  d=1;j=a+1
  while d:
   d+=1 if (key>>np.uint64(j))&np.uint64(1) else -1;j+=1
  adjusted^=np.uint64(1)<<np.uint64(j-1)
 elif not left and not right:
  d=1;j=u-1
  while d:
   d+=-1 if (key>>np.uint64(j))&np.uint64(1) else 1;j-=1
  adjusted^=np.uint64(1)<<np.uint64(j+1)
 return (adjusted&((np.uint64(1)<<np.uint64(u))-np.uint64(1)))|((adjusted>>np.uint64(u+2))<<np.uint64(u))

def prefixes(n,length):
 def rec(i,b,word):
  if i==length:yield word;return
  if b:yield from rec(i+1,b-1,word)
  if b<2*n-i-1:yield from rec(i+1,b+1,word|(1<<i))
 return rec(0,0,0)

@njit(inline="always")
def mirror(key,m):
 if m==0:return U(1)
 x=key&((U(1)<<U(m))-U(1))
 x=((x>>U(1))&U(0x5555555555555555))|((x&U(0x5555555555555555))<<U(1))
 x=((x>>U(2))&U(0x3333333333333333))|((x&U(0x3333333333333333))<<U(2))
 x=((x>>U(4))&U(0x0f0f0f0f0f0f0f0f))|((x&U(0x0f0f0f0f0f0f0f0f))<<U(4))
 x=((x>>U(8))&U(0x00ff00ff00ff00ff))|((x&U(0x00ff00ff00ff00ff))<<U(8))
 x=((x>>U(16))&U(0x0000ffff0000ffff))|((x&U(0x0000ffff0000ffff))<<U(16))
 x=(x>>U(32))|(x<<U(32));return (U(1)<<U(m))|((~(x>>U(64-m)))&((U(1)<<U(m))-U(1)))

@njit(inline="always")
def hash64(x):
 x^=x>>U(30);x*=U(0xbf58476d1ce4e5b9);x^=x>>U(27);x*=U(0x94d049bb133111eb);return x^(x>>U(31))

@njit(inline="always")
def terminal(key,r):
 # Both upper and lower must close at every remaining vertex. At most one completion.
 for k in range(r,0,-1):
  key=splice(key,k)
  if not key:return k==1
 return False
