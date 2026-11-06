"""Small exact universal terminal operator, built from the frontier recurrence."""
import time,math,psutil
from functools import lru_cache
import numpy as np
from numba import njit
from rank_tools import ways_table,unrank
from primitives import length,splice
U=np.uint64
@njit(cache=True)
def build(R):
 sizes=np.zeros(R+1,np.int64);cats=np.zeros(R+1,np.int64);cats[0]=1
 for g in range(1,R+1):cats[g]=cats[g-1]*2*(2*g-1)//(g+1)
 for g in range(R+1):sizes[g]=cats[g]+(sizes[g-1] if g else 0)
 keys=np.empty(sizes[-1],np.uint64);lookup=np.full(1<<(2*R+1),65535,np.uint16);at=0
 for g in range(R+1):
  ways=ways_table(g)
  for rank in range(cats[g]):
   key=(U(1)<<U(2*g))|unrank(g,U(rank),ways);keys[at]=key;lookup[key]=at;at+=1
 previous=np.ones((1,1),np.uint16)
 for r in range(1,R+1):
  current=np.zeros((1<<r,sizes[r]),np.uint16)
  for suffix in range(1<<r):
   ones=0
   for bit in range(r):ones+=(suffix>>bit)&1
   b=r-2*ones
   if b<0:continue
   h=b;valid=True
   for bit in range(r):
    h+=1 if (suffix>>bit)&1 else -1
    if h<0:valid=False;break
   if not valid or h:continue
   opened=bool(suffix&1)
   for j in range(sizes[r]):
    key=keys[j];m=length(key);a=m-b
    if a<0 or a>r:continue
    answer=0
    if a<r:
     dest=(key&((U(1)<<U(a))-U(1)))|(U(1)<<U(a))|((key>>U(a))<<U(a+2)) if opened else key
     ix=int(lookup[dest])
     if ix>=sizes[r-1]:raise ArithmeticError('Invalid tail upper continuation')
     answer+=int(previous[suffix>>1,ix])
    if a:
     dest=key if opened else splice(key,a)
     if dest:
      ix=int(lookup[dest])
      if ix>=sizes[r-1]:raise ArithmeticError('Invalid tail lower continuation')
      answer+=int(previous[suffix>>1,ix])
     elif r==1 and m==2:answer+=1
    if answer>=65536:raise OverflowError('Exact tail uint16 capacity')
    current[suffix,j]=answer
  previous=current
 return lookup,previous,keys,sizes
@lru_cache(maxsize=4)
def get(R):
 if type(R)!=int or not 1<=R<=10:raise ValueError('Audited exact terminal horizon1..10')
 count=sum(math.comb(2*g,g)//(g+1) for g in range(R+1));peak_bound=count*(1<<R)*6+(1<<(2*R+1))*2+count*8
 if peak_bound>psutil.virtual_memory().available-(3<<30):raise MemoryError('Exact terminal constructor RAM guard')
 start=time.perf_counter();lookup,table,keys,sizes=build(R);maximum=int(table.max())
 if maximum<=255:table=table.astype(np.uint8)
 lookup.flags.writeable=False;table.flags.writeable=False
 return lookup,table,dict(status='PASS',horizon=R,seconds=time.perf_counter()-start,key_slots=len(lookup),frontier_keys=len(keys),constant_bytes=lookup.nbytes+table.nbytes,max_completion_count=int(table.max()),candidate_Mn_input=False,encoding=str(table.dtype)+' after checked exhaustive construction')
