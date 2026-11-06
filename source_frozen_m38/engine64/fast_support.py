import numpy as np
from numba import njit
EMPTY=np.zeros(1,np.int64);EMPTY.flags.writeable=False
@njit(cache=True)
def collect(ids,lo,hi,delta,origin,mapped,mapping,tags,epoch,buffer,count):
 for i in range(lo,hi):
  value=int(mapping[ids[i]-origin]) if mapped else int(ids[i])+delta
  if tags[value]!=epoch:tags[value]=epoch;buffer[count]=value;count+=1
 return count
@njit(cache=True)
def fill_inverse(ids,index):
 for j in range(len(ids)):index[ids[j]]=j
@njit(cache=True)
def translate(ids,lo,hi,delta,origin,mapped,mapping,index):
 result=np.empty(hi-lo,np.int64)
 for i in range(lo,hi):
  value=int(mapping[ids[i]-origin]) if mapped else int(ids[i])+delta
  result[i-lo]=index[value]
 return result
