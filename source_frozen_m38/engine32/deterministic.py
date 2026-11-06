"""Deterministic lower parser: decide each branch size at its closing return.
No guessed future size, no maximum-multiplicity coordinate. Completed pointed
cycle weights are inserted exactly once; unrestricted suffixes are merged.
"""
import os
os.environ.setdefault('NUMBA_NUM_THREADS','24');os.environ.setdefault('NUMBA_THREADING_LAYER','omp');os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE');os.environ.setdefault('KMP_BLOCKTIME','0')
import math,time
from functools import lru_cache
import numpy as np,psutil
from weighted import Operators
class Compiler:
 def __init__(self,n,ts,ops,share_free=True):self.n=n;self.ts=frozenset(ts);self.max_t=max(ts);self.ops=ops;self.share_free=share_free
 def normalize(self,state):
  i,mode,t,ell,b,active=state;weight=1;remaining=2*self.n-i
  if mode==1 and self.share_free and ((active and ell+remaining<2*t) or (not active and ell+remaining<=2*t)):
   if active:weight=self.n
   mode=2;t=ell=active=0
  return (i,mode,t,ell,b,active),weight
 @lru_cache(maxsize=131072)
 def choices(self,state):
  i,mode,t,ell,b,active=state;r=2*self.n-i
  if not r:return ()
  out=[]
  for opened in (1,0):
   if opened and b>=r:continue
   if not opened and not b:continue
   at=i+1;nb=b+(1 if opened else -1);remaining=2*self.n-at
   if nb>remaining:continue
   weight=1
   if mode==0:
    if nb:
     if at+nb>2*self.max_t:continue
     after=(at,0,0,0,nb,0)
    else:
     size=at//2
     if size not in self.ts:continue
     after,m=self.normalize((at,1,size,0,0,1));weight=m*(1 if 2*size==self.n+1 else 2)
   elif mode==1:
    length=ell+1;flag=active
    if length+nb>2*t:continue
    if not nb:
     if active and length==2*t:weight=(at-length)//2;flag=0
     length=0
    after,m=self.normalize((at,1,t,length,nb,flag));weight*=m
   else:after=(at,2,0,0,nb,0)
   width=2 if self.n==1 else 2*(2*self.n//3)
   if min(at,remaining)+nb>width:raise ArithmeticError('Deterministic centroid width invariant')
   out.append((after,opened,weight))
  return tuple(out)
 @lru_cache(maxsize=131072)
 def lower_count(self,state):
  if state[0]==2*self.n:return int(state[4]==0 and state[1]!=0)
  return sum(weight*self.lower_count(after) for after,opened,weight in self.choices(state))
 @lru_cache(maxsize=65536)
 def language(self,state,r):
  if not r:
   if state[4] or state[0]!=2*self.n or state[1]==0:raise ArithmeticError('Incomplete deterministic lower path')
   return ((0,1),)
  result={}
  for after,label,weight in self.choices(state):
   for suffix,mult in self.language(after,r-1):
    word=(suffix<<1)|label
    if word in result:raise ArithmeticError('Deterministic grammar emitted duplicate word')
    result[word]=weight*mult
  return tuple(sorted(result.items()))
 def compile(self):
  n=self.n;cut=2*n-self.ops.R;initial=(0,0,0,0,0,0);nodes=[initial];ids={initial:0};heights=[0];opids=[-1];offsets=[0];targets=[];labels=[];weights=[];levels=[[0]]
  for step in range(cut+1):
   following=[]
   for g in levels[step]:
    if step==cut:opids[g]=self.ops.register(self.language(nodes[g],self.ops.R));offsets.append(len(targets));continue
    for after,label,weight in self.choices(nodes[g]):
     if after not in ids:
      if len(nodes)>=200000:raise MemoryError('Deterministic metadata cap')
      ids[after]=len(nodes);nodes.append(after);heights.append(after[4]);opids.append(-1);following.append(len(nodes)-1)
     targets.append(ids[after]);labels.append(label);weights.append(weight)
    offsets.append(len(targets))
   if step<cut:levels.append(following)
  width=2 if n==1 else 2*(2*n//3)
  if len(nodes)>=(1<<(64-width-1)):raise OverflowError('Deterministic joint key width')
  return dict(nodes=len(nodes),edges=len(targets),offsets=np.asarray(offsets,np.uint32),targets=np.asarray(targets,np.uint32),labels=np.asarray(labels,np.uint8),weights=np.asarray(weights,np.uint16),heights=np.asarray(heights,np.uint8),opids=np.asarray(opids,np.int32))
def calculate(n,workers=6,group_size=32,tail=8,memory_mib=256,timeout=120,share_free=True):
 if type(n)!=int or not 1<=n<=25 or type(workers)!=int or not 1<=workers<=24 or type(group_size)!=int or not 1<=group_size<=32 or type(tail)!=int or not 1<=tail<=10:raise ValueError('Audited deterministic domain')
 if not math.isfinite(memory_mib) or memory_mib<1 or not math.isfinite(timeout) or timeout<=0:raise ValueError('Invalid resource limits')
 if psutil.virtual_memory().available<workers*int(memory_mib*2**20)+(256<<20)+(3<<30):raise MemoryError('Fresh RAM guard')
 start=time.perf_counter();ops=Operators(min(tail,2*n));ts=list(range(1,(n+1)//2+1));pending=[];groups=[];prep=0.;coverage=0
 for j in range(0,len(ts),group_size):
  if time.perf_counter()-start>timeout:raise TimeoutError('Deterministic preparation deadline')
  t=time.perf_counter();group=ts[j:j+group_size];compiler=Compiler(n,group,ops,share_free);coverage+=compiler.lower_count((0,0,0,0,0,0));g=compiler.compile();prep+=time.perf_counter()-t;pending.append(g);groups.append(group)
 C=math.comb(2*n,n)//(n+1)
 if coverage!=C:raise ArithmeticError('Pointed lower coverage failed')
 from batch96 import run_batch
 profiles=[];values,stats,numeric,metadata=run_batch(n,pending,ops,workers,memory_mib,bound_override=2*n*C*C,profile=profiles)
 if time.perf_counter()-start>timeout:raise TimeoutError('Deterministic completed past acceptance deadline')
 return dict(status='PASS',variant='DETERMINISTIC_POINTED_CENTROID',n=n,Mn=str(sum(values)),workers=workers,group_size=group_size,share_free=share_free,tail=tail,seconds=time.perf_counter()-start,prep_seconds=prep,numeric_seconds=numeric,visits=int(stats[:,1].sum()),peak_states=int(stats[:,2].max()),workspace_peak_upper_bytes=int(stats[:,3].sum()),packed_metadata_bytes=metadata,grammar_nodes=sum(g['nodes'] for g in pending),jobs=len(pending),weighted_lower_coverage=str(coverage),job_profiles=[dict(maximum_branch_sizes=t,weighted_contribution=str(v),visits=s[0],peak_states=s[1],workspace_bytes=s[2],worker=s[3]) for t,v,s in zip(groups,values,profiles)],matrix_read_bytes=0,matrix_write_bytes=0,independent_CRT=False)
