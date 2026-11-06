"""Compiled exact support union and translation. Hashing is per block, not per coefficient."""
import numpy as np
from numba import njit,types,prange
from numba.typed import Dict,List
from wide_kernels import mul_small
U=np.uint64
ARR=types.Array(types.uint32,1,'C')
MARR=types.Array(types.int64,2,'C')
@njit(inline='always')
def density_le(R,C,lbR,lbC,percent):
 if percent>4294967295:raise OverflowError('Density multiplier domain')
 lo,hi=mul_small(U(R)*U(C),U(0),U(100))
 ro,rh=mul_small(U(lbR)*U(lbC),U(0),U(percent))
 return hi<rh or (hi==rh and lo<=ro)

@njit(inline='always')
def lower(ids,value):
 lo=0;hi=len(ids)
 while lo<hi:
  m=(lo+hi)//2
  if ids[m]<value:lo=m+1
  else:hi=m
 return lo
@njit(inline='always')
def push(di,aa,hh,kind,r0,r1,c0,c1,offr,offc,mapm,maph,w,si,remaining,heights,tab,mapoff,maplen,mapmult,rs,cs,ids,keys,heads,descs):
 if aa<0 or aa>remaining-1 or hh<0 or hh>aa or hh>heights[di] or r0>=r1 or c0>=c1:return
 if not tab[aa,hh] or not tab[heights[di],hh]:return
 ri=rs[si];ci=cs[si];lr=lower(ri,r0);hr=lower(ri,r1);lc=lower(ci,c0);hc=lower(ci,c1)
 if lr==hr or lc==hc:return
 code=(U(di)<<U(12))|(U(aa)<<U(6))|U(hh)
 if code in ids:dest=ids[code]
 else:dest=len(keys);ids[code]=dest;keys.append((di,aa,hh));heads.append(np.int64(-1))
 start=0;mult=1
 if kind:
  start=mapoff[mapm,maph];mult=mapmult[mapm,maph]
  if not maplen[mapm,maph]:raise ArithmeticError('Missing compressed closing map')
 descs.append((si,dest,lr,hr,lc,hc,kind,offr-r0,offc-c0,r0,c0,start,w,mult,heads[dest]));heads[dest]=len(descs)-1

@njit(cache=True,nogil=True)
def discover(n,step,skeys,mx,rs,cs,offsets,targets,labels,weights,heights,tab,mapoff,maplen,mapmult):
 ids=Dict.empty(types.uint64,types.int64);z=np.int64(0);keys=[(z,z,z)];keys.pop();heads=[z];heads.pop();descs=[(z,z,z,z,z,z,z,z,z,z,z,z,z,z,z)];descs.pop();remaining=2*n-step
 for si in range(len(skeys)):
  if mx[si]==0:continue
  gid=skeys[si,0];a=skeys[si,1];h=skeys[si,2];b=np.int64(heights[gid]);Na=tab[a,h];Nb=tab[b,h];Up=tab[a-1,h-1] if a and h else 0;Uq=tab[b-1,h-1] if b and h else 0
  for e in range(offsets[gid],offsets[gid+1]):
   d=np.int64(targets[e]);w=np.int64(weights[e])
   if labels[e]:
    if a<remaining:push(d,a+1,h+1,0,0,Na,0,Nb,0,0,0,0,w,si,remaining,heights,tab,mapoff,maplen,mapmult,rs,cs,ids,keys,heads,descs)
    if a:
     off=tab[b,h-2] if h>=2 else 0;push(d,a-1,h-1,0,0,Up,0,Nb,0,off,0,0,w,si,remaining,heights,tab,mapoff,maplen,mapmult,rs,cs,ids,keys,heads,descs)
     push(d,a-1,h+1,0,Up,Na,0,Nb,0,0,0,0,w,si,remaining,heights,tab,mapoff,maplen,mapmult,rs,cs,ids,keys,heads,descs)
   else:
    if a<remaining:
     off=tab[a,h-2] if h>=2 else 0;push(d,a+1,h-1,0,0,Na,0,Uq,off,0,0,0,w,si,remaining,heights,tab,mapoff,maplen,mapmult,rs,cs,ids,keys,heads,descs)
     push(d,a+1,h+1,0,0,Na,Uq,Nb,0,0,0,0,w,si,remaining,heights,tab,mapoff,maplen,mapmult,rs,cs,ids,keys,heads,descs)
    if a:
     push(d,a-1,h+1,0,Up,Na,Uq,Nb,0,0,0,0,w,si,remaining,heights,tab,mapoff,maplen,mapmult,rs,cs,ids,keys,heads,descs)
     if h>=1:
      push(d,a-1,h-1,1,0,Up,Uq,Nb,0,0,b-1,h+1,w,si,remaining,heights,tab,mapoff,maplen,mapmult,rs,cs,ids,keys,heads,descs)
      push(d,a-1,h-1,2,Up,Na,0,Uq,0,0,a-1,h+1,w,si,remaining,heights,tab,mapoff,maplen,mapmult,rs,cs,ids,keys,heads,descs)
 outkeys=np.empty((len(keys),3),np.int64);harray=np.asarray(heads,dtype=np.int64);darray=np.empty((len(descs),15),np.int64)
 for i in range(len(keys)):
  for k in range(3):outkeys[i,k]=keys[i][k]
 for i in range(len(descs)):
  for k in range(15):darray[i,k]=descs[i][k]
 return outkeys,harray,darray

@njit(cache=True,nogil=True)
def prepare_one(di,keys,heads,descs,skeys,rs,cs,heights,tab,mapdata,invR,invC,tagR,tagC,bufR,bufC,epoch,dense_cut):
 count=0;j=heads[di]
 while j!=-1:count+=1;j=descs[j,14]
 metas=np.empty((count,14),np.int64);at=op=0
 nr=nc=0;minR=minC=4294967295;maxR=maxC=0;j=heads[di];da=keys[di][1];dh=keys[di][2];db=np.int64(heights[keys[di][0]]);full=tab[da,dh]*tab[db,dh]<=dense_cut
 if dense_cut<0:
  lbR=lbC=0;vhead=j
  while vhead!=-1:
   vv=descs[vhead];rlen=vv[3]-vv[2];clen=vv[5]-vv[4];mult=vv[13]
   lbR=max(lbR,(rlen+mult-1)//mult if vv[6]==2 else rlen);lbC=max(lbC,(clen+mult-1)//mult if vv[6]==1 else clen);vhead=vv[14]
  full=density_le(tab[da,dh],tab[db,dh],lbR,lbC,-dense_cut)
 if full:
  rr=mapdata[:tab[da,dh]];cc=mapdata[:tab[db,dh]];nr=len(rr);nc=len(cc)
 else:
  while j!=-1:
   v=descs[j];ri=rs[v[0]];ci=cs[v[0]]
   for k in range(v[2],v[3]):
    val=np.int64(mapdata[v[11]+ri[k]-v[9]]) if v[6]==2 else np.int64(ri[k])+v[7]
    if tagR[val]!=epoch:tagR[val]=epoch;bufR[nr]=val;nr+=1;minR=min(minR,val);maxR=max(maxR,val)
   for k in range(v[4],v[5]):
    val=np.int64(mapdata[v[11]+ci[k]-v[10]]) if v[6]==1 else np.int64(ci[k])+v[8]
    if tagC[val]!=epoch:tagC[val]=epoch;bufC[nc]=val;nc+=1;minC=min(minC,val);maxC=max(maxC,val)
   j=v[14]
  if maxR-minR+1<=8*nr:
   rr=np.empty(nr,np.uint32);atR=0
   for val in range(minR,maxR+1):
    if tagR[val]==epoch:rr[atR]=val;atR+=1
   if atR!=nr:raise ArithmeticError('Ordered row support mismatch')
  else:rr=np.sort(bufR[:nr].copy())
  if maxC-minC+1<=8*nc:
   cc=np.empty(nc,np.uint32);atC=0
   for val in range(minC,maxC+1):
    if tagC[val]==epoch:cc[atC]=val;atC+=1
   if atC!=nc:raise ArithmeticError('Ordered column support mismatch')
  else:cc=np.sort(bufC[:nc].copy())
 invready=False
 rlinear=rr[-1]-rr[0]+1==nr;clinear=cc[-1]-cc[0]+1==nc
 if not full:
  if not rlinear:
   for k in range(nr):invR[rr[k]]=k
  if not clinear:
   for k in range(nc):invC[cc[k]]=k
  invready=True
 j=heads[di]
 while j!=-1:
  v=descs[j];ri=rs[v[0]];ci=cs[v[0]];si=v[0];sa=skeys[si,1];sh=skeys[si,2];sb=np.int64(heights[skeys[si,0]]);fullsource=len(ri)==tab[sa,sh] and len(ci)==tab[sb,sh]
  kind=2 if v[6]==2 else 1;rb=cb=rstart=cstart=0;mapmask=0
  if nr==tab[da,dh] and nc==tab[db,dh] and fullsource:
   if v[6]==0:kind=0;rb=v[2]+v[7];cb=v[4]+v[8]
   elif v[6]==1:cstart=v[11];mapmask=3
   else:rstart=v[11];mapmask=3
  else:
   if not invready:
    if not rlinear:
     for k in range(nr):invR[rr[k]]=k
    if not clinear:
     for k in range(nc):invC[cc[k]]=k
    invready=True
   nrV=v[3]-v[2];ncV=v[5]-v[4];rcontig=False;ccontig=False
   if v[6]!=2:
    rfirst=ri[v[2]]+v[7]-rr[0] if rlinear else invR[ri[v[2]]+v[7]];rlast=ri[v[3]-1]+v[7]-rr[0] if rlinear else invR[ri[v[3]-1]+v[7]];rcontig=rlast-rfirst+1==nrV
   if v[6]!=1:
    cfirst=ci[v[4]]+v[8]-cc[0] if clinear else invC[ci[v[4]]+v[8]];clast=ci[v[5]-1]+v[8]-cc[0] if clinear else invC[ci[v[5]-1]+v[8]];ccontig=clast-cfirst+1==ncV
   if v[6]==0 and rcontig and ccontig:kind=0;rb=rfirst;cb=cfirst
   else:
    if rcontig:rstart=rfirst;mapmask|=1
    else:
     rstart=at;at+=nrV
    if ccontig:cstart=cfirst;mapmask|=2
    else:
     cstart=at;at+=ncV
  row=(v[0],di,v[2],v[3],v[4],v[5],kind,rb,cb,rstart,cstart,mapmask,v[12],v[13])
  for k in range(14):metas[op,k]=row[k]
  op+=1;j=v[14]
 flat=np.empty(at,np.uint32);j=heads[di];op=0
 while j!=-1:
  v=descs[j];m=metas[op];ri=rs[v[0]];ci=cs[v[0]]
  if m[6]!=0:
   if not m[11]&1:
    for k in range(v[3]-v[2]):
     val=np.int64(mapdata[v[11]+ri[v[2]+k]-v[9]]) if v[6]==2 else np.int64(ri[v[2]+k])+v[7]
     flat[m[9]+k]=val-rr[0] if rlinear else invR[val]
   if not m[11]&2:
    for k in range(v[5]-v[4]):
     val=np.int64(mapdata[v[11]+ci[v[4]+k]-v[10]]) if v[6]==1 else np.int64(ci[v[4]+k])+v[8]
     flat[m[10]+k]=val-cc[0] if clinear else invC[val]
  op+=1;j=v[14]
 return rr,cc,metas,flat

@njit(cache=True,nogil=True)
def plan(keys,heads,descs,heights,tab,dense_cut):
 dims=np.zeros((len(keys),2),np.int64);cost=np.zeros(len(keys),np.int64)
 for di in range(len(keys)):
  a=keys[di,1];h=keys[di,2];b=heights[keys[di,0]];R=tab[a,h];C=tab[b,h]
  full=R*C<=dense_cut;lbR=lbC=0;j=heads[di];weight=0
  while j!=-1:
   v=descs[j];nr=v[3]-v[2];nc=v[5]-v[4];mult=v[13]
   lbR=max(lbR,(nr+mult-1)//mult if v[6]==2 else nr)
   lbC=max(lbC,(nc+mult-1)//mult if v[6]==1 else nc)
   weight+=nr+nc;j=v[14]
  if dense_cut<0:full=density_le(R,C,lbR,lbC,-dense_cut)
  if not full:dims[di,0]=R;dims[di,1]=C
  cost[di]=weight if not full else weight//8
 return dims,cost

@njit(cache=True,nogil=True,parallel=True)
def merge_packets(keys,packR,packC,packM,packI,index_pool,mapdata,split_maps):
 M=np.empty(len(keys)+1,np.int64);I=np.empty(len(keys)+1,np.int64);M[0]=0;I[0]=0
 for di in range(len(keys)):
  if len(packM[di])>9223372036854775807-M[di] or len(packI[di])>2305843009213693951-I[di]:raise OverflowError('Geometry prefix layout')
  M[di+1]=M[di]+len(packM[di]);I[di+1]=I[di]+len(packI[di])
 totalM=M[-1];totalI=I[-1];base=0 if split_maps else len(mapdata)
 if base>2305843009213693951-totalI:raise OverflowError('Geometry index byte layout')
 metas=np.empty((totalM,14),np.int64);new_pool=len(index_pool)<base+totalI
 flat=np.empty(base+totalI,np.uint32) if new_pool else index_pool
 if not split_maps and new_pool:
  for k in prange(base):flat[k]=mapdata[k]
 shape=np.empty((len(keys),2),np.int64)
 # Every destination owns disjoint metadata and translated-index intervals.
 for target in prange(len(keys)):
  di=np.int64(target)
  shape[di,0]=len(packR[di]);shape[di,1]=len(packC[di]);m=packM[di];at=I[di];op=M[di]
  for j in range(len(m)):
   for k in range(14):metas[op+j,k]=m[j,k]
   if m[j,6]!=0:
    if not m[j,11]&1:metas[op+j,9]+=at+base
    if not m[j,11]&2:metas[op+j,10]+=at+base
   if not split_maps:metas[op+j,11]=0
  piece=packI[di]
  for k in range(len(piece)):flat[base+at+k]=piece[k]
 return keys,shape,metas,flat,packR,packC,totalI

@njit(cache=True,nogil=True)
def run_group(group,keys,heads,descs,skeys,rs,cs,heights,tab,mapdata,invR,invC,tagR,tagC,bufR,bufC,epoch,dense_cut):
 rr=List.empty_list(ARR);cc=List.empty_list(ARR);mm=List.empty_list(MARR);ii=List.empty_list(ARR)
 for di in group:
  epoch+=1
  r,c,m,i=prepare_one(di,keys,heads,descs,skeys,rs,cs,heights,tab,mapdata,invR,invC,tagR,tagC,bufR,bufC,epoch,dense_cut)
  rr.append(r);cc.append(c);mm.append(m);ii.append(i)
 return rr,cc,mm,ii

@njit(cache=True,nogil=True)
def merge_groups(keys,where_worker,where_local,gR,gC,gM,gI,index_pool,mapdata,split_maps):
 rr=List.empty_list(ARR);cc=List.empty_list(ARR);mm=List.empty_list(MARR);ii=List.empty_list(ARR)
 for di in range(len(keys)):
  g=where_worker[di];j=where_local[di]
  rr.append(gR[g][j]);cc.append(gC[g][j]);mm.append(gM[g][j]);ii.append(gI[g][j])
 return merge_packets(keys,rr,cc,mm,ii,index_pool,mapdata,split_maps)

@njit(cache=True,nogil=True)
def assign_groups(dims,cost,workers,capR,capC,pooled):
 # Stable cost ordering and a bounded retry; one group always fits if each target fits.
 order=np.argsort(cost,kind='mergesort')[::-1]
 for active in range(workers,0,-1):
  assignment=np.full(len(cost),-1,np.int32);maxR=np.zeros(active,np.int64);maxC=np.zeros(active,np.int64);loads=np.zeros(active,np.int64)
  usedR=usedC=0;ok=True
  for di in order:
   r=dims[di,0];c=dims[di,1];chosen=-1;best=9223372036854775807
   if r>capR or c>capC:raise MemoryError('Target exceeds original scratch capacity')
   for j in range(active):
    er=max(0,r-maxR[j]);ec=max(0,c-maxC[j])
    if ((usedR+usedC+er+ec<=capR+capC) if pooled else (usedR+er<=capR and usedC+ec<=capC)) and loads[j]<best:chosen=j;best=loads[j]
   if chosen<0:ok=False;break
   usedR+=max(0,r-maxR[chosen]);usedC+=max(0,c-maxC[chosen]);maxR[chosen]=max(maxR[chosen],r);maxC[chosen]=max(maxC[chosen],c)
   assignment[di]=chosen;loads[chosen]+=cost[di]+256
  if ok:return assignment,maxR,maxC,loads
 raise MemoryError('No bounded target schedule')

class Geometry:
 """One discovery, compiled target groups, disjoint original scratch slices.

 Each group reuses its own slices sequentially. The sum of its maximum row
 and column dimensions never exceeds the original shared arenas. There are
 no new full-size tag/inverse/buffer arrays and no shared mutable hash tables.
 """
 def __init__(self,workers,split_maps=True):
  from concurrent.futures import ThreadPoolExecutor
  self.split_maps=bool(split_maps);self.workers=int(workers);self.pool=ThreadPoolExecutor(max_workers=self.workers) if workers>1 else None;self.stats=[];self.scratch=None
 def bind_scratch(self,inv,tags,buf):
  assert len(inv)==len(tags)==len(buf)
  self.scratch=(inv,tags,buf)
 def close(self):
  if self.pool:self.pool.shutdown(wait=True)
 def prepare(self,n,step,skeys,mx,rs,cs,offsets,targets,labels,weights,heights,tab,mapoff,maplen,mapmult,mapdata,invR,invC,tagR,tagC,bufR,bufC,epoch,dense_cut,index_pool):
  import time
  t=time.perf_counter();keys,heads,descs=discover(n,step,skeys,mx,rs,cs,offsets,targets,labels,weights,heights,tab,mapoff,maplen,mapmult)
  discovery=time.perf_counter()-t;t=time.perf_counter();dims,cost=plan(keys,heads,descs,heights,tab,dense_cut)
  capR=len(invR);capC=len(invC)
  assignment,maxR,maxC,loads=assign_groups(dims,cost,self.workers,capR,capC,self.scratch is not None)
  groups=[np.flatnonzero(assignment==j).astype(np.int64) for j in range(len(maxR))]
  usedR=int(np.sum(maxR));usedC=int(np.sum(maxC))
  assert usedR+usedC<=capR+capC if self.scratch is not None else usedR<=capR and usedC<=capC
  if epoch+max(map(len,groups),default=0)+1>=2**32:tagR[:]=0;tagC[:]=0;epoch=0
  jobs=[];ro=co=0;where_worker=np.empty(len(keys),np.int64);where_local=np.empty(len(keys),np.int64)
  for j,group in enumerate(groups):
   if len(group):
    ids=np.asarray(group,np.int64);where_worker[ids]=len(jobs);where_local[ids]=np.arange(len(group),dtype=np.int64)
    jobs.append((ids,ro,maxR[j],co,maxC[j]))
   ro+=maxR[j];co+=maxC[j]
  if self.scratch is not None:
   inv_all,tag_all,buf_all=self.scratch;at=0;newjobs=[]
   for ids,_,r,_,c in jobs:
    newjobs.append((ids,at,r,at+r,c));at+=r+c
   assert at<=len(inv_all)
   jobs=newjobs
  schedule_seconds=time.perf_counter()-t;t=time.perf_counter()
  def work(item):
   ids,ro,r,co,c=item
   wall_start=time.perf_counter();cpu_start=time.thread_time()
   if self.scratch is None:
    views=(invR[ro:ro+r],invC[co:co+c],tagR[ro:ro+r],tagC[co:co+c],bufR[ro:ro+r],bufC[co:co+c])
   else:
    views=(inv_all[ro:ro+r],inv_all[co:co+c],tag_all[ro:ro+r],tag_all[co:co+c],buf_all[ro:ro+r],buf_all[co:co+c])
   result=run_group(ids,keys,heads,descs,skeys,rs,cs,heights,tab,mapdata,*views,epoch,dense_cut)
   return (*result,time.perf_counter()-wall_start,time.thread_time()-cpu_start)
  results=list(self.pool.map(work,jobs)) if self.pool and len(jobs)>1 else [work(item) for item in jobs]
  target_seconds=time.perf_counter()-t;t=time.perf_counter()
  gR=List.empty_list(types.ListType(ARR));gC=List.empty_list(types.ListType(ARR));gM=List.empty_list(types.ListType(types.Array(types.int64,2,'C')));gI=List.empty_list(types.ListType(ARR))
  for result in results:gR.append(result[0]);gC.append(result[1]);gM.append(result[2]);gI.append(result[3])
  import numba
  old_threads=numba.get_num_threads();numba.set_num_threads(min(old_threads,self.workers))
  try:result=merge_groups(keys,where_worker,where_local,gR,gC,gM,gI,index_pool,mapdata,self.split_maps)
  finally:numba.set_num_threads(old_threads)
  self.stats.append(dict(stage=step+1,discovery_seconds=discovery,schedule_seconds=schedule_seconds,target_seconds=target_seconds,merge_seconds=time.perf_counter()-t,blocks=len(keys),descriptors=len(descs),groups=len(jobs),scratch_reserved_peak_bytes=12*(usedR+usedC),scratch_capacity_bytes=12*(capR+capC),scratch_pooled=self.scratch is not None,translated_index_bytes=4*result[-1],group_work_estimates=loads.tolist(),requested_workers=self.workers,target_CPU_seconds=sum(r[5] for r in results),group_wall_seconds=[r[4] for r in results],group_CPU_seconds=[r[5] for r in results]))
  return (*result[:6],epoch+max(map(len,groups),default=0))
