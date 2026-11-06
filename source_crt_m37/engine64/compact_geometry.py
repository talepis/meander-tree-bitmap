"""Compiled exact support union and translation. Hashing is per block, not per coefficient."""
import numpy as np
from compressed_maps import lookup
from numba import njit,types
from numba.typed import Dict,List
U=np.uint64
ARR=types.Array(types.int32,1,'C')
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
def prepare_compact(n,step,skeys,mx,rs,cs,offsets,targets,labels,weights,heights,tab,mapoff,maplen,mapmult,mapdata,invR,invC,tagR,tagC,bufR,bufC,epoch,dense_cut,index_pool,mapstarts,mapshifts):
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
 outkeys=np.empty((len(keys),3),np.int64);shape=np.empty((len(keys),2),np.int64);metas=np.empty((len(descs),14),np.int64);pieces=List.empty_list(ARR);outR=List.empty_list(ARR);outC=List.empty_list(ARR);at=0;op=0
 for di in range(len(keys)):
  epoch+=1;nr=nc=0;minR=minC=2147483647;maxR=maxC=0;j=heads[di];da=keys[di][1];dh=keys[di][2];db=np.int64(heights[keys[di][0]]);full=tab[da,dh]*tab[db,dh]<=dense_cut
  if dense_cut<0:
   lbR=lbC=0;vhead=j
   while vhead!=-1:
    vv=descs[vhead];rlen=vv[3]-vv[2];clen=vv[5]-vv[4];mult=vv[13]
    lbR=max(lbR,(rlen+mult-1)//mult if vv[6]==2 else rlen);lbC=max(lbC,(clen+mult-1)//mult if vv[6]==1 else clen);vhead=vv[14]
   full=100*tab[da,dh]*tab[db,dh]<=(-dense_cut)*lbR*lbC
  if full:
   rr=mapdata[:tab[da,dh]];cc=mapdata[:tab[db,dh]];nr=len(rr);nc=len(cc)
  else:
   while j!=-1:
    v=descs[j];ri=rs[v[0]];ci=cs[v[0]]
    for k in range(v[2],v[3]):
     val=np.int64(lookup(mapdata,mapstarts,mapshifts,v[11]+ri[k]-v[9])) if v[6]==2 else np.int64(ri[k])+v[7]
     if tagR[val]!=epoch:tagR[val]=epoch;bufR[nr]=val;nr+=1;minR=min(minR,val);maxR=max(maxR,val)
    for k in range(v[4],v[5]):
     val=np.int64(lookup(mapdata,mapstarts,mapshifts,v[11]+ci[k]-v[10])) if v[6]==1 else np.int64(ci[k])+v[8]
     if tagC[val]!=epoch:tagC[val]=epoch;bufC[nc]=val;nc+=1;minC=min(minC,val);maxC=max(maxC,val)
    j=v[14]
   if maxR-minR+1<=8*nr:
    rr=np.empty(nr,np.int32);atR=0
    for val in range(minR,maxR+1):
     if tagR[val]==epoch:rr[atR]=val;atR+=1
    if atR!=nr:raise ArithmeticError('Ordered row support mismatch')
   else:rr=np.sort(bufR[:nr].copy())
   if maxC-minC+1<=8*nc:
    cc=np.empty(nc,np.int32);atC=0
    for val in range(minC,maxC+1):
     if tagC[val]==epoch:cc[atC]=val;atC+=1
    if atC!=nc:raise ArithmeticError('Ordered column support mismatch')
   else:cc=np.sort(bufC[:nc].copy())
  outR.append(rr);outC.append(cc);invready=False
  rlinear=rr[-1]-rr[0]+1==nr;clinear=cc[-1]-cc[0]+1==nc
  if not full:
   if not rlinear:
    for k in range(nr):invR[rr[k]]=k
   if not clinear:
    for k in range(nc):invC[cc[k]]=k
   invready=True
  shape[di,0]=nr;shape[di,1]=nc
  for k in range(3):outkeys[di,k]=keys[di][k]
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
      dr=np.empty(nrV,np.int32)
      for k in range(nrV):
       val=np.int64(lookup(mapdata,mapstarts,mapshifts,v[11]+ri[v[2]+k]-v[9])) if v[6]==2 else np.int64(ri[v[2]+k])+v[7];dr[k]=val-rr[0] if rlinear else invR[val]
      rstart=at;pieces.append(dr);at+=nrV
     if ccontig:cstart=cfirst;mapmask|=2
     else:
      dc=np.empty(ncV,np.int32)
      for k in range(ncV):
       val=np.int64(lookup(mapdata,mapstarts,mapshifts,v[11]+ci[v[4]+k]-v[10])) if v[6]==1 else np.int64(ci[v[4]+k])+v[8];dc[k]=val-cc[0] if clinear else invC[val]
      cstart=at;pieces.append(dc);at+=ncV
   row=(v[0],di,v[2],v[3],v[4],v[5],kind,rb,cb,rstart,cstart,mapmask,v[12],v[13])
   for k in range(14):metas[op,k]=row[k]
   op+=1;j=v[14]
 if len(index_pool)<at:
  flat=np.empty(at,np.int32)
 else:flat=index_pool
 at=0
 for p in pieces:flat[at:at+len(p)]=p;at+=len(p)
 return outkeys,shape,metas,flat,outR,outC,epoch
