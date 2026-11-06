"""Geometry-only support planning and a fully compiled uint64 arena transfer."""
import gc,math,time
import memory_release as MEM
import numpy as np,numba,psutil
from deterministic import Compiler
from weighted import Operators
import dense_geometry as G
import fast_support as F
import arena_kernels as K
import packed_kernels as P
import fast_maps as N
import native_linear_support as Q
import wide_kernels as W
from numba.typed import List
from numba import types
def shape_layout(geo):
 keys=list(geo);shape=np.asarray([(len(geo[k][0]),len(geo[k][1])) for k in keys],np.int64);offset=np.zeros(len(keys)+1,np.int64);offset[1:]=np.cumsum(shape[:,0]*shape[:,1]);return keys,shape,offset
def stages(n,g,dense_cut=4096):
 cur={(0,0,0):(np.zeros(1,np.int64),np.zeros(1,np.int64))};maxB=max(G.B(m,h) for m in range(n+1) for h in range(m+1));invR=np.empty(maxB,np.int64);invC=invR.copy();tagR=np.zeros(maxB,np.int64);tagC=tagR.copy();bufR=invR.copy();bufC=invR.copy();epoch=0
 cut=2*n-g['tail_R']
 for step in range(cut):
  incoming={};remaining=2*n-step;skeys,sshape,soff=shape_layout(cur);sourceids={k:i for i,k in enumerate(skeys)}
  for source,(ri,ci) in cur.items():
   gid,a,h=source;b=int(g['heights'][gid]);Na,Nb=G.B(a,h),G.B(b,h);Up,Uq=G.B(a-1,h-1),G.B(b-1,h-1);rowcut=Up if len(ri)==Na else int(np.searchsorted(ri,Up));colcut=Uq if len(ci)==Nb else int(np.searchsorted(ci,Uq))
   def push(dest,kind,rect,off=(0,0),mp=None):
    dg,aa,hh=dest;bb=int(g['heights'][dg])
    if aa<0 or aa>remaining-1 or hh<0 or not G.B(aa,hh) or not G.B(bb,hh):return
    r0,r1,c0,c1=rect;lr=0 if r0==0 else rowcut;hr=len(ri) if r1==Na else rowcut;lc=0 if c0==0 else colcut;hc=len(ci) if c1==Nb else colcut
    if lr==hr or lc==hc:return
    mapping=F.EMPTY;mult=1
    if mp is not None:mapping,mult=G.close_map(*mp)
    incoming.setdefault(dest,[]).append((source,kind,(lr,hr,lc,hc),(off[0]-r0,r0,kind=='rows'),(off[1]-c0,c0,kind=='cols'),mapping,w,mult))
   for e in range(int(g['offsets'][gid]),int(g['offsets'][gid+1])):
    dest=int(g['targets'][e]);w=int(g['weights'][e]);opened=int(g['labels'][e])
    if opened:
     if a<remaining:push((dest,a+1,h+1),'add',(0,Na,0,Nb))
     if a:
      push((dest,a-1,h-1),'add',(0,Up,0,Nb),(0,G.B(b,h-2)));push((dest,a-1,h+1),'add',(Up,Na,0,Nb))
    else:
     if a<remaining:
      push((dest,a+1,h-1),'add',(0,Na,0,Uq),(G.B(a,h-2),0));push((dest,a+1,h+1),'add',(0,Na,Uq,Nb))
     if a:
      push((dest,a-1,h+1),'add',(Up,Na,Uq,Nb))
      if h>=1:
       push((dest,a-1,h-1),'cols',(0,Up,Uq,Nb),mp=(b-1,h+1));push((dest,a-1,h-1),'rows',(Up,Na,0,Uq),mp=(a-1,h+1))
  target={};meta=[];index=[];at=0
  for di,(dest,descs) in enumerate(incoming.items()):
   dg,da,dh=dest;db=int(g['heights'][dg]);epoch+=1;nr=nc=0
   if epoch>=2**32:tagR[:]=0;tagC[:]=0;epoch=1
   full=G.B(da,dh)*G.B(db,dh)<=dense_cut
   if full:tr=np.arange(G.B(da,dh),dtype=np.int64);tc=np.arange(G.B(db,dh),dtype=np.int64)
   else:
    for source,kind,rect,rowrule,colrule,mapping,w,mult in descs:
     ri,ci=cur[source];lr,hr,lc,hc=rect;nr=F.collect(ri,lr,hr,*rowrule,mapping,tagR,epoch,bufR,nr);nc=F.collect(ci,lc,hc,*colrule,mapping,tagC,epoch,bufC,nc)
    tr=np.sort(bufR[:nr].copy());tc=np.sort(bufC[:nc].copy())
   target[dest]=(tr,tc);F.fill_inverse(tr,invR);F.fill_inverse(tc,invC)
   for source,kind,rect,rowrule,colrule,mapping,w,mult in descs:
    ri,ci=cur[source];lr,hr,lc,hc=rect;sg,sa,sh=source;fullsource=len(ri)==G.B(sa,sh) and len(ci)==G.B(int(g['heights'][sg]),sh)
    if full and fullsource and kind=='add':k=0;rb=lr+rowrule[0];cb=lc+colrule[0];rstart=cstart=0
    else:
     dr=F.translate(ri,lr,hr,*rowrule,mapping,invR);dc=F.translate(ci,lc,hc,*colrule,mapping,invC)
     if kind=='add' and int(dr[-1])-int(dr[0])+1==len(dr) and int(dc[-1])-int(dc[0])+1==len(dc):k=0;rb=int(dr[0]);cb=int(dc[0]);rstart=cstart=0
     else:
      k=2 if kind=='rows' else 1;rb=cb=0;rstart=at;index.append(dr);at+=len(dr);cstart=at;index.append(dc);at+=len(dc)
    meta.append((sourceids[source],di,lr,hr,lc,hc,k,rb,cb,rstart,cstart,0,w,mult))
  dkeys,dshape,doff=shape_layout(target);meta=np.asarray(meta,np.int64).reshape(-1,14);indices=np.concatenate(index) if index else np.zeros(0,np.int64)
  yield dict(stage=step+1,skeys=skeys,dkeys=dkeys,sshape=sshape,dshape=dshape,soff=soff,doff=doff,meta=meta,indices=indices,geo=target,scratch_bytes=maxB*48)
  cur=target
def calculate(n,workers=12,tail=8,memory_mib=32768,timeout=600,dense_cut=0,threshold=1000000,progress=None,layer_check=None,index_prealloc_bytes=0,minimize_grammar=False,split_maps=False,plan_hook=None,parallel_geometry=0,lifetime=False,compressed_maps=False,vector_maxima=False,geometry_mode="baseline",numeric_mode="baseline",lower_sizes=None,modulus=None):
 if compressed_maps or not(split_maps and lifetime and geometry_mode=='target' and parallel_geometry):raise ValueError('Rank64 backend requires the audited split-map/target/lifetime profile')
 if modulus is not None and modulus not in ((1<<64),(1<<64)-1,7):raise ValueError('Unsupported modular residue ring')
 if type(n)!=int or not 1<=n<=38 or type(workers)!=int or not 1<=workers<=numba.config.NUMBA_NUM_THREADS or type(tail)!=int or not 1<=tail<=10:raise ValueError('Arena audited domain')
 if not math.isfinite(memory_mib) or not math.isfinite(timeout) or memory_mib<=0 or timeout<=0:raise ValueError('Invalid resources')
 if n<37 and (math.comb(2*n,n)//(n+1))**2 >= 1<<128:raise OverflowError('Proved terminal 128-bit bound')
 max_index=max(G.B(m,h) for m in range(n+1) for h in range(m+1))
 if max_index>=1<<60:raise OverflowError('Signed-64 rank byte domain')
 fixed=48*max_index+8*(max_index+sum(G.B(m,h) for m in range(1,n) for h in range(2,m+1)))
 if fixed+(3<<30)>psutil.virtual_memory().available:raise MemoryError('Fixed support preflight')
 sizes=tuple(range(1,(n+1)//2+1)) if lower_sizes is None else tuple(lower_sizes)
 if not sizes or any(type(t)!=int or not 1<=t<=(n+1)//2 for t in sizes) or len(set(sizes))!=len(sizes):raise ValueError('Invalid lower-family subset')
 numba.set_num_threads(workers);start=time.perf_counter();ops=Operators(min(tail,2*n));compiler=Compiler(n,sizes,ops);C=math.comb(2*n,n)//(n+1)
 lower_coverage=compiler.lower_count((0,0,0,0,0,0))
 if lower_sizes is None and lower_coverage!=C:raise ArithmeticError('Arena weighted lower coverage')
 if not 0<lower_coverage<=C:raise ArithmeticError('Invalid lower-family coverage')
 g=compiler.compile();
 if minimize_grammar:
  from grammar_min import minimize
  g=minimize(g)
 if compressed_maps:
  if not split_maps or parallel_geometry or lifetime:raise ValueError('Compressed map experiment uses separate maps without threaded/lifetime variants')
  import compact_geometry as Q
  import compact_wide as W
 elif split_maps:
  import split_geometry as Q
  import split_wide as W
 else:
  import native_linear_support as Q
  import wide_kernels as W
 if geometry_mode=="profile":
  if not split_maps:raise ValueError("Profile requires split maps")
  import profile_geometry as Q
  Q.PROFILE.clear()
 if geometry_mode=="profile_native":
  if split_maps:raise ValueError("Native profiling uses combined maps")
  import profile_native_geometry as Q
  Q.PROFILE.clear()
 if vector_maxima:import vector_maxima as VM
 if modulus is not None:
  import tree_mod_arith as W
  W.configure(modulus)
 g['tail_R']=ops.R;parallel_prep=None
 if parallel_geometry:
  if not split_maps and geometry_mode!="target":raise ValueError('Parallel geometry requires separate immutable maps')
  if geometry_mode=="target":
   from target_geometry import Geometry
   parallel_prep=Geometry(parallel_geometry,split_maps=split_maps)
  else:
   from parallel_geometry import Geometry
   parallel_prep=Geometry(g,parallel_geometry)
 prep=time.perf_counter()-start;source=np.zeros(8,np.uint8);source[0]=1;swidth=np.full(1,8 if modulus is not None else 1,np.uint8);soff=np.asarray([0,8],np.int64);mx=np.ones(1,np.uint64);mxhi=np.zeros(1,np.uint64);numeric=geometrytime=0.;peakbytes=0;peak_arrays=0;peak_metadata=0;steps=[];lastgeo={(0,0,0):(np.zeros(1,np.int64),np.zeros(1,np.int64))};lastkeys=[(0,0,0)];lastshape=np.ones((1,2),np.int64);lastoff=np.asarray([0,8],np.int64);map_tick=time.perf_counter();
 if compressed_maps:
  import compressed_maps as CM
  mapoff,maplen,mapmult,mapdata,mapstarts,mapshifts=CM.maps(n)
 else:mapoff,maplen,mapmult,mapdata=N.maps(n)
 map_seconds=time.perf_counter()-map_tick;skeys=np.asarray([[0,0,0]],np.int64);source_shape=np.ones((1,2),np.int64);schedule_step=0;life_stats=[]
 if lifetime:
  if modulus is not None:
   import tree_mod_lifetime as LT
   LT.configure(modulus,workers)
  elif numeric_mode=="profile":import profile_lifetime as LT
  elif numeric_mode!="baseline":
   import numeric_lifetime as LT
   LT.configure(numeric_mode,workers)
  else:import lifetime as LT
  blocks=List.empty_list(types.Array(types.uint8,1,'C'));blocks.append(source);source=None
 index_pool=np.empty(max(0,(index_prealloc_bytes-mapdata.nbytes+7)//8),np.int64) if split_maps else np.empty(max(len(mapdata),(index_prealloc_bytes+7)//8),np.int64)
 if not split_maps:index_pool[:len(mapdata)]=mapdata
 rs=List.empty_list(Q.ARR);cs=List.empty_list(Q.ARR);rs.append(np.zeros(1,np.int64));cs.append(np.zeros(1,np.int64));maxB=max(G.B(m,h) for m in range(n+1) for h in range(m+1));epoch=0
 if geometry_mode!="target":
  invR=np.empty(maxB,np.int64);invC=invR.copy();tagR=np.zeros(maxB,np.int64);tagC=tagR.copy();bufR=invR.copy();bufC=invR.copy()
 if geometry_mode=="target":
  # Three pooled arrays replace six separately allocated arrays, with equal total size.
  # Both axes can borrow free workspace from the other; no full-size per-thread copies.
  inv_pool=np.empty(2*maxB,np.int64);tag_pool=np.zeros(2*maxB,np.int64);buf_pool=np.empty(2*maxB,np.int64)
  invR=inv_pool[:maxB];invC=inv_pool[maxB:];tagR=tag_pool[:maxB];tagC=tag_pool[maxB:];bufR=buf_pool[:maxB];bufC=buf_pool[maxB:]
  parallel_prep.bind_scratch(inv_pool,tag_pool,buf_pool)
 while True:
  tick=time.perf_counter()
  if schedule_step>=2*n-ops.R:break
  prepare_func=(lambda *args:Q.prepare_compact(*args,mapstarts,mapshifts)) if compressed_maps else (parallel_prep.prepare if parallel_prep else (Q.prepare_split if split_maps else Q.prepare))
  dkeys,dshape,meta,ind,drs,dcs,epoch=prepare_func(n,schedule_step,skeys,mx|mxhi,rs,cs,g['offsets'],g['targets'],g['labels'],g['weights'],g['heights'],G.TAB,mapoff,maplen,mapmult,mapdata,invR,invC,tagR,tagC,bufR,bufC,epoch,dense_cut,index_pool)
  # Every source support has now been consumed by geometry/translation.
  # Numerical kernels need only coefficient blocks, shapes, indices and maps.
  if MEM.active():rs=drs;cs=dcs
  index_pool=ind
  active_ind=ind[:parallel_prep.stats[-1]['translated_index_bytes']//8] if MEM.trims_indices() and geometry_mode=='target' and parallel_prep else ind
  s=dict(stage=schedule_step+1,dkeys=[tuple(map(int,k)) for k in dkeys],dshape=dshape,meta=meta,indices=active_ind,sshape=source_shape)
  stage_geometry_seconds=time.perf_counter()-tick;geometrytime+=stage_geometry_seconds
  if time.perf_counter()-start>timeout:raise TimeoutError('Arena deadline')
  tick=time.perf_counter();cap_tick=tick
  if modulus is not None:
   caps=np.full(len(s['dkeys']),np.uint64(modulus-1),np.uint64);caphi=np.zeros(len(caps),np.uint64)
  elif n>=26:
   caps,caphi=W.caps_exact_128(mx,mxhi,s['meta'],len(s['dkeys']))
  else:
   caps,caphi=W.caps(mx,mxhi,s['meta'],len(s['dkeys']),np.uint64((C*C)&((1<<64)-1)),np.uint64((C*C)>>64))
  if sum(((int(r)*int(c)*16+7)//8)*8 for r,c in s['dshape'])>=1<<63:raise OverflowError('Signed byte-offset domain')
  doff,dwidth=W.layout(caps,caphi,s['dshape']);bytes_needed=int(doff[-1]);metadata=sum(s[k].nbytes for k in ['sshape','dshape','meta','indices']);source_bytes=sum(len(x) for x in blocks) if lifetime else source.nbytes
  cap_seconds=time.perf_counter()-cap_tick;plan_tick=time.perf_counter()
  life_plan=LT.static_plan(s['meta'],np.diff(soff),np.diff(doff)) if lifetime else None
  plan_seconds=time.perf_counter()-plan_tick
  coefficient_charge=int(life_plan[2]) if lifetime else source_bytes+bytes_needed
  charge=coefficient_charge+metadata;peakbytes=max(peakbytes,charge);peak_arrays=max(peak_arrays,coefficient_charge);peak_metadata=max(peak_metadata,metadata)
  fresh_needed=max(0,coefficient_charge-source_bytes) if lifetime else bytes_needed
  if charge>memory_mib*2**20 or fresh_needed+(3<<30)>psutil.virtual_memory().available:raise MemoryError('Arena RAM gate: %d'%charge)
  if plan_hook:plan_hook(s,soff,doff,swidth,dwidth)
  if lifetime:
   transfer_tick=time.perf_counter();old_blocks=blocks;blocks=None
   blocks,lstats=LT.execute(old_blocks,soff,doff,swidth,dwidth,s['sshape'],s['dshape'],s['meta'],s['indices'],mapdata,threshold,split_maps,memory_mib*2**20-metadata,life_plan)
   transfer_seconds=time.perf_counter()-transfer_tick;max_tick=time.perf_counter()
   mx,mxhi=LT.maxima(blocks,dwidth,s['dshape']);max_seconds=time.perf_counter()-max_tick
   life_stats.append(dict(stage=s['stage'],caps_layout_seconds=cap_seconds,planning_seconds=plan_seconds,execute_seconds=transfer_seconds,maxima_seconds=max_seconds,**lstats));target=None
  else:
   target=np.zeros(bytes_needed,np.uint8)
   args=(source,target,soff,doff,swidth,dwidth,s['sshape'],s['dshape'],s['meta'],s['indices'])
   if compressed_maps:W.execute(*args,mapdata,mapstarts,mapshifts,threshold)
   elif split_maps:W.execute(*args,mapdata,threshold)
   else:W.execute(*args,threshold)
   mx,mxhi=(VM.maxima if vector_maxima else W.maxima)(target,doff,dwidth,s['dshape'])
  if np.any((mxhi>caphi)|((mxhi==caphi)&(mx>caps))):raise ArithmeticError('Arena exact maximum exceeds certified bound')
  source=target;soff=doff;swidth=dwidth;stage_numeric_seconds=time.perf_counter()-tick;numeric+=stage_numeric_seconds;lastkeys=s['dkeys'];lastshape=s['dshape'];lastoff=doff;skeys=dkeys;source_shape=dshape;rs=drs;cs=dcs;schedule_step+=1;row=dict(stage=s['stage'],geometry_seconds=stage_geometry_seconds,numeric_seconds=stage_numeric_seconds,coefficient_source_and_target_bytes=charge-metadata,entries=int(np.sum(dshape[:,0]*dshape[:,1])),buffer_bytes=bytes_needed,metadata_bytes=metadata,max_decimal=str(max((int(l)+(int(h)<<64) for l,h in zip(mx,mxhi)),default=0)),wide_blocks=int(np.sum(dwidth==16)),elapsed=time.perf_counter()-start)
  if n>=37:
   row['certified_cap_bits']=max((int(l)+(int(h)<<64) for l,h in zip(caps,caphi)),default=0).bit_length()
   row['certification']='exact_checked_128_bit_cap_before_numeric_writes'
  if layer_check:
   s.update(byte_offsets=doff,widths=dwidth,row_ids=rs,column_ids=cs);layer_check(s['stage'],np.concatenate(list(blocks)) if lifetime else source,s)
  if split_maps and lifetime and parallel_prep:
   # Translation indices for this stage have finished their only numerical use.
   # The old pool was sized for an earlier, much larger stage. Drop every local
   # reference before the next geometry allocation, without changing any values.
   needed_index_bytes=parallel_prep.stats[-1]['translated_index_bytes']
   allocated_index_bytes=index_pool.nbytes
   if allocated_index_bytes>=max(1<<30,2*needed_index_bytes):
    before=psutil.Process().memory_info().rss
    del s,ind,active_ind
    index_pool=np.empty(0,np.int64)
    gc.collect()
    after=psutil.Process().memory_info().rss
    row['released_index_pool_capacity_bytes']=allocated_index_bytes
    row['released_index_pool_RSS_drop_bytes']=max(0,before-after)
  steps.append(row)
  if progress:progress(row)
 tick=time.perf_counter();answer=0;maps={};bound=C*C
 for j,(gid,a,h) in enumerate(lastkeys):
  b=int(g['heights'][gid]);key=a,b,h
  if key not in maps:maps[key]=G.geometry(a,b,h,ops.lookup,G.TAB)
  terminal_bound=min(bound,(1<<128)-1)
  indices=maps[key][np.ix_(rs[j],cs[j])];lo,hi=W.dot128(blocks[j] if lifetime else source,0 if lifetime else int(lastoff[j]),int(swidth[j]),lastshape[j],indices,ops.rows[int(g['opids'][gid])],np.uint64(terminal_bound&((1<<64)-1)),np.uint64(terminal_bound>>64));answer+=int(lo)+(int(hi)<<64)
 terminal=time.perf_counter()-tick
 if time.perf_counter()-start>timeout:raise TimeoutError('Arena result past deadline')
 if modulus is not None:answer%=modulus
 elif not 0<=answer<=lower_coverage*C:raise ArithmeticError('Family answer exceeds proved Catalan-pair bound')
 if parallel_prep:parallel_prep.close()
 return dict(status='PASS',lower_sizes=list(sizes),weighted_lower_coverage=str(lower_coverage),proved_global_bound_decimal=str(C*C),global_bound_basis='weighted deterministic prefix injection into all Catalan pairs',variant='M37_FAMILY_RANK64_MODULAR' if modulus is not None else 'M38_FAMILY_RANK64_CERTIFIED128',n=n,Mn=str(answer),modulus_decimal=str(modulus) if modulus is not None else None,workers=workers,tail=tail,dense_cut=dense_cut,seconds=time.perf_counter()-start,prep_seconds=prep,geometry_seconds=geometrytime,numeric_seconds=numeric,terminal_seconds=terminal,peak_logical_bytes=peakbytes,peak_coefficient_arena_bytes=peak_arrays,peak_geometry_metadata_bytes=peak_metadata,constant_closemap_bytes=mapdata.nbytes+(mapstarts.nbytes+mapshifts.nbytes if compressed_maps else 0),compressed_maps=compressed_maps,vector_maxima=vector_maxima,constant_map_seconds=map_seconds,support_scratch_bytes=maxB*48,steps=steps,grammar_nodes=g['nodes'],grammar_minimization=g.get('minimization'),minimize_grammar=minimize_grammar,split_maps=split_maps,lifetime=lifetime,lifetime_stages=life_stats,numeric_mode=numeric_mode,geometry_profile=Q.PROFILE if geometry_mode in ("profile","profile_native") else None,parallel_geometry_stats=parallel_prep.stats if parallel_prep else None,matrix_read_bytes=0,matrix_write_bytes=0,independent_CRT=False)
