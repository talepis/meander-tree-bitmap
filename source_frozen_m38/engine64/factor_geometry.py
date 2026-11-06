import time
import numpy as np
import psutil
from numba import types
from numba.typed import List
import reference_geometry as R
import axis_plan as A
import factor_supports as F
import bitmap_geometry as B
import factor_metadata as M
import map_ranges as MR
import memory_release as MEM

class Geometry(R.Geometry):
    def __init__(self,workers,split_maps=True,mode='bitmap'):
        if not split_maps:raise ValueError('Factored geometry requires separate immutable maps')
        super().__init__(workers,split_maps)
        self.mode=mode
        self.rowids=np.zeros(1,np.int64);self.colids=self.rowids.copy();self.bank=0
        self.release_bounds=None;self.index_live_bound=0
        self.bit_offsets=np.full(1,-1,np.int64);self.bit_bases=np.zeros(1,np.int64);self.range_tables=None
    def groups(self,dims,cost,capacity):
        pair=np.zeros((len(dims),2),np.int64);pair[:,0]=dims
        assignment,mr,mc,loads=R.assign_groups(pair,cost,self.workers,capacity,0,True)
        jobs=[];at=0
        for j in range(len(mr)):
            ids=np.flatnonzero(assignment==j).astype(np.int64)
            if len(ids):jobs.append((ids,at,int(mr[j])))
            at+=int(mr[j])
        if at>capacity:raise MemoryError('Factored workspace partition')
        return jobs,at
    def mapped(self,fn,jobs):
        try:return list(self.pool.map(fn,jobs)) if self.pool and len(jobs)>1 else [fn(j) for j in jobs]
        except BaseException:
            if self.pool:self.pool.shutdown(wait=True,cancel_futures=True);self.pool=None
            raise
    def prepare(self,*args):
        n,step,skeys,mx,rs,cs,offsets,targets,labels,weights,heights,tab,mapoff,maplen,mapmult,mapdata=args[:16]
        dense_cut=args[23];pool=args[24];start=time.perf_counter();tick=start
        keys,heads,descs=R.discover(n,step,skeys,mx,rs,cs,offsets,targets,labels,weights,heights,tab,mapoff,maplen,mapmult)
        discover_time=time.perf_counter()-tick;tick=time.perf_counter()
        width=2 if n==1 else 2*(2*n//3)
        if np.any(keys[:,1]+heights[keys[:,0]]>width):raise ArithmeticError('Centroid joint-axis domain exceeded')
        p=A.make_plan(keys,heads,descs,self.rowids,self.colids,heights,tab,dense_cut)
        terms,reps,dr,dc,rc,cc,ct,dimensions,identities,raw_union,unique_union=p
        scratch,cost,pure,mapped,interval,nonlinear=F.costs(terms,reps,ct,dimensions,identities,rs,cs)
        if self.range_tables is None:self.range_tables=MR.tables(n,tab)
        bases,spans=MR.axis_bounds(terms,reps,ct,dimensions,identities,skeys,heights,rs,cs,tab,*self.range_tables)
        oldbits=self.scratch[self.bank].view(np.uint64);outbits=self.scratch[1-self.bank].view(np.uint64)
        work=self.scratch[2];cacheoffsets,wordcounts,cachewords=B.cache_layout(spans,identities,cost,0 if self.mode=='bitmap_nocache' else len(outbits))
        need=np.where((cacheoffsets<0)&(identities==0),wordcounts,0).astype(np.int64)
        jobs,reserved=self.groups(need,cost,len(work));plan_time=time.perf_counter()-tick;tick=time.perf_counter()
        def build(job):
            ids,at,size=job
            return B.support_group(ids,terms,reps,ct,dimensions,bases,spans,identities,rs,cs,mapdata,oldbits,self.bit_offsets,self.bit_bases,outbits,cacheoffsets,work[at:at+size].view(np.uint64))
        outputs=self.mapped(build,jobs);class_axes=List.empty_list(A.ARR)
        for _ in range(len(ct)):class_axes.append(np.empty(0,np.int64))
        for job,arrays in zip(jobs,outputs):
            for cid,a in zip(job[0],arrays):class_axes[int(cid)]=a
        axes,rids,cids,canonical=A.canonicalize(class_axes,rc,cc,mapdata)
        bitoffs,bitwords,bitbases=B.canonical_cache(class_axes,canonical,cacheoffsets,bases,spans,len(axes))
        support_time=time.perf_counter()-tick;tick=time.perf_counter()
        shape,meta,poff,pt,byaxis,outR,outC,raw_indices=M.prepare(keys,heads,descs,skeys,rs,cs,heights,tab,terms,dr,dc,axes,rids,cids)
        needed=int(poff[-1]);new_bytes=needed*8
        reuse=step>0 and len(pool)>=needed
        replacement_advice=0
        if MEM.trims_indices() and not reuse and pool.nbytes:
            # Old stage indices have no remaining readers and will not be reused.
            # Reclaim them before allocating their replacement, while caller aliases
            # still keep the old ndarray alive. No extra allocation or arithmetic.
            replacement_advice=MEM.discard_dead(pool)
        if (0 if reuse else new_bytes)+(3<<30)>psutil.virtual_memory().available:raise MemoryError('Fresh factored index allocation guard')
        # The old M30 reservation assumes duplicated indices. Start this representation empty.
        flat=pool if reuse else np.empty(needed,np.int64)
        transdims=np.zeros(len(axes),np.int64);transcost=np.zeros(len(axes),np.int64)
        for cid,a in enumerate(axes):
            if len(byaxis[cid]):
                linear=int(a[-1])-int(a[0])+1==len(a)
                if not linear:transdims[cid]=int(bitwords[cid])+1 if bitoffs[cid]>=0 else int(a[-1])+1
                transcost[cid]=sum(int(poff[k+1]-poff[k]) for k in byaxis[cid])+len(a)
        tjobs,treserved=self.groups(transdims,transcost,len(work))
        metadata_time=time.perf_counter()-tick;tick=time.perf_counter()
        def translate(job):
            ids,at,size=job
            return B.translate_group(ids,axes,bitoffs,bitwords,bitbases,outbits,poff,pt,byaxis,terms,reps,rs,cs,mapdata,flat,work[at:at+size])
        self.mapped(translate,tjobs)
        translation_time=time.perf_counter()-tick
        self.rowids=rids;self.colids=cids;self.bit_offsets=bitoffs;self.bit_bases=bitbases;self.bank=1-self.bank
        release_stats={}
        if MEM.active():
            if self.release_bounds is None:self.release_bounds=[a.nbytes for a in self.scratch]
            release_stats,self.release_bounds,self.index_live_bound=MEM.after_geometry(self.scratch,self.bank,int(cachewords)*8,max(reserved,treserved)*8,flat,new_bytes,self.release_bounds,self.index_live_bound,reuse)
        self.stats.append(dict(stage=step+1,replaced_index_dead_page_advice_bytes=replacement_advice,**release_stats,discovery_seconds=discover_time,planning_seconds=plan_time,
            support_seconds=support_time,metadata_seconds=metadata_time,translation_seconds=translation_time,
            total_geometry_seconds=time.perf_counter()-start,blocks=len(keys),descriptors=len(descs),
            axis_classes=len(ct),canonical_axes=len(axes),raw_union_elements=int(raw_union),unique_union_elements=int(unique_union),
            raw_translation_elements=int(raw_indices),unique_translation_elements=needed,translated_index_bytes=new_bytes,
            bitmap_uncropped_domain_words=int(np.sum((dimensions+63)//64)),bitmap_cropped_domain_words=int(np.sum((spans+63)//64)),bitmap_cached_classes=int(np.sum(cacheoffsets>=0)),bitmap_cache_bytes=int(cachewords)*8,bitmap_cache_capacity_bytes=outbits.nbytes,
            support_workspace_bytes=reserved*8,translation_workspace_bytes=treserved*8,workspace_capacity_bytes=work.nbytes,
            support_groups=len(jobs),translation_groups=len(tjobs)))
        return keys,shape,meta,flat,outR,outC,args[22]+1
