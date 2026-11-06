"""Exact bitmap unions and bitmap-rank translations in the original scratch budget."""
import numpy as np
from numba import njit,types
from numba.typed import List
from numba.extending import intrinsic
from llvmlite import ir
ARR=types.Array(types.int32,1,'C')
U=np.uint64

@intrinsic
def popcount(typingctx,x):
    def emit(context,builder,sig,args):
        fn=builder.module.globals.get('llvm.ctpop.i64')
        if fn is None:fn=ir.Function(builder.module,ir.FunctionType(ir.IntType(64),[ir.IntType(64)]),name='llvm.ctpop.i64')
        return builder.call(fn,[args[0]])
    return types.uint64(types.uint64),emit

@intrinsic
def trailing(typingctx,x):
    def emit(context,builder,sig,args):
        fn=builder.module.globals.get('llvm.cttz.i64')
        if fn is None:fn=ir.Function(builder.module,ir.FunctionType(ir.IntType(64),[ir.IntType(64),ir.IntType(1)]),name='llvm.cttz.i64')
        return builder.call(fn,[args[0],ir.Constant(ir.IntType(1),0)])
    return types.uint64(types.uint64),emit

@njit(inline='always')
def mask_below(bit):
    return (U(1)<<U(bit))-U(1)

@njit(inline='always')
def set_interval(bits,lo,hi):
    first=lo//64;last=(hi-1)//64
    if first==last:bits[first]|=(~mask_below(lo%64))&mask_below((hi-1)%64+1) if hi%64 else ~mask_below(lo%64)
    else:
        bits[first]|=~mask_below(lo%64)
        for k in range(first+1,last):bits[k]=~U(0)
        bits[last]|=mask_below(hi%64) if hi%64 else ~U(0)

@njit(inline='always')
def shifted_or(dest,source,offset,lo,hi,shift):
    first=lo//64;last=(hi-1)//64
    for k in range(first,last+1):
        word=source[offset+k]
        if k==first:word&=~mask_below(lo%64)
        if k==last and hi%64:word&=mask_below(hi%64)
        base=64*k+shift;out=base//64;bit=base%64
        if bit==0:
            if 0<=out<len(dest):dest[out]|=word
        else:
            if 0<=out<len(dest):dest[out]|=word<<U(bit)
            if 0<=out+1<len(dest):dest[out+1]|=word>>U(64-bit)

@njit(cache=True,nogil=True)
def cache_layout(dimensions,identities,work,capacity):
    offsets=np.full(len(dimensions),-1,np.int64);words=(dimensions+63)//64
    score=np.empty(len(dimensions),np.float64)
    for c in range(len(dimensions)):score[c]=float(work[c])/max(1,words[c])
    at=0
    for c in np.argsort(score)[::-1]:
        if not identities[c] and at+words[c]<=capacity:offsets[c]=at;at+=words[c]
    return offsets,words,at

@njit(cache=True,nogil=True)
def support_one(c,terms,reps,class_terms,dimensions,identities,rs,cs,mapdata,oldbits,oldoffsets,outbits,cacheoffsets,workbits):
    if identities[c]:return mapdata[:dimensions[c]]
    nw=(dimensions[c]+63)//64;offset=cacheoffsets[c]
    bits=outbits[offset:offset+nw] if offset>=0 else workbits[:nw]
    if len(bits)!=nw:raise MemoryError('Bitmap workspace capacity')
    bits[:]=0
    for tid in class_terms[c]:
        t=terms[tid];axis,si=reps[tid];a=rs[si] if axis==0 else cs[si]
        lo=t[1];hi=t[2]
        if not t[3]:
            left=np.int64(a[lo]);right=np.int64(a[hi-1])+1
            if left+t[4]<0 or right+t[4]>dimensions[c]:raise ArithmeticError('Affine bitmap domain')
            if right-left==hi-lo:set_interval(bits,left+t[4],right+t[4])
            elif oldoffsets[t[0]]>=0:shifted_or(bits,oldbits,oldoffsets[t[0]],left,right,t[4])
            else:
                for k in range(lo,hi):
                    val=np.int64(a[k])+t[4];bits[val//64]|=U(1)<<U(val%64)
        else:
            for k in range(lo,hi):
                val=np.int64(mapdata[t[5]+a[k]-t[4]]);bits[val//64]|=U(1)<<U(val%64)
    if dimensions[c]%64 and bits[-1]>>U(dimensions[c]%64):raise ArithmeticError('Bitmap rank outside domain')
    count=np.int64(0);first=last=np.int64(-1)
    for k in range(nw):
        word=bits[k]
        if word:
            if first<0:first=64*k+np.int64(trailing(word))
            # Highest bit by repeatedly splitting; only one scan per nonzero word.
            high=63;z=word
            if z>>U(32)==0:high-=32;z<<=U(32)
            if z>>U(48)==0:high-=16;z<<=U(16)
            if z>>U(56)==0:high-=8;z<<=U(8)
            if z>>U(60)==0:high-=4;z<<=U(4)
            if z>>U(62)==0:high-=2;z<<=U(2)
            if z>>U(63)==0:high-=1
            last=64*k+high;count+=np.int64(popcount(word))
    if not count:raise ArithmeticError('Empty factored axis')
    if last-first+1==count:return mapdata[first:last+1]
    out=np.empty(count,np.int32);at=0
    for k in range(nw):
        word=bits[k]
        while word:
            out[at]=64*k+np.int64(trailing(word));at+=1;word&=word-U(1)
    if at!=count:raise ArithmeticError('Bitmap cardinality mismatch')
    return out

@njit(cache=True,nogil=True)
def support_group(group,terms,reps,class_terms,dimensions,identities,rs,cs,mapdata,oldbits,oldoffsets,outbits,cacheoffsets,workbits):
    result=List.empty_list(ARR)
    for c in group:result.append(support_one(c,terms,reps,class_terms,dimensions,identities,rs,cs,mapdata,oldbits,oldoffsets,outbits,cacheoffsets,workbits))
    return result

@njit(cache=True,nogil=True)
def canonical_cache(classaxes,canonical,cacheoffsets,dimensions,axis_count):
    offsets=np.full(axis_count,-1,np.int64);sizes=np.zeros(axis_count,np.int64)
    for c in range(len(classaxes)):
        a=classaxes[c];sid=canonical[c]
        if a[-1]-a[0]+1!=len(a) and cacheoffsets[c]>=0 and offsets[sid]<0:
            offsets[sid]=cacheoffsets[c];sizes[sid]=(dimensions[c]+63)//64
    return offsets,sizes

@njit(cache=True,nogil=True)
def translate_group(group,axes,axis_offsets,axis_words,bitmap,plan_offsets,plan_terms,by_axis,terms,reps,rs,cs,mapdata,flat,workspace):
    for cid in group:
        if not len(by_axis[cid]):continue
        a=axes[cid];linear=a[-1]-a[0]+1==len(a);cached=axis_offsets[cid]>=0
        prefix=workspace;inverse=workspace
        if not linear:
            if cached:
                nw=axis_words[cid]
                if nw+1>len(workspace):raise MemoryError('Bitmap prefix workspace capacity')
                prefix=workspace[:nw+1];prefix[0]=0
                for j in range(nw):prefix[j+1]=prefix[j]+np.int64(popcount(bitmap[axis_offsets[cid]+j]))
                if prefix[nw]!=len(a):raise ArithmeticError('Bitmap rank cardinality')
            else:
                if np.int64(a[-1])+1>len(workspace):raise MemoryError('Inverse workspace capacity')
                inverse=workspace[:np.int64(a[-1])+1]
                for k in range(len(a)):inverse[a[k]]=k
        for pid in by_axis[cid]:
            tid=plan_terms[pid];t=terms[tid];axis,si=reps[tid];src=rs[si] if axis==0 else cs[si];base=plan_offsets[pid]
            for k in range(t[2]-t[1]):
                val=np.int64(mapdata[t[5]+src[t[1]+k]-t[4]]) if t[3] else np.int64(src[t[1]+k])+t[4]
                if linear:rank=val-a[0]
                elif cached:
                    word=bitmap[axis_offsets[cid]+val//64]
                    if not word&(U(1)<<U(val%64)):raise ArithmeticError('Missing target bitmap rank')
                    rank=prefix[val//64]+np.int64(popcount(word&mask_below(val%64)))
                else:rank=inverse[val]
                flat[base+k]=rank
