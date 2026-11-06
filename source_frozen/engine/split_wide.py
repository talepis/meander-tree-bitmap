"""Checked 128-bit fallback; small blocks retain the original packed kernels."""
import numpy as np
from numba import njit, prange
import split_packed as P
U=np.uint64
MAX=U(18446744073709551615)
MASK=U(4294967295)

@njit(inline='always')
def mul_small(lo,hi,w):
    # w <= 2**32-1. Return an exact two-limb result or reject before overflow.
    if w>MASK: raise OverflowError('128-bit multiplier domain')
    p0=(lo&MASK)*w
    p1=(lo>>U(32))*w+(p0>>U(32))
    low=(p0&MASK)|((p1&MASK)<<U(32))
    carry=p1>>U(32)
    if hi and w>MAX//hi: raise OverflowError('128-bit product')
    high=hi*w
    if carry>MAX-high: raise OverflowError('128-bit product carry')
    return low,high+carry

@njit(inline='always')
def plus(lo,hi,vl,vh):
    nl=lo+vl; carry=U(nl<lo)
    if vh>MAX-hi: raise OverflowError('128-bit sum')
    nh=hi+vh
    if carry>MAX-nh: raise OverflowError('128-bit sum carry')
    return nl,nh+carry

@njit(inline='always')
def mul_capped(lo,hi,w,glo,ghi):
    if w>MASK:raise OverflowError('Upper-bound multiplier domain')
    p0=(lo&MASK)*w;p1=(lo>>U(32))*w+(p0>>U(32))
    low=(p0&MASK)|((p1&MASK)<<U(32));carry=p1>>U(32)
    if hi and w>MAX//hi:return glo,ghi
    high=hi*w
    if carry>MAX-high:return glo,ghi
    high+=carry
    if high>ghi or (high==ghi and low>glo):return glo,ghi
    return low,high

@njit(inline='always')
def plus_capped(lo,hi,vl,vh,glo,ghi):
    nl=lo+vl;carry=U(nl<lo)
    if vh>MAX-hi:return glo,ghi
    nh=hi+vh
    if carry>MAX-nh:return glo,ghi
    nh+=carry
    if nh>ghi or (nh==ghi and nl>glo):return glo,ghi
    return nl,nh

@njit(cache=True,nogil=True)
def caps(mxlo,mxhi,meta,blocks,glo,ghi):
    low=np.zeros(blocks,np.uint64);high=low.copy()
    if not glo and not ghi:raise ArithmeticError('Empty global coefficient bound')
    for j in range(len(mxlo)):
        if mxhi[j]>ghi or (mxhi[j]==ghi and mxlo[j]>glo):raise ArithmeticError('Source exceeds proved global bound')
    for op in meta:
        s=op[0];d=op[1];a=U(op[12]);b=U(op[13])
        if b and a>MASK//b: raise OverflowError('Block multiplier')
        # Only the conservative bound is saturated. Coefficient arithmetic
        # retains checked exact two-limb operations and never uses saturation.
        vl,vh=mul_capped(mxlo[s],mxhi[s],a*b,glo,ghi)
        low[d],high[d]=plus_capped(low[d],high[d],vl,vh,glo,ghi)
    return low,high

@njit(cache=True,nogil=True)
def caps_exact_128(mxlo,mxhi,meta,blocks):
    """Prove every target cap fits 128 bits using exact checked arithmetic.

    This is required at n=37 because the Catalan-square bound is 131 bits.
    A failure stops the job before target allocation or unchecked parallel cells.
    """
    low=np.zeros(blocks,np.uint64);high=low.copy()
    for op in meta:
        s=op[0];d=op[1];a=U(op[12]);b=U(op[13])
        if b and a>MASK//b:raise OverflowError('Block multiplier domain')
        vl,vh=mul_small(mxlo[s],mxhi[s],a*b)
        low[d],high[d]=plus(low[d],high[d],vl,vh)
    return low,high

@njit(cache=True,nogil=True)
def layout(lo,hi,shape):
    offsets=np.zeros(len(lo)+1,np.int64);width=np.zeros(len(lo),np.uint8)
    for j in range(len(lo)):
        c=lo[j];w=16 if hi[j] else (1 if c<256 else (2 if c<65536 else (4 if c<4294967296 else 8)))
        width[j]=w;offsets[j+1]=offsets[j]+((shape[j,0]*shape[j,1]*w+7)//8)*8
    return offsets,width

@njit(inline='always')
def load(data,at,w,k):
    if w==1:return U(data[at+k]),U(0)
    if w==2:return U(data.view(np.uint16)[at//2+k]),U(0)
    if w==4:return U(data.view(np.uint32)[at//4+k]),U(0)
    if w==8:return data.view(np.uint64)[at//8+k],U(0)
    z=data.view(np.uint64);i=at//8+2*k
    return z[i],z[i+1]

@njit(inline='always')
def store(data,at,w,k,lo,hi):
    if w==1:data[at+k]=np.uint8(lo)
    elif w==2:data.view(np.uint16)[at//2+k]=np.uint16(lo)
    elif w==4:data.view(np.uint32)[at//4+k]=np.uint32(lo)
    elif w==8:data.view(np.uint64)[at//8+k]=lo
    else:
        z=data.view(np.uint64);i=at//8+2*k;z[i]=lo;z[i+1]=hi

@njit(inline='always')
def cell(source,target,sa,da,sw,dw,snc,dnc,op,indices,mapdata,i,j):
    kind=op[6]
    rr=op[7]+i if kind==0 else (mapdata[op[9]+i] if op[11]&1 else indices[op[9]+i])
    cc=op[8]+j if kind==0 else (mapdata[op[10]+j] if op[11]&2 else indices[op[10]+j])
    x,y=load(source,sa,sw,(op[2]+i)*snc+op[4]+j)
    # The checked cap has already certified this weight, product and every
    # partial sum. Keeping exceptions outside prange permits parallel lowering.
    w=U(op[12]);p0=(x&MASK)*w;p1=(x>>U(32))*w+(p0>>U(32))
    vl=(p0&MASK)|((p1&MASK)<<U(32));vh=y*w+(p1>>U(32))
    tl,th=load(target,da,dw,rr*dnc+cc)
    # The destination cap bounds every nonnegative partial sum and its chosen width.
    nl=tl+vl;nh=th+vh+U(nl<tl)
    store(target,da,dw,rr*dnc+cc,nl,nh)

@njit(cache=True,nogil=True)
def serial(source,target,sa,da,sw,dw,snc,dnc,op,indices,mapdata):
    for i in range(op[3]-op[2]):
        for j in range(op[5]-op[4]):cell(source,target,sa,da,sw,dw,snc,dnc,op,indices,mapdata,i,j)

@njit(cache=True,nogil=True,parallel=True)
def parallel_rows(source,target64,sa,da,sw,dw,snc,dnc,op,indices,mapdata):
    for i in prange(op[3]-op[2]):
        for j in range(op[5]-op[4]):wide_cell(source,target64,sa,da,sw,snc,dnc,op,indices,mapdata,np.int64(i),j)

@njit(cache=True,nogil=True,parallel=True)
def parallel_cols(source,target64,sa,da,sw,dw,snc,dnc,op,indices,mapdata):
    for j in prange(op[5]-op[4]):
        for i in range(op[3]-op[2]):wide_cell(source,target64,sa,da,sw,snc,dnc,op,indices,mapdata,i,np.int64(j))

@njit(inline='always')
def wide_cell(source,target64,sa,da,sw,snc,dnc,op,indices,mapdata,i,j):
    # A typed output view is formed OUTSIDE prange, so its mutations remain
    # visible to Numba's parallel alias analysis.
    rr=op[7]+i if op[6]==0 else (mapdata[op[9]+i] if op[11]&1 else indices[op[9]+i])
    cc=op[8]+j if op[6]==0 else (mapdata[op[10]+j] if op[11]&2 else indices[op[10]+j])
    x,y=load(source,sa,sw,(op[2]+i)*snc+op[4]+j)
    w=U(op[12]);p0=(x&MASK)*w;p1=(x>>U(32))*w+(p0>>U(32))
    vl=(p0&MASK)|((p1&MASK)<<U(32));vh=y*w+(p1>>U(32))
    at=da//8+2*(rr*dnc+cc);tl=target64[at];th=target64[at+1]
    nl=tl+vl;nh=th+vh+U(nl<tl);target64[at]=nl;target64[at+1]=nh

def execute(source,target,soff,doff,swidth,dwidth,sshape,dshape,meta,indices,mapdata,threshold):
    # Batch adjacent narrow operations through the unchanged, cached fast path.
    begin=0
    for j,op in enumerate(meta):
        si,di=map(int,op[:2])
        if swidth[si]<=8 and dwidth[di]<=8:continue
        if begin<j:P.execute(source,target,soff,doff,swidth,dwidth,sshape,dshape,meta[begin:j],indices,mapdata,threshold)
        args=(source,target,int(soff[si]),int(doff[di]),int(swidth[si]),int(dwidth[di]),int(sshape[si,1]),int(dshape[di,1]),op,indices,mapdata)
        count=int((op[3]-op[2])*(op[5]-op[4]))
        if count<threshold or dwidth[di]!=16:serial(*args)
        else:
            pargs=(source,target.view(np.uint64))+args[2:]
            if op[6]==2:parallel_cols(*pargs)
            else:parallel_rows(*pargs)
        begin=j+1
    if begin<len(meta):P.execute(source,target,soff,doff,swidth,dwidth,sshape,dshape,meta[begin:],indices,mapdata,threshold)

@njit(cache=True,nogil=True,parallel=True)
def maxima(data,offsets,widths,shape):
    lows=np.zeros(len(widths),np.uint64);highs=lows.copy()
    for j in prange(len(lows)):
        lo=hi=U(0)
        for k in range(shape[j,0]*shape[j,1]):
            vl,vh=load(data,offsets[j],widths[j],k)
            if vh>hi or (vh==hi and vl>lo):lo=vl;hi=vh
        lows[j]=lo;highs[j]=hi
    return lows,highs

@njit(cache=True,nogil=True)
def dot128(data,at,w,shape,indices,weights,blo,bhi):
    lo=hi=U(0)
    for i in range(shape[0]):
        for j in range(shape[1]):
            vl,vh=load(data,at,w,i*shape[1]+j)
            vl,vh=mul_small(vl,vh,U(weights[indices[i,j]]))
            lo,hi=plus(lo,hi,vl,vh)
            if hi>bhi or (hi==bhi and lo>blo):raise OverflowError('Terminal proved bound')
    return lo,hi
