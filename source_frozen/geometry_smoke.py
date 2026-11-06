import sys,json,random
from pathlib import Path
import numpy as np
from numba import njit
H=Path(__file__).resolve().parent;O=H/'verification';O.mkdir(exist_ok=True)
sys.path.insert(0,str(H/'engine'))
import bitmap_geometry as B,map_ranges as M,dense_geometry as G,fast_maps as F
@njit(cache=True)
def primitive(x):return B.popcount(x),B.trailing(x)

@njit(cache=True)
def scalar_close(m,h,rank,tab):
    word=G.unrank(m,h,rank,tab);stack=np.empty(m,np.int64);depth=0
    for j in range(m):
        if word>>np.uint64(j)&np.uint64(1):stack[depth]=j;depth+=1
        else:depth-=1
    word^=np.uint64(1)<<np.uint64(stack[depth-1]);level=h-2;answer=np.int64(0)
    for mm in range(m,0,-1):
        if word>>np.uint64(mm-1)&np.uint64(1):level-=1
        else:answer+=tab[mm-1,level-1] if level else 0;level+=1
    return answer

def unpack(a):return {64*k+j for k,w in enumerate(a) for j in range(64) if int(w)>>j&1}

def run_checks():
    rng=random.Random(36110407)
    for _ in range(2000):
        x=rng.randrange(1,2**64);p,t=primitive(np.uint64(x))
        assert int(p)==x.bit_count() and int(t)==(x&-x).bit_length()-1
    for _ in range(500):
        dim=rng.randrange(1,600);bits=np.zeros((dim+63)//64,np.uint64);lo=rng.randrange(dim);hi=rng.randrange(lo+1,dim+1)
        B.set_interval(bits,lo,hi);assert unpack(bits)==set(range(lo,hi))
        src=np.zeros(5,np.uint64);values={rng.randrange(300) for j in range(100)}
        for v in values:src[v//64]|=np.uint64(1)<<np.uint64(v%64)
        lo=rng.randrange(300);hi=rng.randrange(lo+1,301);shift=rng.randrange(-lo,201)
        dst=np.zeros(8,np.uint64);B.shifted_or(dst,src,0,lo,hi,shift)
        assert unpack(dst)=={v+shift for v in values if lo<=v<hi}
    
    low,high=M.tables(36,G.TAB);off,lens,mult,data=F.maps(19);range_checks=0
    for m in range(2,19):
        for h in range(2,m+1):
            count=int(lens[m,h])
            if not count:continue
            a=data[int(off[m,h]):int(off[m,h])+count]
            for _ in range(40):
                left=rng.randrange(count);right=rng.randrange(left+1,count+1)
                lo,hi=M.query(m,h,left,right,G.TAB,low,high)
                assert (int(lo),int(hi))==(int(a[left:right].min()),int(a[left:right].max()))
                range_checks+=1
    large_checks=0
    for _ in range(2000):
        m=rng.randrange(19,37);valid=[h for h in range(2,m+1) if G.B(m,h)];h=rng.choice(valid);count=G.B(m,h)
        left=rng.randrange(count);right=min(count,left+rng.randrange(1,65))
        expected=[int(scalar_close(m,h,j,G.TAB)) for j in range(left,right)]
        lo,hi=M.query(m,h,left,right,G.TAB,low,high)
        assert (int(lo),int(hi))==(min(expected),max(expected));large_checks+=1
    
    maxB=1771605360
    virtual=np.lib.stride_tricks.as_strided(np.asarray([np.uint64(2**64-1)]),shape=(maxB,),strides=(0,),writeable=False)
    out=np.zeros(2,np.uint64);B.shifted_or(out,virtual,maxB-2,0,128,0)
    assert all(int(v)==2**64-1 for v in out)
    for _ in range(200):
        spans=np.asarray([rng.randrange(1,maxB+1) for j in range(30)],np.int64)
        identities=np.asarray([rng.randrange(2) for j in spans],np.uint8);work=spans.copy();capacity=rng.randrange(1,maxB)
        offsets,words,used=B.cache_layout(spans,identities,work,capacity)
        intervals=sorted((int(o),int(o+w)) for o,w in zip(offsets,words) if o>=0)
        assert used<=capacity and all(0<=a<b<=capacity for a,b in intervals)
        assert all(intervals[j][1]<=intervals[j+1][0] for j in range(len(intervals)-1))
    receipt=dict(status='PASS',popcount_and_trailing_checks=2000,bitmap_interval_and_shift_checks=1000,
        exact_map_interval_checks=range_checks,large_order_map_interval_checks=large_checks,
        scalar_oracle_max_order=36,cache_partition_checks=200,virtual_large_offset_check=True,
        full_M36_executed=False,no_large_arrays_allocated=True)
    return receipt

if __name__ == "__main__":
    receipt=run_checks()
    (O/"PRIMITIVE_QA.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))
