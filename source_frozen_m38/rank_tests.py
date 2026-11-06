"""Large-rank tests use tiny cropped axes, not multi-billion-entry arrays."""
import bisect
from common import ballot

def run():
    import numpy as np
    from numba.typed import List
    import axis_plan as A,bitmap_geometry as B,factor_metadata as M,reference_geometry as R
    import fast_maps as F,map_ranges as MR,dense_geometry as G,split_wide as W
    checks=0;origin=(1<<32)+17
    def axes(values):
        x=List.empty_list(A.ARR);x.append(np.asarray(values,np.int64));return x
    values=[origin,origin+2,origin+65]
    source=axes(values);terms=np.asarray([[0,0,3,0,0,0]],np.int64);reps=np.asarray([[0,0]],np.int64)
    class_terms=List();class_terms.append(np.asarray([0],np.int64))
    outbits=np.zeros(2,np.uint64)
    result=B.support_group(np.asarray([0],np.int64),terms,reps,class_terms,
        np.asarray([origin+100],np.int64),np.asarray([origin],np.int64),np.asarray([66],np.int64),np.zeros(1,np.uint8),
        source,source,np.empty(0,np.int64),np.zeros(1,np.uint64),np.asarray([-1],np.int64),np.zeros(1,np.int64),
        outbits,np.zeros(1,np.int64),np.zeros(2,np.uint64))
    assert np.array_equal(result[0],np.asarray(values,np.int64))and result[0].dtype==np.int64;checks+=3
    byaxis=List();byaxis.append(np.asarray([0],np.int64));flat=np.empty(3,np.int64)
    B.translate_group(np.asarray([0],np.int64),result,np.zeros(1,np.int64),np.asarray([2],np.int64),np.asarray([origin],np.int64),outbits,
        np.asarray([0,3],np.int64),np.asarray([0],np.int64),byaxis,terms,reps,source,source,np.empty(0,np.int64),flat,np.empty(3,np.int64))
    assert np.array_equal(flat,np.arange(3,dtype=np.int64));checks+=3
    for value in (origin-1,origin,origin+1,origin+2,origin+65,origin+66):
        assert R.lower(result[0],value)==bisect.bisect_left(values,value);checks+=1
    for k,value in enumerate(values):assert M.rank_at(result[0],value)==k;checks+=1
    try:M.rank_at(result[0],origin+3)
    except ArithmeticError:checks+=1
    else:raise AssertionError('Missing rank accepted')
    short=axes([0,1,2]);mapped_terms=terms.copy();mapped_terms[0,3]=1
    mapped=B.support_group(np.asarray([0],np.int64),mapped_terms,reps,class_terms,
        np.asarray([origin+100],np.int64),np.asarray([origin],np.int64),np.asarray([66],np.int64),np.zeros(1,np.uint8),
        short,short,np.asarray(values,np.int64),np.zeros(1,np.uint64),np.asarray([-1],np.int64),np.zeros(1,np.int64),
        outbits,np.zeros(1,np.int64),np.zeros(2,np.uint64))
    assert np.array_equal(mapped[0],result[0]);checks+=3
    contiguous=axes([origin,origin+1,origin+2])
    contiguous_result=B.support_group(np.asarray([0],np.int64),terms,reps,class_terms,
        np.asarray([origin+100],np.int64),np.asarray([origin],np.int64),np.asarray([66],np.int64),np.zeros(1,np.uint8),
        contiguous,contiguous,np.empty(0,np.int64),np.zeros(1,np.uint64),np.asarray([-1],np.int64),np.zeros(1,np.int64),
        outbits,np.zeros(1,np.int64),np.zeros(2,np.uint64))
    assert np.array_equal(contiguous_result[0],contiguous[0]);checks+=3
    canonical,*rest=A.canonicalize(contiguous_result,np.zeros(1,np.int64),np.zeros(1,np.int64),np.empty(0,np.int64))
    assert np.array_equal(canonical[0],contiguous[0]);checks+=3

    def close_python(m,h,rank):
        word=0;level=h;k=rank
        for pos in range(m-1,-1,-1):
            up=ballot(pos,level-1)if level else 0
            if k<up:word|=1<<pos;level-=1
            else:k-=up;level+=1
        stack=[]
        for pos in range(m):
            if word>>pos&1:stack.append(pos)
            else:stack.pop()
        assert len(stack)==h
        word^=1<<stack[-1];level=h-2;answer=0
        for mm in range(m,0,-1):
            if word>>(mm-1)&1:level-=1
            else:
                answer+=ballot(mm-1,level-1)if level else 0;level+=1
        return answer
    large=[]
    for m,h in ((36,6),(37,5),(38,6),(38,4)):
        count=ballot(m,h)
        points=sorted({0,1,count-1,*[k for k in ((1<<31)-1,1<<31,(1<<32)-1,1<<32,(1<<32)+7)if k<count]})
        for rank in points:
            expected=close_python(m,h,rank);actual=int(F.value_at(m,h,rank,G.TAB))
            assert actual==expected and 0<=actual<ballot(m,h-2),(m,h,rank,actual,expected)
            large.append(dict(m=m,h=h,input_rank=rank,output_rank=actual));checks+=1
    low,high=MR.tables(38,G.TAB)
    for left in ((1<<31)-1,(1<<32)-1,1<<32,ballot(38,6)-3):
        right=left+3;expected=[close_python(38,6,k)for k in range(left,right)]
        lo,hi=MR.query(38,6,left,right,G.TAB,low,high)
        assert(int(lo),int(hi))==(min(expected),max(expected));checks+=1
    for r,c,lr,lc,percent in ((6768687870,2,6000000000,2,125),(6768687870,2,1,1,100)):
        assert bool(R.density_le(r,c,lr,lc,percent))==(100*r*c<=percent*lr*lc);checks+=1
    max64=(1<<64)-1;meta=np.zeros((1,14),np.int64);meta[0,12:14]=1
    lo,hi=W.caps_exact_128(np.asarray([max64],np.uint64),np.asarray([max64],np.uint64),meta,1)
    assert int(lo[0])==int(hi[0])==max64;checks+=1
    meta[0,12]=2
    try:W.caps_exact_128(np.asarray([max64],np.uint64),np.asarray([max64],np.uint64),meta,1)
    except OverflowError:checks+=1
    else:raise AssertionError('128-bit overflow accepted')
    return dict(status='PASS',checks=checks,large_rank_cases=large,maximum_tested_rank=max(x['input_rank']for x in large),
        maximum_tested_map_value=max(x['output_rank']for x in large),no_large_rank_arena_allocated=True,
        rank_dtype='int64',index_dtype='int64',coefficient_caps_checked=True)
