"""Compiled bounded allocation batches, preserving per-block last-use release."""
import numpy as np
from numba import njit
import stream_packed as P
import split_wide as W

@njit(cache=True,nogil=True)
def execute(blocks,targets,order,begin,end,sizes,last_heads,last_next,empty,sw,dw,sshape,dshape,offsets,meta,indices,maps,threshold):
    for position in range(begin,end):
        d=order[position];dest=np.zeros(sizes[d],np.uint8);targets[d]=dest
        ops=meta[offsets[d]:offsets[d+1]];narrow=dw[d]<=8
        if narrow:
            for op in ops:
                if sw[op[0]]>8:narrow=False;break
        if narrow:P.execute(blocks,dest,sw,dw,sshape,dshape,ops,indices,maps,threshold)
        else:
            for op in ops:
                s=op[0];src=blocks[s];count=(op[3]-op[2])*(op[5]-op[4])
                if count<threshold or dw[d]!=16:
                    W.serial(src,dest,0,0,sw[s],dw[d],sshape[s,1],dshape[d,1],op,indices,maps)
                elif op[6]==2:
                    W.parallel_cols(src,dest.view(np.uint64),0,0,sw[s],dw[d],sshape[s,1],dshape[d,1],op,indices,maps)
                else:
                    W.parallel_rows(src,dest.view(np.uint64),0,0,sw[s],dw[d],sshape[s,1],dshape[d,1],op,indices,maps)
        s=last_heads[position]
        while s!=-1:blocks[s]=empty;s=last_next[s]
