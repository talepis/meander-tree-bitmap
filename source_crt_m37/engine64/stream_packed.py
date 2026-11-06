import numpy as np
from numba import njit
from split_packed import apply_target
@njit(cache=True,nogil=True)
def execute(sources,target,swidth,dwidth,sshape,dshape,meta,indices,mapdata,threshold):
 for op in meta:
  si=op[0];di=op[1];source=sources[si];at=0;count=sshape[si,0]*sshape[si,1];w=swidth[si]
  if w==1:
   S1=source[at:at+count].reshape(sshape[si,0],sshape[si,1]);V1=S1[op[2]:op[3],op[4]:op[5]];apply_target(V1,target,0,dwidth[di],dshape[di,0],dshape[di,1],op,indices,mapdata,threshold)
  elif w==2:
   S2=source.view(np.uint16)[at//2:at//2+count].reshape(sshape[si,0],sshape[si,1]);V2=S2[op[2]:op[3],op[4]:op[5]];apply_target(V2,target,0,dwidth[di],dshape[di,0],dshape[di,1],op,indices,mapdata,threshold)
  elif w==4:
   S4=source.view(np.uint32)[at//4:at//4+count].reshape(sshape[si,0],sshape[si,1]);V4=S4[op[2]:op[3],op[4]:op[5]];apply_target(V4,target,0,dwidth[di],dshape[di,0],dshape[di,1],op,indices,mapdata,threshold)
  else:
   S8=source.view(np.uint64)[at//8:at//8+count].reshape(sshape[si,0],sshape[si,1]);V8=S8[op[2]:op[3],op[4]:op[5]];apply_target(V8,target,0,dwidth[di],dshape[di,0],dshape[di,1],op,indices,mapdata,threshold)
