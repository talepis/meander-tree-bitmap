"""Per-block certified integer widths in an aligned byte arena."""
import numpy as np
from numba import njit,prange
import arena_kernels as A
@njit(cache=True,nogil=True)
def layout(caps,shape):
 offsets=np.zeros(len(caps)+1,np.int64);width=np.zeros(len(caps),np.uint8)
 for j in range(len(caps)):
  c=caps[j];w=1 if c<256 else (2 if c<65536 else (4 if c<4294967296 else 8));width[j]=w;offsets[j+1]=offsets[j]+((shape[j,0]*shape[j,1]*w+7)//8)*8
 return offsets,width
@njit(cache=True,nogil=True,parallel=True)
def maxima(data,offsets,widths,shape):
 out=np.zeros(len(widths),np.uint64)
 for j in prange(len(out)):
  count=shape[j,0]*shape[j,1];at=offsets[j];w=widths[j]
  if w==1:out[j]=data[at:at+count].max()
  elif w==2:out[j]=data.view(np.uint16)[at//2:at//2+count].max()
  elif w==4:out[j]=data.view(np.uint32)[at//4:at//4+count].max()
  else:out[j]=data.view(np.uint64)[at//8:at//8+count].max()
 return out
@njit(cache=True,nogil=True)
def apply_target(V,target,at,w,nr,nc,op,indices,threshold):
 count=nr*nc
 if w==1:
  T1=target[at:at+count].reshape(nr,nc);transfer(T1,V,op,indices,threshold)
 elif w==2:
  T2=target.view(np.uint16)[at//2:at//2+count].reshape(nr,nc);transfer(T2,V,op,indices,threshold)
 elif w==4:
  T4=target.view(np.uint32)[at//4:at//4+count].reshape(nr,nc);transfer(T4,V,op,indices,threshold)
 else:
  T8=target.view(np.uint64)[at//8:at//8+count].reshape(nr,nc);transfer(T8,V,op,indices,threshold)
@njit(cache=True,nogil=True)
def transfer(T,V,op,indices,threshold):
 if op[6]==0:
  if V.size>=threshold:A.rect_parallel(T,V,op[7],op[8],op[12])
  else:A.rect(T,V,op[7],op[8],op[12])
 else:
  ri=indices[op[9]:op[9]+V.shape[0]];ci=indices[op[10]:op[10]+V.shape[1]]
  if V.size>=threshold:
   if op[6]==2:A.indexed_cols(T,V,ri,ci,op[12])
   else:A.indexed_rows(T,V,ri,ci,op[12])
  else:A.indexed(T,V,ri,ci,op[12])
@njit(cache=True,nogil=True)
def execute(source,target,soff,doff,swidth,dwidth,sshape,dshape,meta,indices,threshold):
 for op in meta:
  si=op[0];di=op[1];at=soff[si];count=sshape[si,0]*sshape[si,1];w=swidth[si]
  if w==1:
   S1=source[at:at+count].reshape(sshape[si,0],sshape[si,1]);V1=S1[op[2]:op[3],op[4]:op[5]];apply_target(V1,target,doff[di],dwidth[di],dshape[di,0],dshape[di,1],op,indices,threshold)
  elif w==2:
   S2=source.view(np.uint16)[at//2:at//2+count].reshape(sshape[si,0],sshape[si,1]);V2=S2[op[2]:op[3],op[4]:op[5]];apply_target(V2,target,doff[di],dwidth[di],dshape[di,0],dshape[di,1],op,indices,threshold)
  elif w==4:
   S4=source.view(np.uint32)[at//4:at//4+count].reshape(sshape[si,0],sshape[si,1]);V4=S4[op[2]:op[3],op[4]:op[5]];apply_target(V4,target,doff[di],dwidth[di],dshape[di,0],dshape[di,1],op,indices,threshold)
  else:
   S8=source.view(np.uint64)[at//8:at//8+count].reshape(sshape[si,0],sshape[si,1]);V8=S8[op[2]:op[3],op[4]:op[5]];apply_target(V8,target,doff[di],dwidth[di],dshape[di,0],dshape[di,1],op,indices,threshold)
