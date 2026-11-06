"""One typed numeric arena; compiled transfer schedule and checked per-block bounds."""
import numpy as np
from numba import njit,prange
U=np.uint64
@njit(cache=True,nogil=True,parallel=True)
def maxima(data,offsets):
 out=np.zeros(len(offsets)-1,np.uint64)
 for b in prange(len(out)):
  v=U(0)
  for j in range(offsets[b],offsets[b+1]):v=max(v,data[j])
  out[b]=v
 return out
@njit(cache=True,nogil=True)
def caps(mx,meta,blocks):
 out=np.zeros(blocks,np.uint64);limit=U(18446744073709551615)
 for op in meta:
  source=op[0];dest=op[1];factor=U(op[12])*U(op[13]);value=mx[source]
  if value and factor>limit//value:return out,2
  term=value*factor
  if term>limit-out[dest]:return out,2
  out[dest]+=term
 return out,0
@njit(cache=True,nogil=True)
def rect(T,V,r,c,w):
 if w==1:
  for i in range(V.shape[0]):
   for j in range(V.shape[1]):T[r+i,c+j]+=V[i,j]
 else:
  for i in range(V.shape[0]):
   for j in range(V.shape[1]):T[r+i,c+j]+=V[i,j]*U(w)
@njit(cache=True,nogil=True,parallel=True)
def rect_parallel(T,V,r,c,w):
 if w==1:
  for i in prange(V.shape[0]):
   for j in range(V.shape[1]):T[r+i,c+j]+=V[i,j]
 else:
  for i in prange(V.shape[0]):
   for j in range(V.shape[1]):T[r+i,c+j]+=V[i,j]*U(w)
@njit(cache=True,nogil=True)
def indexed(T,V,ri,ci,w):
 if w==1:
  for i in range(V.shape[0]):
   for j in range(V.shape[1]):T[ri[i],ci[j]]+=V[i,j]
 else:
  for i in range(V.shape[0]):
   for j in range(V.shape[1]):T[ri[i],ci[j]]+=V[i,j]*U(w)
@njit(cache=True,nogil=True,parallel=True)
def indexed_rows(T,V,ri,ci,w):
 if w==1:
  for i in prange(V.shape[0]):
   for j in range(V.shape[1]):T[ri[i],ci[j]]+=V[i,j]
 else:
  for i in prange(V.shape[0]):
   for j in range(V.shape[1]):T[ri[i],ci[j]]+=V[i,j]*U(w)
@njit(cache=True,nogil=True,parallel=True)
def indexed_cols(T,V,ri,ci,w):
 if w==1:
  for j in prange(V.shape[1]):
   for i in range(V.shape[0]):T[ri[i],ci[j]]+=V[i,j]
 else:
  for j in prange(V.shape[1]):
   for i in range(V.shape[0]):T[ri[i],ci[j]]+=V[i,j]*U(w)
@njit(cache=True,nogil=True)
def execute(source,target,soff,doff,sshape,dshape,meta,indices,threshold):
 for op in meta:
  si=op[0];di=op[1];S=source[soff[si]:soff[si+1]].reshape(sshape[si,0],sshape[si,1]);T=target[doff[di]:doff[di+1]].reshape(dshape[di,0],dshape[di,1]);V=S[op[2]:op[3],op[4]:op[5]];kind=op[6];w=op[12]
  if kind==0:
   if V.size>=threshold:rect_parallel(T,V,op[7],op[8],w)
   else:rect(T,V,op[7],op[8],w)
  else:
   ri=indices[op[9]:op[9]+V.shape[0]];ci=indices[op[10]:op[10]+V.shape[1]]
   if V.size>=threshold:
    if kind==2:indexed_cols(T,V,ri,ci,w)
    else:indexed_rows(T,V,ri,ci,w)
   else:indexed(T,V,ri,ci,w)
