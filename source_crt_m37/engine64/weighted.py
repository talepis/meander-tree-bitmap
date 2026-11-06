"""Exact weighted local completion operators; fixed small horizon."""
import numpy as np
from terminal_table import get
class Operators:
 def __init__(self,R,capacity=1024):
  self.R=R;self.lookup,self.literal,self.meta=get(R);self.ids={};self.rows=np.empty((capacity,self.literal.shape[1]),np.uint32)
 def register(self,language):
  if language in self.ids:return self.ids[language]
  if len(self.ids)>=len(self.rows):raise MemoryError('Bounded weighted terminal cache')
  row=np.zeros(self.literal.shape[1],np.uint64)
  for word,weight in language:row+=self.literal[word,:].astype(np.uint64)*weight
  if row.max(initial=0)>=2**32:raise OverflowError('Weighted local completion row exceeds uint32')
  j=len(self.ids);self.ids[language]=j;self.rows[j]=row;return j
