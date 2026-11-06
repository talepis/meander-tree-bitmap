import numpy as np
import reference_geometry as R
import axis_plan as A
import factor_supports as F

class Geometry(R.Geometry):
    def __init__(self,workers,split_maps=True):
        super().__init__(workers,split_maps)
        self.rowids=np.zeros(1,np.int64);self.colids=self.rowids.copy()
    def prepare(self,*args):
        n,step,skeys,mx,rs,cs,offsets,targets,labels,weights,heights,tab,mapoff,maplen,mapmult,mapdata=args[:16]
        keys,heads,descs=R.discover(n,step,skeys,mx,rs,cs,offsets,targets,labels,weights,heights,tab,mapoff,maplen,mapmult)
        p=A.make_plan(keys,heads,descs,self.rowids,self.colids,heights,tab,args[23])
        extra=F.costs(p[0],p[1],p[6],p[7],p[8],rs,cs)
        result=super().prepare(*args)
        axes,self.rowids,self.colids,raw,unique,raw_axes,unique_axes=A.diagnostic_outputs(result[4],result[5],p[4],p[5],p[0],p[2],p[3],heads,descs,result[2],mapdata)
        self.stats[-1].update(mapped_union_elements=int(extra[3]),interval_union_elements=int(extra[4]),nonlinear_union_elements=int(extra[5]),pure_interval_classes=int(extra[2].sum()),axis_classes=len(p[6]),canonical_axes=len(axes),raw_union_elements=int(p[-2]),unique_union_elements=int(p[-1]),
            raw_translation_elements=int(raw),unique_translation_elements=int(unique),raw_support_elements=int(raw_axes),unique_support_elements=int(unique_axes))
        return result
