from reference_geometry import *
import os
MODE=os.environ.get('MEANDER_GEOMETRY_ENGINE','baseline')
def Geometry(workers,split_maps=True):
 if MODE=='baseline':return __import__('reference_geometry').Geometry(workers,split_maps)
 if MODE=='diagnostic':return __import__('geometry_diagnostic').Geometry(workers,split_maps)
 return __import__('factor_geometry').Geometry(workers,split_maps,MODE)
