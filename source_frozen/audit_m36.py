"""M36 domain checks: no M36 coefficient arrays or large maps are allocated."""
import math
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
from numba import njit

@njit(cache=True)
def virtual_probe(view, at):
    return len(view), np.int64(at), view[at]

def run_audit():
    sys.path.insert(0,str(Path(__file__).resolve().parent/'engine'))
    from server_meander import dimensions
    from deterministic import Compiler
    from weighted import Operators
    from grammar_min import minimize
    from target_geometry import assign_groups, density_le
    import linear_support as L
    import fast_maps
    import psutil
    d=dimensions(36); C=int(d['catalan']); G=C*C; R=d['max_ballot_rank_count']
    assert C==11959798385860453492 and C>2**63-1
    assert G==143036777430430308793718801051894994064 and G.bit_length()==127
    assert R==1771605360 and R<2**31 and 2*R>2**31 and 2*R<2**63
    assert d['closemap_entries_including_identity']==10190696803
    assert d['closemap_entries_including_identity']>2**33
    compiler=Compiler(36,range(1,19),Operators(8))
    assert compiler.lower_count((0,0,0,0,0,0))==C
    g=compiler.compile(); compact=minimize(g)
    assert g['nodes']<2**15 and compact['nodes']<2**15
    assert int(g['weights'].max())<2**16
    assert g['targets'].dtype==np.uint32 and compact['heights'].dtype==np.uint8
    assert 2*(2*36//3)==48 and 48+1+math.ceil(math.log2(g['nodes']))<=64
    # A stride-zero view tests 64-bit lengths and offsets without reserving GB.
    tiny=np.asarray([79],np.int32)
    view=np.lib.stride_tricks.as_strided(tiny,shape=(2*R,),strides=(0,),writeable=False)
    length,at,value=virtual_probe(view,2*R-1)
    assert length==2*R and at==2*R-1 and value==79
    assert len(view[2**31:])==2*R-2**31
    dims=np.asarray([[R,1],[1,R],[R//2,R//2],[R//4,R//4]],np.int64)
    costs=np.asarray([2*R,2*R,R,R//2],np.int64)
    assignment,mr,mc,loads=assign_groups(dims,costs,16,R,R,True)
    assert sum(map(int,mr))+sum(map(int,mc))<=2*R and np.all(assignment>=0)
    for p in (99,100,125,127,4294967295):
        assert density_le(R,R,R,R,p)==(100*R*R<=p*R*R)
    # M36 ranks are valid uint32; arena addresses and map offsets stay int64.
    offsets=np.asarray([2**31+17,2*R-1,d['closemap_entries_including_identity']-1],np.int64)
    assert list(map(int,offsets))==[2147483665,3543210719,10190696802]
    low_vm=psutil.virtual_memory()._replace(available=0)
    with patch.object(psutil,'virtual_memory',return_value=low_vm), patch.object(
        fast_maps,'maps',side_effect=AssertionError('Large maps must not be allocated')):
        try:L.calculate(36,workers=1,geometry_mode='target',parallel_geometry=1)
        except MemoryError:pass
        else:raise AssertionError('M36 fresh-memory preflight did not stop')
        try:L.calculate(37,workers=1,geometry_mode='target',parallel_geometry=1)
        except MemoryError:pass
        else:raise AssertionError('M37 fresh-memory preflight did not stop')
    d37=dimensions(37)
    assert d37['max_ballot_rank_count']==3511574910
    assert d37['bound_bits']==131 and d37['rank_value_dtype']=='uint32'
    return dict(status='PASS',**d,grammar_nodes_original=g['nodes'],
        grammar_nodes_minimized=compact['nodes'],maximum_grammar_weight=int(g['weights'].max()),
        weighted_lower_coverage=str(C),joint_key_width_bits=48,
        virtual_64bit_length_test=True,no_large_arrays_allocated=True,
        insufficient_memory_rejected=True,M37_dimensions_accepted=True,
        M37_insufficient_memory_rejected=True,full_M36_executed=False)

if __name__=='__main__':
    import json
    from server_meander import configure,HERE
    configure(2,HERE/'.numba_cache')
    print(json.dumps(run_audit(),indent=2))
