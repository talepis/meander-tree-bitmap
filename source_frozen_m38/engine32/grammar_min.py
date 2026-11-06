"""Exact bottom-up bisimulation of a deterministic acyclic weighted grammar."""
import numpy as np

def minimize(g):
    signatures={};which={};representatives=[]
    for v in range(g['nodes']-1,-1,-1):
        op=int(g['opids'][v]);h=int(g['heights'][v])
        if op>=0:sig=(h,op,())
        else:
            edges=tuple(sorted((int(g['labels'][e]),int(g['weights'][e]),which[int(g['targets'][e])])
                for e in range(int(g['offsets'][v]),int(g['offsets'][v+1]))))
            sig=(h,-1,edges)
        if sig not in signatures:
            signatures[sig]=len(representatives);representatives.append(v)
        which[v]=signatures[sig]
    # Reachable BFS order preserves stage layering and gives root ID zero.
    pending=[which[0]];ids={pending[0]:0};offsets=[0];targets=[];labels=[];weights=[];heights=[];ops=[]
    for cls in pending:
        v=representatives[cls];heights.append(g['heights'][v]);ops.append(g['opids'][v])
        for e in range(int(g['offsets'][v]),int(g['offsets'][v+1])):
            child=which[int(g['targets'][e])]
            if child not in ids:ids[child]=len(ids);pending.append(child)
            targets.append(ids[child]);labels.append(g['labels'][e]);weights.append(g['weights'][e])
        offsets.append(len(targets))
    out=dict(g);out.update(nodes=len(pending),edges=len(targets),offsets=np.asarray(offsets,np.uint32),
        targets=np.asarray(targets,np.uint32),labels=np.asarray(labels,np.uint8),weights=np.asarray(weights,np.uint16),
        heights=np.asarray(heights,np.uint8),opids=np.asarray(ops,np.int32))
    out['old_to_new']=np.asarray([ids[which[v]] for v in range(g['nodes'])],np.uint32)
    out['minimization']=dict(original_nodes=g['nodes'],minimized_nodes=len(pending),original_edges=g['edges'],minimized_edges=len(targets))
    return out
