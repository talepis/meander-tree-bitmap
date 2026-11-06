"""Disjoint-target streaming transfers in three coprime residue rings."""
import numpy as np
from numba import njit, prange
import tree_mod_arith as M


@njit(inline='always')
def _one(blocks, targets, d, sshape, dshape, offsets, meta, indices, maps, mode):
    dst = targets[d].view(np.uint64)
    dnc = dshape[d, 1]
    for k in range(offsets[d], offsets[d + 1]):
        op = meta[k]
        s = op[0]
        src = blocks[s].view(np.uint64)
        snc = sshape[s, 1]
        if mode == 1:
            for i in range(op[3] - op[2]):
                for j in range(op[5] - op[4]):
                    M._cell_ones(src, dst, 0, 0, snc, dnc, op, indices, maps, i, j)
        elif mode == 2:
            for i in range(op[3] - op[2]):
                for j in range(op[5] - op[4]):
                    M._cell7(src, dst, 0, 0, snc, dnc, op, indices, maps, i, j)
        else:
            for i in range(op[3] - op[2]):
                for j in range(op[5] - op[4]):
                    M._cell64(src, dst, 0, 0, snc, dnc, op, indices, maps, i, j)


@njit(cache=True, nogil=True, parallel=True)
def compute_wave(blocks, targets, order, begin, end, sshape, dshape,
                 offsets, meta, indices, maps, mode):
    for p0 in prange(begin, end):
        d = order[np.int64(p0)]
        _one(blocks, targets, d, sshape, dshape, offsets, meta, indices, maps, mode)


@njit(cache=True, nogil=True)
def compute_serial(blocks, targets, order, begin, end, sshape, dshape,
                   offsets, meta, indices, maps, mode):
    for p in range(begin, end):
        d = order[p]
        _one(blocks, targets, d, sshape, dshape, offsets, meta, indices, maps, mode)


def compute_single(blocks, targets, d, sshape, dshape, offsets, meta,
                   indices, maps, mode, threshold):
    dst = targets[d].view(np.uint64)
    dnc = int(dshape[d, 1])
    for k in range(int(offsets[d]), int(offsets[d + 1])):
        op = meta[k]
        s = int(op[0])
        src = blocks[s].view(np.uint64)
        args = (src, dst, 0, 0, int(sshape[s, 1]), dnc, op, indices, maps)
        cells = int(op[3] - op[2]) * int(op[5] - op[4])
        if mode == 1:
            if cells < threshold:
                M._serial_ones(*args)
            elif op[6] == 2:
                M._cols_ones(*args)
            else:
                M._rows_ones(*args)
        elif mode == 2:
            if cells < threshold:
                M._serial7(*args)
            elif op[6] == 2:
                M._cols7(*args)
            else:
                M._rows7(*args)
        elif cells < threshold:
            M._serial64(*args)
        elif op[6] == 2:
            M._cols64(*args)
        else:
            M._rows64(*args)
