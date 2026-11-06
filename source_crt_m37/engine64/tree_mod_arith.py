"""Three exact residue rings for the isolated tree/bitmap CRT experiment.

The geometry and transfer descriptors are inherited from the frozen tree
algorithm.  Only coefficient arithmetic is changed.  Both rings have fixed
eight-byte coefficients, so no exact-value-dependent packed-width decision
can leak into a modular pass.
"""
import numpy as np
from numba import njit, prange

U = np.uint64
MAX = U(0xffffffffffffffff)
MASK = U(0xffffffff)
MOD64 = 1 << 64
MODONES = MOD64 - 1
MODE = None


def configure(modulus):
    global MODE
    if modulus not in (MOD64, MODONES, 7):
        raise ValueError('Only the three audited, coprime residue rings are supported')
    MODE = modulus


def layout(caps, highs, shape):
    if MODE is None:
        raise RuntimeError('Configure the modular ring first')
    if len(caps) != len(shape) or len(highs) != len(shape):
        raise ValueError('Invalid modular block layout')
    widths = np.full(len(shape), 8, np.uint8)
    offsets = np.zeros(len(shape) + 1, np.int64)
    sizes = shape[:, 0] * shape[:, 1] * 8
    offsets[1:] = np.cumsum(sizes, dtype=np.int64)
    return offsets, widths


@njit(inline='always')
def _plus_ones(a, b):
    """Addition modulo 2**64-1 by an end-around carry."""
    out = a + b
    if out < a:
        out += U(1)
    if out == MAX:
        out = U(0)
    return out


@njit(inline='always')
def _times_ones(x, w):
    """Reduce the exact 96-bit product of two 64/32-bit factors."""
    a = (x & MASK) * w
    b = (x >> U(32)) * w
    middle = (a >> U(32)) + (b & MASK)
    low = (a & MASK) | ((middle & MASK) << U(32))
    high = (b >> U(32)) + (middle >> U(32))
    return _plus_ones(low, high)


@njit(inline='always')
def _positions(op, indices, maps, i, j):
    if op[6] == 0:
        return op[7] + i, op[8] + j
    rr = maps[op[9] + i] if op[11] & 1 else indices[op[9] + i]
    cc = maps[op[10] + j] if op[11] & 2 else indices[op[10] + j]
    return rr, cc


@njit(inline='always')
def _cell64(src, dst, sa, da, snc, dnc, op, indices, maps, i, j):
    rr, cc = _positions(op, indices, maps, i, j)
    dst[da + rr * dnc + cc] += src[sa + (op[2] + i) * snc + op[4] + j] * U(op[12])


@njit(inline='always')
def _cell_ones(src, dst, sa, da, snc, dnc, op, indices, maps, i, j):
    rr, cc = _positions(op, indices, maps, i, j)
    k = da + rr * dnc + cc
    x = src[sa + (op[2] + i) * snc + op[4] + j]
    dst[k] = _plus_ones(dst[k], _times_ones(x, U(op[12])))


@njit(inline='always')
def _cell7(src, dst, sa, da, snc, dnc, op, indices, maps, i, j):
    rr, cc = _positions(op, indices, maps, i, j)
    k = da + rr * dnc + cc
    x = src[sa + (op[2] + i) * snc + op[4] + j]
    # x < 7 and w <= 2**32-1, so this product and addition fit uint64.
    dst[k] = (dst[k] + x * U(op[12])) % U(7)


@njit(cache=True, nogil=True)
def _serial64(src, dst, sa, da, snc, dnc, op, indices, maps):
    for i in range(op[3] - op[2]):
        for j in range(op[5] - op[4]):
            _cell64(src, dst, sa, da, snc, dnc, op, indices, maps, i, j)


@njit(cache=True, nogil=True)
def _serial_ones(src, dst, sa, da, snc, dnc, op, indices, maps):
    for i in range(op[3] - op[2]):
        for j in range(op[5] - op[4]):
            _cell_ones(src, dst, sa, da, snc, dnc, op, indices, maps, i, j)


@njit(cache=True, nogil=True)
def _serial7(src, dst, sa, da, snc, dnc, op, indices, maps):
    for i in range(op[3] - op[2]):
        for j in range(op[5] - op[4]):
            _cell7(src, dst, sa, da, snc, dnc, op, indices, maps, i, j)


@njit(cache=True, nogil=True, parallel=True)
def _rows64(src, dst, sa, da, snc, dnc, op, indices, maps):
    for ii in prange(op[3] - op[2]):
        for j in range(op[5] - op[4]):
            _cell64(src, dst, sa, da, snc, dnc, op, indices, maps, np.int64(ii), j)


@njit(cache=True, nogil=True, parallel=True)
def _cols64(src, dst, sa, da, snc, dnc, op, indices, maps):
    for jj in prange(op[5] - op[4]):
        for i in range(op[3] - op[2]):
            _cell64(src, dst, sa, da, snc, dnc, op, indices, maps, i, np.int64(jj))


@njit(cache=True, nogil=True, parallel=True)
def _rows_ones(src, dst, sa, da, snc, dnc, op, indices, maps):
    for ii in prange(op[3] - op[2]):
        for j in range(op[5] - op[4]):
            _cell_ones(src, dst, sa, da, snc, dnc, op, indices, maps, np.int64(ii), j)


@njit(cache=True, nogil=True, parallel=True)
def _rows7(src, dst, sa, da, snc, dnc, op, indices, maps):
    for ii in prange(op[3] - op[2]):
        for j in range(op[5] - op[4]):
            _cell7(src, dst, sa, da, snc, dnc, op, indices, maps, np.int64(ii), j)


@njit(cache=True, nogil=True, parallel=True)
def _cols_ones(src, dst, sa, da, snc, dnc, op, indices, maps):
    for jj in prange(op[5] - op[4]):
        for i in range(op[3] - op[2]):
            _cell_ones(src, dst, sa, da, snc, dnc, op, indices, maps, i, np.int64(jj))


@njit(cache=True, nogil=True, parallel=True)
def _cols7(src, dst, sa, da, snc, dnc, op, indices, maps):
    for jj in prange(op[5] - op[4]):
        for i in range(op[3] - op[2]):
            _cell7(src, dst, sa, da, snc, dnc, op, indices, maps, i, np.int64(jj))


def execute(source, target, soff, doff, swidth, dwidth, sshape, dshape,
            meta, indices, maps, threshold):
    if MODE is None or np.any(swidth != 8) or np.any(dwidth != 8):
        raise RuntimeError('Modular transfer requires configured fixed-width arrays')
    src = source.view(np.uint64)
    dst = target.view(np.uint64)
    for op in meta:
        w = int(op[12])
        if not 0 <= w <= int(MASK):
            raise OverflowError('Audited 32-bit tree weight domain exceeded')
        si, di = int(op[0]), int(op[1])
        args = (src, dst, int(soff[si] // 8), int(doff[di] // 8),
                int(sshape[si, 1]), int(dshape[di, 1]), op, indices, maps)
        cells = int(op[3] - op[2]) * int(op[5] - op[4])
        if MODE == MOD64:
            if cells < threshold:
                _serial64(*args)
            elif op[6] == 2:
                _cols64(*args)
            else:
                _rows64(*args)
        elif MODE == 7:
            if cells < threshold:
                _serial7(*args)
            elif op[6] == 2:
                _cols7(*args)
            else:
                _rows7(*args)
        elif cells < threshold:
            _serial_ones(*args)
        elif op[6] == 2:
            _cols_ones(*args)
        else:
            _rows_ones(*args)


@njit(cache=True, nogil=True, parallel=True)
def _maxima(data, offsets, shape):
    values = data.view(np.uint64)
    out = np.zeros(len(shape), np.uint64)
    for j in prange(len(shape)):
        m = U(0)
        at = offsets[j] // 8
        for k in range(shape[j, 0] * shape[j, 1]):
            if values[at + k] > m:
                m = values[at + k]
        out[j] = m
    return out


def maxima(data, offsets, widths, shape):
    if np.any(widths != 8):
        raise RuntimeError('Modular maxima require fixed-width blocks')
    lo = _maxima(data, offsets, shape)
    return lo, np.zeros(len(lo), np.uint64)


@njit(cache=True, nogil=True)
def _dot64(data, at, shape, indices, weights):
    values = data.view(np.uint64)
    out = U(0)
    for i in range(shape[0]):
        for j in range(shape[1]):
            out += values[at + i * shape[1] + j] * U(weights[indices[i, j]])
    return out


@njit(cache=True, nogil=True)
def _dot_ones(data, at, shape, indices, weights):
    values = data.view(np.uint64)
    out = U(0)
    for i in range(shape[0]):
        for j in range(shape[1]):
            out = _plus_ones(out, _times_ones(values[at + i * shape[1] + j],
                                              U(weights[indices[i, j]])))
    return out


@njit(cache=True, nogil=True)
def _dot7(data, at, shape, indices, weights):
    values = data.view(np.uint64)
    out = U(0)
    for i in range(shape[0]):
        for j in range(shape[1]):
            out = (out + values[at + i * shape[1] + j] * U(weights[indices[i, j]])) % U(7)
    return out


def dot128(data, at, width, shape, indices, weights, bound_lo, bound_hi):
    if MODE is None or width != 8 or at % 8:
        raise RuntimeError('Modular terminal requires fixed-width blocks')
    if len(weights) and int(np.max(weights)) > int(MASK):
        raise OverflowError('Terminal weight exceeds audited 32-bit domain')
    result = (_dot64 if MODE == MOD64 else _dot_ones if MODE == MODONES else _dot7)(
        data, at // 8, shape, indices, weights)
    return result, U(0)
