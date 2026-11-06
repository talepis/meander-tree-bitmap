"""Monotonic wall clock callable from nopython profiling code; no object mode."""
import ctypes
import os
from llvmlite import ir, binding
from numba import types
from numba.core import cgutils
from numba.extending import intrinsic

WINDOWS = os.name == 'nt'
if WINDOWS:
    _clock = ctypes.windll.kernel32.QueryPerformanceCounter
    _freq = ctypes.c_longlong()
    if not ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(_freq)):
        raise OSError('QueryPerformanceFrequency failed')
    FREQUENCY = float(_freq.value)
else:
    _clock = ctypes.CDLL(None).clock_gettime
    FREQUENCY = 1e9
binding.add_symbol('meander_profile_monotonic', ctypes.cast(_clock, ctypes.c_void_p).value)

@intrinsic
def read_ticks(typingctx):
    def codegen(context, builder, signature, args):
        i32 = ir.IntType(32)
        i64 = ir.IntType(64)
        slot = cgutils.alloca_once(builder, i64, size=ir.Constant(i32, 2))
        ftype = ir.FunctionType(i32, [i64.as_pointer()] if WINDOWS else [i32, i64.as_pointer()])
        function = builder.module.globals.get('meander_profile_monotonic')
        if function is None:
            function = ir.Function(builder.module, ftype, name='meander_profile_monotonic')
        builder.call(function, [slot] if WINDOWS else [ir.Constant(i32, 1), slot])
        first = builder.load(slot)
        if WINDOWS:
            return first
        second = builder.load(builder.gep(slot, [ir.Constant(i32, 1)]))
        return builder.add(builder.mul(first, ir.Constant(i64, 1000000000)), second)
    return types.int64(), codegen
