from triton.experimental import gluon
from triton.experimental.gluon import language as tl

@gluon.jit
def _exp(Table, x):
    argument = x * 1.4426950408889634074
    bits = argument.to(tl.uint32, bitcast=True)
    magnitude = bits & 0x7fffffff
    negative = (bits & 0x80000000) != 0
    valid = negative & (magnitude >= 0x2f800000) & (magnitude < 0x43180000)
    relative = magnitude - 0x2f800000
    index = 12 + (relative >> 8) * 2
    table32 = Table.to(tl.pointer_type(tl.uint32))
    base = tl.load(table32 + index, valid, other=0)
    tag = tl.load(table32 + index + 1, valid, other=0)
    kind = tag >> 30
    payload = Table + 10272816 + (tag & 0x3fffffff)
    cell = relative & 255
    d1 = tl.load(payload + cell, valid & (kind == 1), other=0).to(tl.uint32)
    d2 = tl.load(payload.to(tl.pointer_type(tl.uint16)) + cell, valid & (kind == 2), other=0).to(tl.uint32)
    d4 = tl.load(payload.to(tl.pointer_type(tl.uint32)) + cell, valid & (kind == 3), other=0)
    result = base + d1 + d2 + d4
    small = (magnitude < 0x2f800000) | ((~negative) & (magnitude < 0x33000000))
    result = tl.where(magnitude >= 0x43180000, 0, result)
    result = tl.where((magnitude > 0x7f800000) | (~negative), 0x7fc00000, result)
    return tl.where(small, 0x3f800000, result).to(tl.uint32).to(tl.float32, bitcast=True)


