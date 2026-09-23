"""Experimental tiled GDN operators using the existing exact scalar contracts.

No runtime selection. The host must verify the complete QEX2TBL1 artifact
against sm121_exp2_table.h before passing its device pointer.
"""
import triton
import triton.language as tl
from integer_u import _group16


@triton.jit
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


@triton.jit(do_not_specialize=['T'])
def gram_kernel(Q, K, Beta, G, Table, Out, T,
                BM: tl.constexpr, BN: tl.constexpr, SCORE: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    column = tl.program_id(1) * BN + tl.arange(0, BN)
    head = tl.program_id(2)
    chunk = (tl.program_id(0) * BM // 64) * 64
    source = chunk + column
    accumulator = tl.full((BM, BN), 0, tl.float32)
    if not SCORE:
        beta = tl.load(Beta + row * 32 + head, row < T, other=0)
    for start in tl.static_range(0, 128, 16):
        k = start + tl.arange(0, 16)
        left = tl.load(Q + (row[:, None] * 16 + head // 2) * 128 + k[None, :], row[:, None] < T, other=0)
        right = tl.load(K + (source[:, None] * 16 + head // 2) * 128 + k[None, :], source[:, None] < T, other=0)
        if not SCORE:
            left = (left.to(tl.float32) * beta[:, None]).to(tl.bfloat16)
        accumulator = _group16(left, right, accumulator)
    gr = tl.load(G + row * 32 + head, row < T, other=0)
    gc = tl.load(G + source * 32 + head, source < T, other=0)
    result = accumulator * _exp(Table, gr[:, None] - gc[None, :])
    triangle = row[:, None] >= source[None, :] if SCORE else row[:, None] > source[None, :]
    result = tl.where(triangle & (source[None, :] < T), result, 0)
    tl.store(Out + (row[:, None] * 32 + head) * 64 + column[None, :], result.to(Out.dtype.element_ty), row[:, None] < T)


@triton.jit(do_not_specialize=['T'])
def w_kernel(A, K, Beta, G, Table, W, T, BM: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    column = tl.program_id(1) * BN + tl.arange(0, BN)
    head = tl.program_id(2)
    chunk = (tl.program_id(0) * BM // 64) * 64
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for start in tl.static_range(0, 64, 16):
        k = start + tl.arange(0, 16)
        source = chunk + k
        left = tl.load(A + (row[:, None] * 32 + head) * 64 + k[None, :], row[:, None] < T, other=0)
        keys = tl.load(K + (source[None, :] * 16 + head // 2) * 128 + column[:, None], source[None, :] < T, other=0)
        beta = tl.load(Beta + source * 32 + head, source < T, other=0)
        gate = tl.load(G + source * 32 + head, source < T, other=0)
        kb = (keys.to(tl.float32) * beta[None, :]).to(tl.bfloat16)
        right = (kb.to(tl.float32) * _exp(Table, gate)[None, :]).to(tl.bfloat16)
        accumulator = _group16(left, right, accumulator)
    tl.store(W + (row[:, None] * 32 + head) * 128 + column[None, :], accumulator.to(tl.bfloat16), row[:, None] < T)


@triton.jit(do_not_specialize=['T'])
def residual_kernel(W, U, H0, G, Table, VNew, Residual, T,
                    BM: tl.constexpr, BN: tl.constexpr):
    # One recurrent chunk per invocation; H0 is its actual FP32 incoming state.
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    column = tl.program_id(1) * BN + tl.arange(0, BN)
    head = tl.program_id(2)
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for start in tl.static_range(0, 128, 16):
        k = start + tl.arange(0, 16)
        left = tl.load(W + (row[:, None] * 32 + head) * 128 + k[None, :], row[:, None] < T, other=0)
        right = tl.load(H0 + (head * 128 + column[:, None]) * 128 + k[None, :]).to(tl.bfloat16)
        accumulator = _group16(left, right, accumulator)
    offsets = (row[:, None] * 32 + head) * 128 + column[None, :]
    u = tl.load(U + offsets, row[:, None] < T, other=0).to(tl.float32)
    residual = u - accumulator
    gate = tl.load(G + row * 32 + head, row < T, other=0)
    last = tl.load(G + (T - 1) * 32 + head)
    scaled = residual * _exp(Table, last - gate)[:, None]
    tl.store(VNew + offsets, residual.to(tl.bfloat16), row[:, None] < T)
    tl.store(Residual + offsets, scaled.to(tl.bfloat16), row[:, None] < T)


@triton.jit(do_not_specialize=['T'])
def state_kernel(K, Residual, H0, G, Table, Final, T,
                 BM: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    column = tl.program_id(1) * BN + tl.arange(0, BN)
    head = tl.program_id(2)
    offsets = (head * 128 + row[:, None]) * 128 + column[None, :]
    initial = tl.load(H0 + offsets)
    last = tl.load(G + (T - 1) * 32 + head)
    decay = _exp(Table, last)
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for start in tl.static_range(0, 64, 16):
        k = start + tl.arange(0, 16)
        left = tl.load(Residual + (k[None, :] * 32 + head) * 128 + row[:, None], k[None, :] < T, other=0)
        right = tl.load(K + (k[None, :] * 16 + head // 2) * 128 + column[:, None], k[None, :] < T, other=0)
        accumulator = _group16(left, right, accumulator)
    # Original CUDA decay is fused after the independent K64 dot.
    tl.store(Final + offsets, tl.fma(initial, decay, accumulator))


@triton.jit(do_not_specialize=['T'])
def output_kernel(Q, VNew, H0, G, Scores, Table, Out, T,
                  BM: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    column = tl.program_id(1) * BN + tl.arange(0, BN)
    head = tl.program_id(2)
    old = tl.full((BM, BN), 0, tl.float32)
    local = tl.full((BM, BN), 0, tl.float32)
    for start in tl.static_range(0, 128, 16):
        k = start + tl.arange(0, 16)
        left = tl.load(Q + (row[:, None] * 16 + head // 2) * 128 + k[None, :], row[:, None] < T, other=0)
        right = tl.load(H0 + (head * 128 + column[:, None]) * 128 + k[None, :]).to(tl.bfloat16)
        old = _group16(left, right, old)
    for start in tl.static_range(0, 64, 16):
        k = start + tl.arange(0, 16)
        left = tl.load(Scores + (row[:, None] * 32 + head) * 64 + k[None, :], row[:, None] < T, other=0)
        right = tl.load(VNew + (k[None, :] * 32 + head) * 128 + column[:, None], k[None, :] < T, other=0)
        local = _group16(left, right, local)
    gate = tl.load(G + row * 32 + head, row < T, other=0)
    prior = old * _exp(Table, gate)[:, None]
    result = tl.fma(local, 0.08838834764831845, prior * 0.08838834764831845)
    tl.store(Out + (row[:, None] * 32 + head) * 128 + column[None, :], result.to(tl.bfloat16), row[:, None] < T)
