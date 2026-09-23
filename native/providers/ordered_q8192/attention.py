"""Experimental original-order q8192 attention components, outside the runtime.

Reuses qualified unsigned32 K16 arithmetic and the model-independent SM121
exp2/reciprocal tables. Host-side table validation is required before dispatch.
"""
import triton
import triton.language as tl
from integer_u import _group16
from dense_selected import _pair16
from ordered_pipeline import _exp


@triton.jit(do_not_specialize=['T', 'QS', 'QT'])
def original_qk_kernel(Q, K, Scores, T, QS, QT):
    # Original unified_attention_2d geometry: two queries, eight shared-KV
    # heads, thirty-two keys, one uninterrupted K256 BF16 dot.
    m = tl.arange(0, 16)
    n = tl.program_id(1) * 32 + tl.arange(0, 32)
    d = tl.arange(0, 256)
    local = tl.program_id(0) * 2 + m // 8
    head = tl.program_id(2) * 8 + m % 8
    left = tl.load(Q + ((QS + local[:, None]) * 16 + head[:, None]) * 256 + d[None, :], local[:, None] < QT, other=0)
    right = tl.load(K + (n[None, :] * 2 + tl.program_id(2)) * 256 + d[:, None], n[None, :] < T, other=0)
    scores = tl.full((16, 32), 0, tl.float32)
    scores += 0.0625 * tl.dot(left, right)
    scores = tl.where(n[None, :] <= QS + local[:, None], scores, -float('inf'))
    tl.store(Scores + (local[:, None] * 16 + head[:, None]) * T + n[None, :], scores,
             (local[:, None] < QT) & (n[None, :] < T))


@triton.jit(do_not_specialize=['T', 'QS', 'QT'])
def ordered_qk_kernel(Q, K, Scores, T, QS, QT, BM: tl.constexpr, BN: tl.constexpr):
    local = tl.program_id(0) * BM + tl.arange(0, BM)
    key = tl.program_id(1) * BN + tl.arange(0, BN)
    head = tl.program_id(2)
    accumulator = tl.full((BM, BN), 0, tl.float32)
    if tl.program_id(1) * BN <= QS + tl.minimum(tl.program_id(0) * BM + BM, QT) - 1:
        for start in tl.static_range(0, 256, 16):
            d = start + tl.arange(0, 16)
            left = tl.load(Q + ((QS + local[:, None]) * 16 + head) * 256 + d[None, :], local[:, None] < QT, other=0)
            right = tl.load(K + (key[:, None] * 2 + head // 8) * 256 + d[None, :], key[:, None] < T, other=0)
            accumulator = _group16(left, right, accumulator)
    result = tl.where(key[None, :] <= QS + local[:, None], accumulator * 0.0625, -float('inf'))
    tl.store(Scores + (local[:, None] * 16 + head) * T + key[None, :], result,
             (local[:, None] < QT) & (key[None, :] < T))


@triton.jit
def _sum32_original(probability):
    # Observed SM121 unified-attention row reduction, unchanged in the C++
    # reference-compatible probability implementation.
    for mask in tl.static_range(0, 5):
        shift = (1, 4, 2, 16, 8)[mask]
        indices = tl.broadcast_to((tl.arange(0, 32) ^ shift)[None, :], (probability.shape[0], 32))
        probability = probability + tl.gather(probability, indices, axis=1)
    return tl.sum(tl.where(tl.arange(0, 32)[None, :] == 0, probability, 0), axis=1)


@triton.jit(do_not_specialize=['T', 'QS', 'QT'])
def probability_kernel(Scores, Probability, Scales, Table, T, QS, QT,
                       BM: tl.constexpr, FUSED_L: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    head = tl.program_id(1)
    offsets = tl.arange(0, 32)
    tiles = tl.cdiv(T, 32)
    maximum = tl.full((BM,), -float('inf'), tl.float32)
    denominator = tl.full((BM,), 1, tl.float32)
    limit = tl.cdiv(QS + tl.minimum(tl.program_id(0) * BM + BM, QT), 32)
    for tile in range(0, limit):
        key = tile * 32 + offsets
        value = tl.load(Scores + (row[:, None] * 16 + head) * T + key[None, :],
                        (row[:, None] < QT) & (key[None, :] < T), other=-float('inf'))
        next_max = tl.maximum(maximum, tl.max(value, axis=1))
        next_max = tl.where(next_max > -float('inf'), next_max, 0)
        probability = _exp(Table, value - next_max[:, None])
        alpha = _exp(Table, maximum - next_max)
        subtotal = _sum32_original(probability)
        if FUSED_L:
            denominator = tl.fma(denominator, alpha, subtotal)
        else:
            denominator = denominator * alpha + subtotal
        maximum = next_max
        tl.store(Probability + (row[:, None] * 16 + head) * T + key[None, :], probability.to(tl.bfloat16),
                 (row[:, None] < QT) & (key[None, :] < T))
        tl.store(Scales + (row * 16 + head) * (tiles + 1) + tile, alpha, row < QT)
    tl.store(Scales + (row * 16 + head) * (tiles + 1) + tiles, denominator, row < QT)


@triton.jit
def _reciprocal(Table, denominator):
    bits = denominator.to(tl.uint32, bitcast=True)
    mantissa = bits & 0x7fffff
    exponent = ((bits >> 23) & 255).to(tl.int32) - 127
    normalized = (0x3f800000 | mantissa).to(tl.uint32).to(tl.float32, bitcast=True)
    rounded = tl.div_rn(1.0, normalized)
    delta = tl.load((Table + 32 + mantissa).to(tl.pointer_type(tl.int8))).to(tl.int32)
    result = rounded.to(tl.uint32, bitcast=True) + delta.to(tl.uint32) - (exponent.to(tl.uint32) << 23)
    valid = (bits < 0x80000000) & (exponent >= 0) & (exponent <= 18)
    return tl.where(valid, result, 0x7fc00000).to(tl.uint32).to(tl.float32, bitcast=True)


@triton.jit(do_not_specialize=['T', 'QS', 'QT'])
def selected_pv_kernel(Probability, Value, Scales, RcpTable, Counts, Indices,
                       Output, RawAccumulator, T, QS, QT, BM: tl.constexpr,
                       CAPTURE_F32: tl.constexpr):
    count = tl.load(Counts).to(tl.int32)
    offsets = tl.arange(0, BM)
    products = tl.arange(0, 16)
    stride = tl.cdiv(T, 32) + 1
    for first in range(tl.program_id(0) * BM, count, tl.num_programs(0) * BM):
        slot = first + offsets
        live = slot < count
        cell = tl.load(Indices + slot, live, other=0).to(tl.int32)
        dim = cell % 256
        row = cell // 256
        local = row // 16
        head = row % 16
        absolute = QS + local
        accumulator = tl.full((BM,), 0, tl.float32)
        tile_limit = tl.cdiv(tl.max(tl.where(live, absolute + 1, 0), axis=0), 32)
        for tile in range(0, tile_limit):
            alpha = tl.load(Scales + row * stride + tile, live & (tile * 32 <= absolute), other=1)
            accumulator = accumulator * alpha
            for half in tl.static_range(0, 2):
                key = tile * 32 + half * 16 + products
                probability = tl.load(Probability + row[:, None] * T + key[None, :],
                    live[:, None] & (key[None, :] <= absolute[:, None]) & (key[None, :] < T), other=0)
                value = tl.load(Value + (head[:, None] // 8 * 256 + dim[:, None]) * T + key[None, :],
                    live[:, None] & (key[None, :] < T), other=0)
                accumulator = _pair16(probability, value, accumulator)
        denominator = tl.load(Scales + row * stride + stride - 1, live, other=1)
        result = accumulator * _reciprocal(RcpTable, denominator)
        tl.store(Output + cell, result.to(tl.bfloat16), live)
        if CAPTURE_F32:
            tl.store(RawAccumulator + cell, accumulator, live)
