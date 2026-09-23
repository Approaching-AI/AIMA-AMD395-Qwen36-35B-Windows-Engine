"""Explicit lane layouts for original-order q8192 attention diagnostics."""
from triton.experimental import gluon
from triton.experimental.gluon import language as tl
from explicit_group16 import _group16
from explicit_dense import _pair16
from explicit_exp2 import _exp


@gluon.jit
def _axes(BM: tl.constexpr, BN: tl.constexpr):
    base: tl.constexpr = tl.BlockedLayout([1, 1, 1], [1, 2, 16], [1, 4, 1], [2, 1, 0])
    left: tl.constexpr = tl.SliceLayout(1, base)
    right: tl.constexpr = tl.SliceLayout(0, base)
    result: tl.constexpr = tl.SliceLayout(2, base)
    ml = tl.arange(0, BM, layout=tl.SliceLayout(1, left))
    nr = tl.arange(0, BN, layout=tl.SliceLayout(1, right))
    kl = tl.arange(0, 16, layout=tl.SliceLayout(0, left))
    kr = tl.arange(0, 16, layout=tl.SliceLayout(0, right))
    mo = tl.arange(0, BM, layout=tl.SliceLayout(1, result))
    no = tl.arange(0, BN, layout=tl.SliceLayout(0, result))
    return ml, nr, kl, kr, mo, no


@gluon.jit
def _zero(BM: tl.constexpr, BN: tl.constexpr):
    base: tl.constexpr = tl.BlockedLayout([1, 1, 1], [1, 2, 16], [1, 4, 1], [2, 1, 0])
    return tl.full((BM, BN), 0, tl.float32, layout=tl.SliceLayout(2, base))


@gluon.jit(do_not_specialize=['T', 'QS', 'QT'])
def ordered_qk_kernel(Q, K, Scores, T, QS, QT, BM: tl.constexpr, BN: tl.constexpr):
    ml, nr, kl, kr, mo, no = _axes(BM, BN)
    ml += tl.program_id(0) * BM
    mo += tl.program_id(0) * BM
    nr += tl.program_id(1) * BN
    no += tl.program_id(1) * BN
    head = tl.program_id(2)
    accumulator = _zero(BM, BN)
    if tl.program_id(1) * BN <= QS + tl.minimum(tl.program_id(0) * BM + BM, QT) - 1:
        for start in tl.static_range(0, 256, 16):
            left = tl.load(Q + ((QS + ml[:, None]) * 16 + head) * 256 + start + kl[None, :], ml[:, None] < QT, other=0)
            right = tl.load(K + (nr[:, None] * 2 + head // 8) * 256 + start + kr[None, :], nr[:, None] < T, other=0)
            accumulator = _group16(left, right, accumulator)
    result = tl.where(no[None, :] <= QS + mo[:, None], accumulator * 0.0625, -float('inf'))
    tl.store(Scores + (mo[:, None] * 16 + head) * T + no[None, :], result,
             (mo[:, None] < QT) & (no[None, :] < T))


@gluon.jit
def _sum32_original(probability):
    base: tl.constexpr = tl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    # Observed SM121 unified-attention row reduction, unchanged in the C++
    # reference-compatible probability implementation.
    for mask in tl.static_range(0, 5):
        shift = (1, 4, 2, 16, 8)[mask]
        indices = tl.broadcast_to((tl.arange(0, 32, layout=tl.SliceLayout(0, base)) ^ shift)[None, :], (probability.shape[0], 32))
        probability = probability + tl.gather(probability, indices, axis=1)
    return tl.sum(tl.where(tl.arange(0, 32, layout=tl.SliceLayout(0, base))[None, :] == 0, probability, 0), axis=1)


@gluon.jit(do_not_specialize=['T', 'QS', 'QT'])
def probability_kernel(Scores, Probability, Scales, Table, T, QS, QT,
                       BM: tl.constexpr, FUSED_L: tl.constexpr):
    base: tl.constexpr = tl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    rows: tl.constexpr = tl.SliceLayout(1, base)
    row = tl.program_id(0) * BM + tl.arange(0, BM, layout=rows)
    head = tl.program_id(1)
    offsets = tl.arange(0, 32, layout=tl.SliceLayout(0, base))
    tiles = tl.cdiv(T, 32)
    maximum = tl.full((BM,), -float('inf'), tl.float32, layout=rows)
    denominator = tl.full((BM,), 1, tl.float32, layout=rows)
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


@gluon.jit
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


@gluon.jit(do_not_specialize=['T', 'QS', 'QT'])
def selected_pv_kernel(Probability, Value, Scales, RcpTable, Counts, Indices,
                       Output, RawAccumulator, T, QS, QT, BM: tl.constexpr,
                       CAPTURE_F32: tl.constexpr):
    count = tl.load(Counts).to(tl.int32)
    base: tl.constexpr = tl.BlockedLayout([1, 1], [2, 16], [4, 1], [1, 0])
    pairs: tl.constexpr = tl.SliceLayout(1, base)
    offsets = tl.arange(0, BM, layout=pairs)
    products = tl.arange(0, 16, layout=tl.SliceLayout(0, base))
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
        accumulator = tl.full((BM,), 0, tl.float32, layout=pairs)
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


@gluon.jit(do_not_specialize=['T', 'QS', 'QT'])
def tiled_pv_kernel(Probability, Value, Scales, RcpTable, Output, T, QS, QT,
                    BM: tl.constexpr, BN: tl.constexpr):
    ml, nr, kl, kr, mo, no = _axes(BM, BN)
    ml += tl.program_id(0) * BM
    mo += tl.program_id(0) * BM
    nr += tl.program_id(1) * BN
    no += tl.program_id(1) * BN
    head = tl.program_id(2)
    left_row = ml * 16 + head
    row = mo * 16 + head
    stride = tl.cdiv(T, 32) + 1
    accumulator = _zero(BM, BN)
    limit = tl.cdiv(QS + tl.minimum(tl.program_id(0) * BM + BM, QT), 32)
    for tile in range(0, limit):
        alpha = tl.load(Scales + row * stride + tile,
                        (mo < QT) & (tile * 32 <= QS + mo), other=1)
        accumulator = accumulator * alpha[:, None]
        for half in tl.static_range(0, 2):
            left_key = tile * 32 + half * 16 + kl
            right_key = tile * 32 + half * 16 + kr
            probability = tl.load(Probability + left_row[:, None] * T + left_key[None, :],
                (ml[:, None] < QT) & (left_key[None, :] <= QS + ml[:, None]) & (left_key[None, :] < T), other=0)
            value = tl.load(Value + (head // 8 * 256 + nr[:, None]) * T + right_key[None, :],
                (nr[:, None] < 256) & (right_key[None, :] < T), other=0)
            accumulator = _group16(probability, value, accumulator)
    denominator = tl.load(Scales + row * stride + stride - 1, mo < QT, other=1)
    result = accumulator * _reciprocal(RcpTable, denominator)[:, None]
    tl.store(Output + row[:, None] * 256 + no[None, :], result.to(tl.bfloat16),
             (mo[:, None] < QT) & (no[None, :] < 256))
