"""Queued routed-expert replay on original BF16 operands and FP32 weights."""
import triton
import triton.language as tl
from dense_selected import _pair16


@triton.jit(do_not_specialize=['Routes'])
def routed_replay_kernel(X, W, RouteIds, RouteWeights, Counts, Indices,
                         Output, Debug, Invalid, Routes,
                         DOWN: tl.constexpr, BM: tl.constexpr,
                         WINDOW: tl.constexpr, CAPTURE_F32: tl.constexpr):
    N: tl.constexpr = 2048 if DOWN else 1024
    K: tl.constexpr = 512 if DOWN else 2048
    window = tl.program_id(1)
    count = tl.load(Counts + window).to(tl.int32)
    if count < 0 or count > WINDOW:
        if tl.program_id(0) == 0:
            tl.atomic_or(Invalid, 64, sem='relaxed')
        return
    offsets = tl.arange(0, BM)
    products = tl.arange(0, 16)
    for first in range(tl.program_id(0) * BM, count, tl.num_programs(0) * BM):
        candidate = first + offsets
        live = candidate < count
        index = tl.load(Indices + window * WINDOW + candidate, live, other=0).to(tl.int32)
        valid_index = (index >= 0) & (index < Routes * N)
        tl.atomic_or(Invalid, 64, mask=tl.sum((live & ~valid_index).to(tl.int32), 0) != 0, sem='relaxed')
        live &= valid_index
        route = index // N
        row = index % N
        expert = tl.load(RouteIds + route, live, other=0).to(tl.int32)
        valid_expert = (expert >= 0) & (expert < 256)
        tl.atomic_or(Invalid, 8, mask=tl.sum((live & ~valid_expert).to(tl.int32), 0) != 0, sem='relaxed')
        tl.store(Output + index, 0, live & ~valid_expert)
        live &= valid_expert
        input_row = route if DOWN else route // 8
        weight_row = expert.to(tl.int64) * N + row
        accumulator = tl.full((BM,), 0, tl.float32)
        for start in range(0, K, 16):
            k = start + products
            left = tl.load(X + input_row[:, None].to(tl.int64) * K + k[None, :], live[:, None], other=0)
            right = tl.load(W + weight_row[:, None] * K + k[None, :], live[:, None], other=0)
            accumulator = _pair16(left, right, accumulator)
        if DOWN:
            scale = tl.load(RouteWeights + route, live, other=0)
            scale_bits = scale.to(tl.uint32, bitcast=True)
            invalid_scale = ((scale_bits & 0x7f800000) == 0x7f800000) | (scale < 0) | (scale > 1)
            tl.atomic_or(Invalid, 32, mask=tl.sum((live & invalid_scale).to(tl.int32), 0) != 0, sem='relaxed')
            accumulator = accumulator * scale
        bits = accumulator.to(tl.uint32, bitcast=True)
        invalid_result = (bits & 0x7f800000) == 0x7f800000
        tl.atomic_or(Invalid, 32, mask=tl.sum((live & invalid_result).to(tl.int32), 0) != 0, sem='relaxed')
        tl.store(Output + index, accumulator.to(tl.bfloat16), live)
        if CAPTURE_F32:
            tl.store(Debug + index, accumulator, live)
