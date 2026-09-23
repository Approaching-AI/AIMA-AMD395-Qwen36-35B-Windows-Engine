"""Experimental tiled GDN U with the characterized SM121 BF16 accumulator.

Arithmetic follows q1_moe_hawkeye_bf16_accumulator.h (MIT Hawkeye adaptation).
This module is a component experiment and is not selected by the runtime.
"""
import triton
import triton.language as tl


@triton.jit
def _group16(left, right, accumulator):
    # left[M,16], right[N,16], accumulator[M,N].
    lb = left.to(tl.uint16, bitcast=True).to(tl.uint32)
    rb = right.to(tl.uint16, bitcast=True).to(tl.uint32)
    le = (lb >> 7) & 255
    re = (rb >> 7) & 255
    lm = (lb & 127) | tl.where(le != 0, 128, 0)
    rm = (rb & 127) | tl.where(re != 0, 128, 0)
    product = (lm[:, None, :] * rm[None, :, :]) << 9
    exponent = tl.maximum(le, 1).to(tl.int32)[:, None, :] + tl.maximum(re, 1).to(tl.int32)[None, :, :] - 254
    exponent = tl.where(product != 0, exponent, -133)
    negative = ((lb[:, None, :] ^ rb[None, :, :]) & 32768) != 0

    cb = accumulator.to(tl.uint32, bitcast=True)
    ce = (cb >> 23) & 255
    cm = (cb & 0x7fffff) | tl.where(ce != 0, 0x800000, 0)
    cexp = tl.where(cm == 0, -133, tl.where(ce == 0, -126, ce.to(tl.int32) - 127))
    maximum = tl.maximum(tl.max(exponent, axis=2), cexp)
    shift = maximum[:, :, None] - exponent
    # Each product magnitude is at most 255^2 * 2^11. All 16
    # products and the FP32 carry total at most 2,197,848,060 < 2^32.
    # Separate positive/negative sums therefore need no 64-bit arithmetic.
    aligned = (product << 2) >> tl.minimum(shift, 31)
    aligned = tl.where(shift < 32, aligned, 0).to(tl.uint32)
    cshift = maximum - cexp
    carry = (cm << 2) >> tl.minimum(cshift, 31)
    carry = tl.where(cshift < 32, carry, 0).to(tl.uint32)
    positive = tl.sum(tl.where(negative, 0, aligned), axis=2)
    negative_sum = tl.sum(tl.where(negative, aligned, 0), axis=2)
    carry_negative = (cb & 0x80000000) != 0
    positive += tl.where(carry_negative, 0, carry)
    negative_sum += tl.where(carry_negative, carry, 0)
    result_negative = positive < negative_sum
    magnitude = tl.where(result_negative, negative_sum - positive, positive - negative_sum).to(tl.uint32)

    # Conversion can round just below a power of two upward. The integer
    # comparison corrects that case, yielding exact integer bit width.
    fb = magnitude.to(tl.float32).to(tl.uint32, bitcast=True)
    width = tl.where(magnitude == 0, 0, ((fb >> 23) & 255).to(tl.int32) - 126)
    threshold = 1 << tl.maximum(width - 1, 0).to(tl.uint32)
    width -= tl.where((magnitude != 0) & (magnitude < threshold), 1, 0)
    normalized = tl.where(width > 26,
        magnitude >> tl.maximum(width - 26, 0),
        magnitude << tl.maximum(26 - width, 0))
    output_exp = maximum + width - 26
    underflow_shift = tl.maximum(-126 - output_exp, 0)
    normalized = tl.where(underflow_shift < 32,
        normalized >> tl.minimum(underflow_shift, 31), 0)
    output_exp = tl.maximum(output_exp, -126)
    significand = (normalized >> 2).to(tl.uint32)
    encoded_exp = output_exp + 127 - tl.where((significand & 0x800000) == 0, 1, 0)
    encoded = tl.where(result_negative, 0x80000000, 0).to(tl.uint32)
    encoded |= (encoded_exp.to(tl.uint32) & 255) << 23
    encoded |= significand & 0x7fffff
    return tl.where(significand == 0, 0, encoded).to(tl.uint32).to(tl.float32, bitcast=True)


@triton.jit(do_not_specialize=['T'])
def integer_u_kernel(A, V, Beta, U, Debug, T,
                     H: tl.constexpr, NV: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr,
                     CAPTURE_F32: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    columns = tl.program_id(1) * BN + tl.arange(0, BN)
    head = tl.program_id(2)
    chunk_start = (tl.program_id(0) * BM // 64) * 64
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for start in tl.static_range(0, 64, 16):
        k = start + tl.arange(0, 16)
        keys = chunk_start + k
        left = tl.load(A + (rows[:, None] * H + head) * 64 + k[None, :],
                       rows[:, None] < T, other=0)
        values = tl.load(V + (keys[None, :] * H + head) * NV + columns[:, None],
                         (keys[None, :] < T) & (columns[:, None] < NV), other=0)
        beta = tl.load(Beta + keys * H + head, keys < T, other=0).to(tl.float32)
        right = (values.to(tl.float32) * beta[None, :]).to(tl.bfloat16)
        accumulator = _group16(left, right, accumulator)
    offsets = (rows[:, None] * H + head) * NV + columns[None, :]
    mask = (rows[:, None] < T) & (columns[None, :] < NV)
    tl.store(U + offsets, accumulator.to(tl.bfloat16), mask)
    if CAPTURE_F32:
        tl.store(Debug + offsets, accumulator, mask)
