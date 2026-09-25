"""Exact BF16 value-plane transpose for the ordered attention component."""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T'])
def pack_value_kernel(Source, Packed, T):
    column = tl.program_id(0) * 32 + tl.arange(0, 32)
    token = tl.program_id(1) * 32 + tl.arange(0, 32)
    value = tl.load(Source + token[:, None] * 512 + column[None, :], token[:, None] < T, other=0)
    tl.store(Packed + column[None, :] * T + token[:, None], value, token[:, None] < T)
