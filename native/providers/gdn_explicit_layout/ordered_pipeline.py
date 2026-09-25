"""Experimental GDN kernels with a single explicit K16 reduction layout.

Every dot keeps its accumulator in SliceLayout(2, BASE); operand coordinates
have the matching parent slices. Arithmetic follows the original ordered
pipeline. No model runtime selects these diagnostic kernels.
"""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from group16 import _group16
from exp2 import _exp


@gluon.jit
def _axes(BM: gl.constexpr, BN: gl.constexpr):
    base: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 2, 16], [1, 4, 1], [2, 1, 0])
    left: gl.constexpr = gl.SliceLayout(1, base)
    right: gl.constexpr = gl.SliceLayout(0, base)
    result: gl.constexpr = gl.SliceLayout(2, base)
    ml = gl.arange(0, BM, layout=gl.SliceLayout(1, left))
    nr = gl.arange(0, BN, layout=gl.SliceLayout(1, right))
    kl = gl.arange(0, 16, layout=gl.SliceLayout(0, left))
    kr = gl.arange(0, 16, layout=gl.SliceLayout(0, right))
    mo = gl.arange(0, BM, layout=gl.SliceLayout(1, result))
    no = gl.arange(0, BN, layout=gl.SliceLayout(0, result))
    return ml, nr, kl, kr, mo, no


@gluon.jit
def _zero(BM: gl.constexpr, BN: gl.constexpr):
    base: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 2, 16], [1, 4, 1], [2, 1, 0])
    return gl.full((BM, BN), 0, gl.float32, layout=gl.SliceLayout(2, base))


@gluon.jit(do_not_specialize=['T'])
def gram_kernel(Q, K, Beta, G, Table, Out, T,
                BM: gl.constexpr, BN: gl.constexpr, SCORE: gl.constexpr):
    ml, nr, kl, kr, mo, no = _axes(BM, BN)
    row = gl.program_id(0) * BM
    column = gl.program_id(1) * BN
    head = gl.program_id(2)
    chunk = (row // 64) * 64
    ml += row
    mo += row
    nr += chunk + column
    no += chunk + column
    accumulator = _zero(BM, BN)
    if not SCORE:
        beta = gl.load(Beta + ml * 32 + head, ml < T, other=0)
    for start in gl.static_range(0, 128, 16):
        left = gl.load(Q + (ml[:, None] * 16 + head // 2) * 128 + start + kl[None, :],
                       ml[:, None] < T, other=0)
        right = gl.load(K + (nr[:, None] * 16 + head // 2) * 128 + start + kr[None, :],
                        nr[:, None] < T, other=0)
        if not SCORE:
            left = (left.to(gl.float32) * beta[:, None]).to(gl.bfloat16)
        accumulator = _group16(left, right, accumulator)
    gr = gl.load(G + mo * 32 + head, mo < T, other=0)
    gc = gl.load(G + no * 32 + head, no < T, other=0)
    result = accumulator * _exp(Table, gr[:, None] - gc[None, :])
    triangle = mo[:, None] >= no[None, :] if SCORE else mo[:, None] > no[None, :]
    result = gl.where(triangle & (no[None, :] < T), result, 0)
    offsets = (mo[:, None] * 32 + head) * 64 + (no - chunk)[None, :]
    gl.store(Out + offsets, result.to(Out.dtype.element_ty), mo[:, None] < T)


@gluon.jit(do_not_specialize=['T'])
def w_kernel(A, K, Beta, G, Table, W, T, BM: gl.constexpr, BN: gl.constexpr):
    ml, nr, kl, kr, mo, no = _axes(BM, BN)
    row = gl.program_id(0) * BM
    column = gl.program_id(1) * BN
    head = gl.program_id(2)
    chunk = (row // 64) * 64
    ml += row
    mo += row
    nr += column
    no += column
    accumulator = _zero(BM, BN)
    for start in gl.static_range(0, 64, 16):
        source = chunk + start + kr
        left = gl.load(A + (ml[:, None] * 32 + head) * 64 + start + kl[None, :],
                       ml[:, None] < T, other=0)
        keys = gl.load(K + (source[None, :] * 16 + head // 2) * 128 + nr[:, None],
                       source[None, :] < T, other=0)
        beta = gl.load(Beta + source * 32 + head, source < T, other=0)
        gate = gl.load(G + source * 32 + head, source < T, other=0)
        kb = (keys.to(gl.float32) * beta[None, :]).to(gl.bfloat16)
        right = (kb.to(gl.float32) * _exp(Table, gate)[None, :]).to(gl.bfloat16)
        accumulator = _group16(left, right, accumulator)
    offsets = (mo[:, None] * 32 + head) * 128 + no[None, :]
    gl.store(W + offsets, accumulator.to(gl.bfloat16), mo[:, None] < T)


@gluon.jit(do_not_specialize=['T'])
def integer_u_kernel(A, V, Beta, U, Debug, T, H: gl.constexpr, NV: gl.constexpr,
                     BM: gl.constexpr, BN: gl.constexpr, CAPTURE_F32: gl.constexpr):
    ml, nr, kl, kr, mo, no = _axes(BM, BN)
    row = gl.program_id(0) * BM
    column = gl.program_id(1) * BN
    head = gl.program_id(2)
    chunk = (row // 64) * 64
    ml += row
    mo += row
    nr += column
    no += column
    accumulator = _zero(BM, BN)
    for start in gl.static_range(0, 64, 16):
        source = chunk + start + kr
        left = gl.load(A + (ml[:, None] * H + head) * 64 + start + kl[None, :],
                       ml[:, None] < T, other=0)
        values = gl.load(V + (source[None, :] * H + head) * NV + nr[:, None],
                         (source[None, :] < T) & (nr[:, None] < NV), other=0)
        beta = gl.load(Beta + source * H + head, source < T, other=0)
        right = (values.to(gl.float32) * beta[None, :]).to(gl.bfloat16)
        accumulator = _group16(left, right, accumulator)
    offsets = (mo[:, None] * H + head) * NV + no[None, :]
    mask = (mo[:, None] < T) & (no[None, :] < NV)
    gl.store(U + offsets, accumulator.to(gl.bfloat16), mask)
    if CAPTURE_F32:
        gl.store(Debug + offsets, accumulator, mask)


@gluon.jit(do_not_specialize=['T'])
def residual_kernel(W, U, H0, G, Table, VNew, Residual, T,
                    BM: gl.constexpr, BN: gl.constexpr):
    ml, nr, kl, kr, mo, no = _axes(BM, BN)
    row = gl.program_id(0) * BM
    column = gl.program_id(1) * BN
    head = gl.program_id(2)
    ml += row
    mo += row
    nr += column
    no += column
    accumulator = _zero(BM, BN)
    for start in gl.static_range(0, 128, 16):
        left = gl.load(W + (ml[:, None] * 32 + head) * 128 + start + kl[None, :],
                       ml[:, None] < T, other=0)
        right = gl.load(H0 + (head * 128 + nr[:, None]) * 128 + start + kr[None, :]).to(gl.bfloat16)
        accumulator = _group16(left, right, accumulator)
    offsets = (mo[:, None] * 32 + head) * 128 + no[None, :]
    u = gl.load(U + offsets, mo[:, None] < T, other=0).to(gl.float32)
    residual = u - accumulator
    gate = gl.load(G + mo * 32 + head, mo < T, other=0)
    last = gl.load(G + (T - 1) * 32 + head)
    scaled = residual * _exp(Table, last - gate)[:, None]
    gl.store(VNew + offsets, residual.to(gl.bfloat16), mo[:, None] < T)
    gl.store(Residual + offsets, scaled.to(gl.bfloat16), mo[:, None] < T)


@gluon.jit(do_not_specialize=['T'])
def state_kernel(K, Residual, H0, G, Table, Final, T,
                 BM: gl.constexpr, BN: gl.constexpr):
    ml, nr, kl, kr, mo, no = _axes(BM, BN)
    row = gl.program_id(0) * BM
    column = gl.program_id(1) * BN
    head = gl.program_id(2)
    ml += row
    mo += row
    nr += column
    no += column
    offsets = (head * 128 + mo[:, None]) * 128 + no[None, :]
    initial = gl.load(H0 + offsets)
    decay = _exp(Table, gl.load(G + (T - 1) * 32 + head))
    accumulator = _zero(BM, BN)
    for start in gl.static_range(0, 64, 16):
        left = gl.load(Residual + ((start + kl[None, :]) * 32 + head) * 128 + ml[:, None],
                       start + kl[None, :] < T, other=0)
        right = gl.load(K + ((start + kr[None, :]) * 16 + head // 2) * 128 + nr[:, None],
                        start + kr[None, :] < T, other=0)
        accumulator = _group16(left, right, accumulator)
    gl.store(Final + offsets, gl.fma(initial, decay, accumulator))


@gluon.jit(do_not_specialize=['T'])
def output_kernel(Q, VNew, H0, G, Scores, Table, Out, T,
                  BM: gl.constexpr, BN: gl.constexpr):
    ml, nr, kl, kr, mo, no = _axes(BM, BN)
    row = gl.program_id(0) * BM
    column = gl.program_id(1) * BN
    head = gl.program_id(2)
    ml += row
    mo += row
    nr += column
    no += column
    old = _zero(BM, BN)
    local = _zero(BM, BN)
    for start in gl.static_range(0, 128, 16):
        left = gl.load(Q + (ml[:, None] * 16 + head // 2) * 128 + start + kl[None, :],
                       ml[:, None] < T, other=0)
        right = gl.load(H0 + (head * 128 + nr[:, None]) * 128 + start + kr[None, :]).to(gl.bfloat16)
        old = _group16(left, right, old)
    for start in gl.static_range(0, 64, 16):
        left = gl.load(Scores + (ml[:, None] * 32 + head) * 64 + start + kl[None, :],
                       ml[:, None] < T, other=0)
        right = gl.load(VNew + ((start + kr[None, :]) * 32 + head) * 128 + nr[:, None],
                        start + kr[None, :] < T, other=0)
        local = _group16(left, right, local)
    gate = gl.load(G + mo * 32 + head, mo < T, other=0)
    prior = old * _exp(Table, gate)[:, None]
    result = gl.fma(local, 0.08838834764831845, prior * 0.08838834764831845)
    offsets = (mo[:, None] * 32 + head) * 128 + no[None, :]
    gl.store(Out + offsets, result.to(gl.bfloat16), mo[:, None] < T)
