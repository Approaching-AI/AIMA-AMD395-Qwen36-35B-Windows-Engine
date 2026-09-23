"""Compact looped recurrence with shared carriers split into 16-row tiles.

The ordered K16 arithmetic, BF16 boundaries and FP32 state fma are unchanged.
Explicit shared-memory carriers separate the three phases. No runtime binds
this experiment until original-model and native comparisons pass.
"""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from group16 import _group16
from exp2 import _exp
from ordered_pipeline import _axes, _zero


@gluon.jit(do_not_specialize=['T'])
def persistent_kernel(Q, K, W, U, G, Scores, Table, Initial, Final, Out,
                      VNew, Checkpoints, T, CAPTURE: gl.constexpr):
    base: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 2, 16], [1, 4, 1], [2, 1, 0])
    left_layout: gl.constexpr = gl.SliceLayout(1, base)
    right_layout: gl.constexpr = gl.SliceLayout(0, base)
    result_layout: gl.constexpr = gl.SliceLayout(2, base)
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    head = gl.program_id(0)
    value_start = gl.program_id(1) * 8
    state_v = gl.arange(0, 8, layout=gl.SliceLayout(1, right_layout)) + value_start
    state_k = gl.arange(0, 16, layout=gl.SliceLayout(0, right_layout))
    state_offsets = (head * 128 + state_v[:, None]) * 128 + state_k[None, :]
    state = gl.allocate_shared_memory(gl.float32, [8, 8, 16], shared)
    for column_group in range(0, 8):
        state.index(column_group).store(gl.load(Initial + state_offsets + column_group * 16))
    v_new = gl.allocate_shared_memory(gl.bfloat16, [4, 16, 8], shared)
    residual = gl.allocate_shared_memory(gl.bfloat16, [4, 16, 8], shared)
    gl.thread_barrier()
    ml, nr, kl, kr, mo, no = _axes(8, 8)
    nr += value_start
    no += value_start
    for first in range(0, T, 64):
        valid = gl.minimum(64, T - first)
        last = gl.load(G + (first + valid - 1) * 32 + head)
        if CAPTURE:
            checkpoint_offsets = ((first // 64 * 32 + head) * 128 + state_v[:, None]) * 128 + state_k[None, :]
            for column_group in range(0, 8):
                checkpoint = state.index(column_group).load(right_layout).to(gl.bfloat16)
                gl.store(Checkpoints + checkpoint_offsets + column_group * 16, checkpoint)

        # Complete every residual row before any output or state consumes it.
        for row_group in range(0, 4):
            for row_half in gl.static_range(0, 2):
                row_l = first + row_group * 16 + row_half * 8 + ml
                row_o = first + row_group * 16 + row_half * 8 + mo
                accumulator = _zero(8, 8)
                for start in range(0, 128, 16):
                    left = gl.load(W + (row_l[:, None] * 32 + head) * 128 + start + kl[None, :],
                                   row_l[:, None] < T, other=0)
                    right = state.index(start // 16).load(right_layout).to(gl.bfloat16)
                    accumulator = _group16(left, right, accumulator)
                offsets = (row_o[:, None] * 32 + head) * 128 + no[None, :]
                u = gl.load(U + offsets, row_o[:, None] < T, other=0).to(gl.float32)
                r = u - accumulator
                gate = gl.load(G + row_o * 32 + head, row_o < T, other=0)
                scaled = r * _exp(Table, last - gate)[:, None]
                r = gl.where(row_o[:, None] < T, r, 0)
                scaled = gl.where(row_o[:, None] < T, scaled, 0)
                v_new.index(row_group).slice(row_half * 8, 8, 0).store(r.to(gl.bfloat16))
                residual.index(row_group).slice(row_half * 8, 8, 0).store(scaled.to(gl.bfloat16))
                if CAPTURE:
                    gl.store(VNew + offsets, r.to(gl.bfloat16), row_o[:, None] < T)
        gl.thread_barrier()

        # Every output sees the incoming state, before the update overwrites it.
        for row_group in range(0, 8):
            row_l = first + row_group * 8 + ml
            row_o = first + row_group * 8 + mo
            old = _zero(8, 8)
            local = _zero(8, 8)
            for start in range(0, 128, 16):
                left = gl.load(Q + (row_l[:, None] * 16 + head // 2) * 128 + start + kl[None, :],
                               row_l[:, None] < T, other=0)
                right = state.index(start // 16).load(right_layout).to(gl.bfloat16)
                old = _group16(left, right, old)
            for start in range(0, 64, 16):
                left = gl.load(Scores + (row_l[:, None] * 32 + head) * 64 + start + kl[None, :],
                               row_l[:, None] < T, other=0)
                right = v_new.index(start // 16).permute([1, 0]).load(right_layout)
                local = _group16(left, right, local)
            gate = gl.load(G + row_o * 32 + head, row_o < T, other=0)
            prior = old * _exp(Table, gate)[:, None]
            result = gl.fma(local, 0.08838834764831845, prior * 0.08838834764831845)
            offsets = (row_o[:, None] * 32 + head) * 128 + no[None, :]
            gl.store(Out + offsets, result.to(gl.bfloat16), row_o[:, None] < T)
        gl.thread_barrier()

        # The value rows are independent. Each block updates its complete K128.
        svl, skr, stl, str_, svo, sko = _axes(8, 16)
        svl += value_start
        svo += value_start
        decay = _exp(Table, last)
        for column_group in range(0, 8):
            column_r = column_group * 16 + skr
            view = state.index(column_group)
            old_state = view.load(result_layout)
            state_accumulator = _zero(8, 16)
            for start in range(0, 64, 16):
                state_left = residual.index(start // 16).permute([1, 0]).load(left_layout)
                source = first + start + str_
                state_right = gl.load(K + (source[None, :] * 16 + head // 2) * 128 + column_r[:, None],
                                source[None, :] < T, other=0)
                state_accumulator = _group16(state_left, state_right, state_accumulator)
            view.store(gl.fma(old_state, decay, state_accumulator))
        gl.thread_barrier()
    for column_group in range(0, 8):
        gl.store(Final + state_offsets + column_group * 16, state.index(column_group).load(right_layout))
