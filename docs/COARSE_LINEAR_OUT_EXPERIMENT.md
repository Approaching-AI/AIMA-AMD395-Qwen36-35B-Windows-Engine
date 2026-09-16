# Coarse linear-attention OUT experiment

The retained coarse OUT owner covers ten full-attention layers. A completed
profile attributes about 2.92 seconds to OUT in the thirty linear-attention
layers. Those projections have the same q8192, 2048-row, K4096 geometry, so this
experiment extends the existing owner to that larger call surface.

`QRT_QWEN36_COARSE_LINEAR_OUT_PRODUCER` is a strict, default-off `0`/`1` option.
It applies only when linear OUT already requests correction with radius 512
and PPB 1000, and residual/post-normalization consumes the BF16 endpoint.
Host diagnostic materialization, other token counts and streamed linear
workspace paths retain their original dispatch. An 8192-token seed inside a
longer request can use this route; a q8192 test does not qualify that request.
Conflicting original OUT sweeps, terminal diagnostics, shadow audits and
residual/interval filters are rejected when this route would execute.

The owner retains its 295,739,396-byte descriptor-owned arena, C64 matrix
producer, conditional native error coefficient 2^-19, complete compaction,
bounded original K16 replay and BF16 RNE endpoint. There is no PPB reduction
or changed external numerical tolerance. An optional F32 carrier is populated
from the actual BF16 output for later tracing; it is explicitly a rounded
carrier, not a record of the unrounded original matrix result. Both output
conversions are drained before the arena returns to its owner, including
failure paths. Existing full-attention calls keep their original interface
and do not request this extra conversion.

This is a numerical and performance experiment, not an accepted runtime
profile. The empirical native matrix error premise is not a universal
hardware proof. Qualification requires a same-DLL control and experiment on
baiying using the real model, original 8192 prompt IDs, all 512 GB10 output
tokens, the unchanged first-logit tolerance of 0.125 and actual streaming
callbacks. The original TTFT and load targets remain in force. Nothing is
published or enabled by this source change.
