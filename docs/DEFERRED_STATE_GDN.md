# Deferred recurrent state experiment

Source `3ae0440` evaluates a complete GDN route with deferred original state
dots. Each 64-token checkpoint still publishes the original BF16 values.
Between checkpoints, native C64 matrix results and the unchanged error bound
propagate FP32 state intervals through the original ordered FMAs. An ambiguous
checkpoint expands a cached original K64 tail backwards until its BF16 result
is fixed. Each 1024-token segment ends only after every state cell resolves to
the original FP32 bits. Full original replay remains available from the seed.

Canonical scaled residuals are archived for those delayed dots. One variant
computes WH residuals with the original scalar path; the other uses the existing
coarse consumer certificate and scalar fallback. The WU/output kernels, U=V
ownership and state ABI remain unchanged. These kernels are isolated from
runtime dispatch. Original comparison values never choose work.

The UBSan host audit passes 65536 sequences and 622592 boundaries, including
signed zeros, invalid intervals, exceptional exact endpoints and one-time
caches. It checks BF16 checkpoints and final FP32 states independently.
On baiying/gfx1151, 144 safety configurations cover eight lengths through 1025,
three data families, three variants and both U ownership modes. The complete
q8192 test uses eight 1024-token segments, one warmup and three rotated samples.
All 805306368 output and 4039114752 state/intermediate comparisons pass, with
172032 independent CPU dots. Separate original state64 calls reproduce the
state8 control and provide 67633152 FP32 trace values. Candidate warmups check
285212672 produced state-interval values before later refinement. No interval
escapes, unresolved states, guard changes or input ownership errors occur.

| Complete q8192 WU/state/output, ms | Original | Deferred, scalar WH | Deferred, coarse WH |
| --- | ---: | ---: | ---: |
| Separate U | 79.2640 | 176.9948 | 176.0933 |
| Product U=V | 78.4289 | 177.9060 | 173.8865 |

All candidate samples are slower than all control samples. The candidate
computes 38218556 of 67108864 original state dots, but reducing that work does
not improve complete execution. It refines 15065042 of 71303168 boundaries,
with maximum tail depth 16 and 1048284 full-tail resolutions. Counter values
match across all attempts and both variants/ownership modes.

Timing includes all candidate initialization, archiving, cached dot replay,
ordered FMA reevaluation, unique CTA diagnostic writes and finalization.
Allocation/reset, transfers, reference generation and checks are outside the
clock. Timed calls omit the warmup-only interval observer stores. No internal
host wait or device event interval substitutes for full completion. Static
metadata declares no private allocation/spills; this does not establish
occupancy or identify the cause of the measured regression.

Keep both candidates isolated. Generated component correctness and arithmetic
coverage are not a real-model or GB10 continuation gate. The qualified model
baseline remains 24835.3024 ms q8192 TTFT; no runtime, package, performance target
or release change follows.

[Source, bounded commands and native evidence](../benchmarks/correctness/deferred-state-gdn-native-components-20260918.json):
330614 bytes, SHA256
`10c6f5aa377e641b88e8e65e6b952ef39af118f35206ef692e67bf7d318004cf`.
