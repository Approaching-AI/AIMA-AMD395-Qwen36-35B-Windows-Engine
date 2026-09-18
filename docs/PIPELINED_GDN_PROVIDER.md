# GDN provider segment pipeline

`QRT_FLA_GDN_PIPELINED_SEGMENTS=1|2` is an experimental, default-off provider
schedule. Mode 1 overlaps output with the next preparation/state sequence;
mode 2 uses separate preparation, ordered-state and output streams. The original
kernel arithmetic, K16 order, BF16 boundaries and persistent FP32 state stay
unchanged. No model or release acceptance is established yet.

The complete segment function is split into three host phases. Preparation
contains normalization, V/beta conversion, gate scan, KKT, inverse, W/U and
paired output scores. State remains ordered on one stream. Output consumes
the original H, Vnew, Q, gate and prepared scores. Three independent storage
slots prevent the next segment from overwriting a live consumer. Before reuse,
the producer waits for that slot's previous output event. The final output is
joined to the caller stream and completed before return or event reuse.

Each submission window contains at most eight original 1024-token segments.
Each completed phase retains a 100 ms guard. Long invocations use consecutive
windows; one partial K64 tail uses the original neutral-padding path with the
completed prefix state. Checkpoint calls and capture/diagnostic paths retain
the original schedule. The existing state8, scalar-matrix, paired-score,
cooperative exact route is required, with coarse and fused variants disabled.
All caller input/output/state ranges must be disjoint. Unsupported compatible
shapes use the original route; malformed mode values return an error.

The three slots, streams and events are prepared once and released with the
provider. Slots own all segment scratch and preserve the production A/W,
inverse/Vnew and V/U aliases. They add 179,945,472 device bytes for the current
1024-token configuration, separate from the original default owner. Every
partial-submission failure drains all three streams before returning. These
ownership properties are exercised with the production host owner under a
dependency-graph HIP model and ASan/UBSan; that test is not numerical GPU
evidence. Native provider and real-model comparisons remain pending.
