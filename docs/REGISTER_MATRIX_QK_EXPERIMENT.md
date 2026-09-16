# Full private carries and shared scratch reuse for matrix QK

Isolated source `6568fff` moves each full integer K16 carry from shared memory
to its owning thread. Three exact matrix partial arrays replace four. The
compact variants reuse those arrays as bounded per-wave fallback queues after
all matrix readers retire. A queue entry preserves the full significand,
original signed exponent and sign; the owning thread retrieves it only after
fallback completion. Direct and compact 64/128-column variants preserve the
existing integer certificate, original scalar fallback and K16 order. The
previous queue32 route and retained prepared scalar are separate controls.

The first native compile stops before GPU execution because C++ list
initialization rejects an implicit int32-to-int16 exponent restoration.
`5fceaec` explicitly restores the original type at both queue reads. The
original int16 value was losslessly widened into its shared slot. Both source
revisions, bounded commands and the failed build remain in the evidence.

Local C ABI and public hygiene checks pass; no new full Rust/Python suite is
claimed. The corrected baiying build completes in 14,815.463 ms. All 144 native
safety reports pass, covering 24 shapes and six variants, with 9,216 independent
CPU dots. Cases include full rejected queues, ragged 129-row/257-key tails,
causal masks, signed zeros, cancellation, late exceptional values, subnormals
and carries beyond finite FP32. All input and encoding bytes remain unchanged.

Full q7169 and repeated-row q8192 checks pass 23,130,145,152 raw score slots and
2,904 further CPU dots. Every warmup and rotated sample compares complete score
buffers, external redzones and unused output tails. q8192 repeats 1,023 rows
from the original layer3 Q/K capture; it is not a new model prompt, attention
context comparison or GB10 token gate.

| Route | q7169 total ms | q8192 total ms |
| --- | ---: | ---: |
| Retained prepared scalar | 271.8010 | 349.4697 |
| Previous matrix queue32 | 516.4314 | 685.5473 |
| Private carries, direct64 | 560.1949 | 744.8969 |
| Private carries, direct128 | 1138.0692 | 1481.0546 |
| Reused scratch, queue64 | 490.0484 | 638.8626 |
| Reused scratch, queue128 | 891.4481 | 1157.4928 |

Totals include the completed score route, per-slab query encoding and one-time
key preparation. Allocation, resets and validation are outside the timers.
Fixed preparation order does not establish a separate preparation speedup.
All kernels declare wave32 and zero private bytes. The new direct64/direct128/
queue64/queue128 kernels use 95/119/119/139 VGPRs; 64-column variants use
24,128 LDS bytes and 128-column variants use 45,888 bytes. These declarations
do not establish occupancy or a measured cause for the timing differences.

Keep all new routes isolated. Queue64 improves on the prior matrix queue32
control, but remains about 1.83 times slower than retained scalar at q8192.
The larger tiles regress further. No provider integration or real-model run
follows. The retained stack, 10,000 ms TTFT gate and 4,187.415605 ms retained
target remain unchanged, and all outstanding release gates stay open.

Evidence: [source, native runs and complete comparisons](../benchmarks/correctness/register-matrix-qk-native-components-20260917.json),
202,109 bytes, SHA256
`28affbc8df3bd46be282f492df3064f256cf834f8b67b19a69359d50842f446e`.
