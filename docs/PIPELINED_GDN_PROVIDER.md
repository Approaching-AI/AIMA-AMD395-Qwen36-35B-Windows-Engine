# GDN provider segment pipeline

`QRT_FLA_GDN_PIPELINED_SEGMENTS=1|2` is an experimental, default-off provider
schedule. Mode 1 overlaps output with the next preparation/state sequence;
mode 2 uses separate preparation, ordered-state and output streams. The original
kernel arithmetic, K16 order, BF16 boundaries and persistent FP32 state stay
unchanged. Native and scoped real-model correctness comparisons below pass. No performance or release acceptance follows.

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
All caller input/output/state ranges must be disjoint. Other shapes and settings use the original route; malformed mode values return an error.

The three slots, streams and events are prepared once and released with the
provider. Slots own all segment scratch and preserve the production A/W,
inverse/Vnew and V/U aliases. They add 179,945,472 device bytes for the current
1024-token configuration, separate from the original default owner. Every
partial-submission failure drains all three streams before returning. These
ownership properties are exercised with the production host owner under a
dependency-graph HIP model and ASan/UBSan; that test is not numerical GPU
evidence.

Native source `5abfbd7` passes nine probes: 64/65/1025 boundary cases and all
three modes at q7169/q8192. The 65-token observer originally expected one
segment per call, overlooking its separate logical tail; the preserved native
run passes the corrected observer without a GPU rerun. Source `cf86fef` fixes
the initially omitted slot bytes in the exported memory query. Four additional
probes pass with the final DLL, including original q7169 operands and generated
q8192. All output/final-state bits match the qualified original provider and
synchronous/asynchronous calls match. The q7169 operands carry the original
GB10 exact-output/state audit. Generated q8192 alone is not model acceptance.

Both builds have 215 source-inventory entries matching their commits. Across
the memory-only fix, all six GPU code objects (32 kernels) have identical
executable sections and metadata. Final DLL: 959488 bytes, SHA256
`49a9c951216112d1b598533073f393e73a9108355602455f5303880964ce4732`.
The combined host validation covers 21 distinct tests; fixture/extractor
failures and their corrections are retained.

[Complete native evidence](../benchmarks/correctness/pipelined-gdn-provider-native-20260918.json):
347340 bytes, SHA256
`8a357010f3144f60db96f7b77b0fe6b15a161bd68c557d0f254ffb62606a58e1`.
Guarded process walls include setup and reference execution and do not establish
a throughput gain. Fresh same-DLL real-model comparisons also pass all 2048 original GB10 IDs,
first logits 10.375/error 0, original prompts, actual callbacks and owner checks.
All 30 layers activate in each enabled run. Candidate and replay identities
match across all four runs.

| Order / mode | Load ms | TTFT ms | TPOT ms |
| --- | ---: | ---: | ---: |
| 1 / OFF | 21311.9599 | 23273.0217 | 100.884106 |
| 2 / three streams | 21284.4537 | 23338.3490 | 100.656181 |
| 3 / two streams | 21390.4485 | 23468.8446 | 100.808305 |
| 4 / OFF | 21352.4952 | 23278.0577 | 100.393801 |

Both enabled samples are slower than both controls. Keep this schedule disabled;
the qualified 23353.80795 ms baseline and current package remain unchanged.
The measured memory pool stays below the 30-second load limit, but no decode,
other-context or release gain is established. The 10-second and retained
4187.415605 ms targets remain open. The comparison preserves whole `ddacdc9`,
CK `1c2770d`, MoE `9235750`, CLI `24c4304` and all numerical coefficients.

[Four complete model runs](../benchmarks/correctness/pipelined-gdn-product-20260918.json):
1559256 bytes, SHA256
`a6e66433908ac49094c63548cc5ae6e4745c47332e3c1160f3590b837dd0d3a1`.
