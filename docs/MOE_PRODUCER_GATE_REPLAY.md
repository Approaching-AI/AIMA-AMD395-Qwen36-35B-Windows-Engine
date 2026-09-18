# Producer-local MoE gate and up correction

`QRT_QWEN36_MOE_PRODUCER_GATE_REPLAY=1` is a default-off q8192 experiment.
Each completed N32 up tile owns its corresponding gate tile. It reuses the
retained matrix A shared arena for a compact gate queue, performs original
four-lane staged K16 replay, then selects and corrects up before activation.
Gate's low-exponent predicate still sees native up; up's predicate still sees
the corrected BF16 gate. Native gate FP32 remains unchanged. Original norms,
predicates, SiLU LUT, BF16 endpoints and compact down consumer remain intact.

Admission requires the supported M64/overflow32 serial N32 schedule, positive
original L2 bounds and current staged operands. Unsupported replay variants,
parallel gate, classified expert ordering and the optional consumer interval
audit are rejected. Other token counts retain existing dispatch. There is no
additional device allocation; the local queue uses 6144 bytes of the existing
10640-byte matrix shared allocation. Completed matrix profiling includes both
corrections when enabled, as its activation marker declares.

The first generated fixture accidentally selected every cell in its intended
sparse case. Its recorded counts remain valid dense-fallback evidence. Source
`7c79356` fixes only generated norm controls and asserts actual sparse selection;
it also specializes the epilogue to the required original staged primitive.
All five corrected cases pass, including empty/full/sparse queues, low-exponent
dependencies, scattered routes and overflow descriptors. Every native/corrected
FP32 surface, BF16 activation, debug endpoint, count, prepared word and redzone
matches the retained matrix plus global expert-ordered correction.

The generated q8192 case covers 256 experts, 65536 routes, 67108864 projection
cells and 9913088 selected cells. Control/producer/control clocks are
123.612305/112.554649/123.430077 ms, an 8.8783% component reduction. These are
generated operands and norm controls, not real-model performance acceptance.

Runtime source `932b55b` preserves the exact tested matrix body, epilogue,
staged primitive and fixture. Eleven local tests pass, including actual runtime
admission and missing-current-view failures under ASan/UBSan. All 414 compiled
inputs in each native build match their pinned commits. The final DLL declares
70/75 VGPRs for original/producer matrix variants, unchanged 10640-byte LDS and
zero private storage. Static resources do not establish occupancy. A PowerShell
loop-variable collision stopped the first provider wrapper before compilation;
its original record and the corrected successful build are retained.

[Native commands, source comparison and checks](../benchmarks/correctness/moe-producer-gate-native-20260918.json):
453328 bytes, SHA256
`6ae37e57ad97e16fda5e87337d640e7d80e2688f33ea0e19b16825dbc0d060cc`.

Four fresh processes run the real `D:\models\Qwen3.6-35B-A3B` on baiying.
Both arms use the same `932b55b` MoE DLL, 1053696 bytes, SHA256
`5c3ad4e7d27445c0f3b4522798903f1a40dfae2f257d206e7c377fdd44aba4f3`.
Whole `ddacdc9`, CK `df2ea51`, FLA `7b20c90` and CLI `24c4304` remain fixed.
Only the producer option changes between arms.

| Order / mode | Load ms | TTFT ms | TPOT ms |
| --- | ---: | ---: | ---: |
| 1 / OFF | 21434.8047 | 23816.4707 | 100.637769 |
| 2 / ON | 21383.1778 | 23780.8694 | 100.440704 |
| 3 / ON | 21414.7814 | 23819.8666 | 100.541283 |
| 4 / OFF | 21397.6480 | 23842.8110 | 100.435337 |

All 2048 original GB10 output IDs, prompts and actual callbacks match. Every
first logit is 10.375 with zero error. All owner/activation checks and host
guards pass. Dense130, coarse9, compact-down40 and adaptive30 count records
remain identical as diagnostics.

OFF/ON medians are 23829.64085/23800.3680 ms, a 29.27285 ms or 0.1228%
reduction. Their observed ranges overlap. This does not establish a stable or
material product benefit while TTFT exceeds 10 seconds. Keep the option off
and retain MoE `9235750` with the existing 23902.4417 ms qualified baseline;
do not recalibrate that baseline to these new OFF measurements. No decode,
prefix, long-context, package or release qualification follows.

[All four complete model boundaries and the decision](../benchmarks/correctness/moe-producer-gate-product-20260918.json):
1571043 bytes, SHA256
`de2f3cc49a80795e52dec976536ffbf02a17246bfb1a34f2bee02d91e417b9a2`.
