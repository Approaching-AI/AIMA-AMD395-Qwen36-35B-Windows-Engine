# Paired GDN scores and shared arenas

The native fixture at `3c8a535209241a770e34f5a6349839005dd5e354` computes each
Q/K dot once for the two value heads sharing that Q/K head. Each head keeps
its original gate and BF16 conversion. An 8-row by 32-column tile reuses
operands without changing K16 accumulation order or the exceptional fallback.
The combined candidate also reuses the checked state/output shared arenas.

All 288 generated safety configurations pass, including partial chunks and
segments, strong decay, signed zeros, exceptional encodings and both U ownership
modes. Every logical score, W/U, checkpoint, residual, output and final FP32
state is checked against the independent original producers after each attempt.
The original q7169 GB10 capture qualifies 29,364,224 output and W/U/Vnew values,
113 checkpoints and 524,288 final state values. The q8192 extension uses 7,168
original rows followed by 1,024 repeated rows; only its original prefix carries
the external GB10 boundary. Full extended arithmetic still matches originals.

| Input / U ownership | Current control ms | Paired scores ms | Paired + arenas ms |
| --- | ---: | ---: | ---: |
| Generated dense q8192 / separate | 94.0524 | 79.1442 | 74.2795 |
| Generated dense q8192 / U=V | 96.7653 | 78.4655 | 71.1433 |
| Generated decay q8192 / separate | 101.2009 | 89.4670 | 82.5025 |
| Generated decay q8192 / U=V | 105.2240 | 88.7971 | 82.6544 |
| Captured q7169 / separate | 93.8926 | 81.6595 | 71.8419 |
| Captured q7169 / U=V | 89.0581 | 82.6698 | 75.9333 |
| Captured q8192 extension / separate | 105.9406 | 98.0421 | 84.8710 |
| Captured q8192 extension / U=V | 103.2150 | 92.9217 | 85.7028 |

One warmup and three rotated completed host samples include score generation,
WU, state and output across 1,024-token segments, without intermediate host
waits. Every combined-candidate sample is below every corresponding control
sample in all eight cases. Input normalization, gate scans, KKT and inversion
are supplied inputs, so this clock does not measure complete GDN or model TTFT.
The timed control is the current four-lane score producer. The earlier safety
fixture used the old sixteen-lane score control and padded allocation; it has
no retained throughput measurement. The corrected fixture guards exactly
`count*2048` score words and checks every word independently.

The paired kernel declares 10,276 LDS bytes and 70 VGPRs, versus zero LDS and
41 VGPRs for the current four-lane score control. State/output arena resources
remain 26,180/19,264 LDS bytes and 113/77 VGPRs. All seven timed kernels declare
zero private storage. No occupancy is measured.

`QRT_FLA_GDN_PAIRED_SCORE_ARENAS=1` now exposes the combined candidate in the
cooperative provider, default off. Counts must fit a 1,024-token segment.
Scores/output require scalar matrices; state requires scalar columns eight;
coarse-interval modes retain their existing route. Prefix checkpoint state
capture retains its original implementation. A new launch test checks 4,320
option/shape/error combinations and checkpoint fallback under ASan/UBSan;
all six related launch tests pass. The Windows HIP build completes in
51,563.531 ms, producing a 943,104-byte DLL, SHA256
`67209121af32c1e7dce883c8ba4abfb7ea5152fac822dc0909b94aa586aad3a5`,
from source `7b20c9041e14b2db133afb7e5b05b4e3223ed04c`.

The new DLL passes full output/final-state reference comparison and sync/async
parity for generated q64/q65 and original GB10 q7169 with the switch OFF/ON.
The q65 process passes; its post-run observer incorrectly expected two rather
than four markers per operation. Its actual `[64,1]` partition runs twice,
including the asynchronous call. An offline correction validates all three
operation sequences and preserves the initial transport failure and raw logs.
No numerical result is changed or rerun. Native activation checks pass for
all qualified calls. Real q8192/out512 model comparisons are complete below.

Provider evidence: [build and complete provider comparisons](../benchmarks/correctness/paired-score-gdn-provider-native-20260918.json),
112,039 bytes, SHA256
`bdb85abafd94b35fed2d12e403c4adaf14a8d9a4ff2ff8745a323eea8d4172c1`.
Commands are `run-native-paired-score-gdn-provider-build-r1.ps1` and
`run-native-paired-score-gdn-provider-probe-r1.ps1` on baiying.
The component result alone does not establish model performance. The product
comparison below updates the experimental control; package defaults stay off.

Evidence: [complete native record](../benchmarks/correctness/paired-score-gdn-native-components-20260918.json),
990,257 bytes, SHA256
`2219625c0fd025d5fc8f2049237d985624769e085b8f7de42b409c2976638f61`.
It pins baiying, `run-native-paired-score-gdn-r2.ps1`, the source, compiler,
executable, all inputs and original GB10 capture of
`D:\models\Qwen3.6-35B-A3B`. No full model is loaded by this fixture.

## Real q8192/out512 comparison

`run-paired-score-gdn-product-r1.ps1` runs the model at
`D:\models\Qwen3.6-35B-A3B` in four fresh baiying processes, using the same
FLA DLL above. Only `QRT_FLA_GDN_PAIRED_SCORE_ARENAS=0/1` changes. Whole
`6e4908b`, CK `df2ea51`, MoE `9235750`, CLI `24c4304` and all AOT/table assets
remain fixed. Fused probability PV1, compact MoE1 and adaptive linear OUT2
stay enabled with their original numerical bounds.

| Order / mode | Load ms | TTFT ms | TPOT ms |
| --- | ---: | ---: | ---: |
| 1 / OFF | 21363.2014 | 24913.4799 | 100.251951 |
| 2 / ON | 21666.0498 | 24696.2266 | 100.389425 |
| 3 / ON | 21371.0135 | 24723.0012 | 100.635676 |
| 4 / OFF | 21352.5317 | 25031.8847 | 100.571042 |

All 2,048 original GB10 output IDs, prompts and actual callbacks match. All
first logits are 10.375 with zero error. Each ON run records exactly 240
activations for each of scores/state/output, covering all 30 layers and eight
segments per layer. Every retained owner and host guard passes. Dense/coarse
correction counts match as diagnostics, without changing the GB10 boundary.

OFF/ON TTFT medians are 24,972.6823/24,709.6139 ms, a reduction of 263.0684 ms
or 1.0534%. Both ON samples are below both OFF samples. Retain ON in the next
experimental q8192 control. The complete-model improvement is smaller than
the component result; no proportional extrapolation is retained. All loads
remain below 30 seconds. TPOT medians of 100.4114965/100.5125505 ms do not
establish a decode gain. Two observations per arm do not qualify new prompts,
prefix/long contexts or a package. Code/package defaults stay off. The 10,000 ms
first gate and 4,187.415605 ms retained target remain open; nothing is published.

Evidence: [all four product runs](../benchmarks/correctness/paired-score-gdn-product-20260918.json),
1,269,463 bytes, SHA256
`e416abed136f64ed0c9ab8e01d34b3473c9a38695cde36a829f2f884dfe1b8c3`.
