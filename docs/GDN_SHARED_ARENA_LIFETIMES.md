# GDN shared arena lifetimes

Source `3c93193cbce390292ded4c3655847a406bb2bb66` reuses shared memory across
disjoint operand phases while keeping original arithmetic. The state kernel
finishes every W/H dot before K overwrites the W arena. The output kernel
finishes every Q/H dot before scores replace queries and values replace
checkpoints. Two original Q/H results remain in registers. All threads,
including inactive tail threads, reach the barriers before either overwrite.
The WU kernel, K16 dots, exceptional fallback, checkpoints, conversions, final
FMAs and U=V ownership remain unchanged. The initial fixture has no provider dispatch; subsequent opt-in integration
is tracked in [paired score GDN](PAIRED_SCORE_GDN.md).

All 288 native safety configurations pass, covering eight lengths through
1,025, six data families, three variants and both U ownership modes. Checks
include 500,465,664 outputs, 2,728,230,912 state/intermediate values and 291,192
independent CPU dots across the 48 unique generated cases.

The original GB10 q7169 capture matches all 29,364,224 outputs, equal-sized
W/U/Vnew boundaries, 113 BF16 checkpoints and 524,288 final FP32 state values.
The q8192 extension repeats 1,024 rows after 7,168 original rows; that prefix has
the external boundary and all extended outputs/states match original kernels.
Every warmup and timed attempt checks all values, guards, immutable inputs and
alias ownership. References remain observers only.

| Input / U ownership | Original ms | State reuse ms | State + output reuse ms |
| --- | ---: | ---: | ---: |
| Generated dense q8192 / separate | 76.7142 | 73.8886 | 69.8570 |
| Generated dense q8192 / U=V | 75.1935 | 75.5755 | 68.1248 |
| Generated decay q8192 / separate | 87.4666 | 82.6608 | 77.6995 |
| Generated decay q8192 / U=V | 82.7166 | 80.1942 | 78.4814 |
| Captured q7169 / separate | 71.7061 | 68.9807 | 68.1046 |
| Captured q7169 / U=V | 73.1806 | 69.5407 | 67.0281 |
| Captured q8192 extension / separate | 82.9689 | 80.2656 | 75.9103 |
| Captured q8192 extension / U=V | 84.3096 | 78.3991 | 76.4320 |

Every combined-candidate sample is below every corresponding control sample
in these eight cases. One warmup and three rotated measurements include all
barriers and arena writes, original dots and final FMAs. Bounded 1,024-token
segments preserve uninterrupted FP32 state without intermediate host waits.
Allocation, table/capture loading, score preparation, resets and comparisons
remain outside this WU/state/output clock; it is not full GDN or model TTFT.

Compiled state LDS falls from 42,820 to 26,180 bytes, with 113 VGPRs in both
kernels. Output LDS falls from 28,736 to 19,264 bytes and VGPRs from 80 to 77.
Both control and candidates declare zero private storage. These are resource
declarations, not measured occupancy or proof of why timing improves.

Retain this as a component building block. Extend the measurement to score
preparation, including the duplicated Q/K dots for adjacent value heads,
before product integration. No model-token result, TTFT baseline, package or
release acceptance changes.

Evidence: [complete native record](../benchmarks/correctness/lifetime-gdn-native-components-20260918.json),
503,752 bytes, SHA256
`b92c1a1acc03142b3c8097fe96dc5947229f0b47de89369525d2d2969288c9e4`.
It pins `run-native-lifetime-gdn-r1.ps1`, baiying, source/compiler/executable,
all inputs and the original GB10 capture of `D:\models\Qwen3.6-35B-A3B`.
No full model is loaded by this component fixture.
