# Exact GDN group windows

Source `44ab13b3c3558827095ed3944d9240ba21a0cba5` replaces the preceding
per-group support checks with one occupancy bit per K16 operand group. The
first candidate visits only the contiguous first-to-last active range; the
second visits occupied groups in ascending order. Whole-zero intersections
return zero immediately. Every visited group retains original arithmetic,
including exceptional fallback; skipped zero groups preserve canonical carries.
The six WU/state/output dots, checkpoints, FMAs and U=V ownership remain ordered.
These kernels have no provider dispatch.

Host UBSan covers all 65,536 BF16 encodings, 65,536 mask pairs, strides 1/8 and
524,288 raw carried-state comparisons. All 288 native safety configurations
pass 500,465,664 output and 2,728,230,912 state/intermediate comparisons, plus
291,192 independent CPU dots across the 48 unique length/data cases.

On baiying, the original GB10 q7169 capture matches all 29,364,224 BF16 outputs,
equal-sized W/U/Vnew boundaries, 113 checkpoints and the final 524,288 FP32 state
values. The extended q8192 geometry repeats the first 1,024 rows after 7,168
original rows. Its original prefix has the external GB10 boundary; all extended
outputs and states match the unchanged kernels. Every warmup and timed attempt
checks complete outputs, state, intermediates, guards, inputs and aliases.
References are observers and never select work.

One audited warmup and three rotated measurements include all occupancy
preparation and complete WU/state/output execution. The owner uses 1,024-token
segments with uninterrupted FP32 state and no intermediate host waits. Table
loading, score preparation, resets and comparisons are outside this clock.

| Input / U ownership | Original ms | Window ms | Set bits ms |
| --- | ---: | ---: | ---: |
| Generated dense q8192 / separate | 76.2581 | 81.5855 | 83.8153 |
| Generated dense q8192 / U=V | 76.0451 | 85.0515 | 83.5508 |
| Generated decay q8192 / separate | 83.5196 | 79.2961 | 82.4192 |
| Generated decay q8192 / U=V | 84.3764 | 79.4348 | 84.5343 |
| Captured q7169 / separate | 73.4082 | 72.0364 | 73.9726 |
| Captured q7169 / U=V | 70.4186 | 74.0223 | 74.6263 |
| Captured q8192 extension / separate | 83.6669 | 83.0402 | 86.3556 |
| Captured q8192 extension / U=V | 83.3770 | 84.7132 | 83.7583 |

The q8192 candidates omit 274,067,543 / 274,091,479 of 1,207,959,552 K16 groups,
including 13,122,815 whole-zero dots. The preceding exact-support masks could
also detect disjoint supports within otherwise occupied groups; these coarser
metadata deliberately do not. Zero counters for the unchanged control mean
that it emits no audit records, not that it performs no work.

All 12 compiled kernels declare zero private storage. Timed window WU/state/
output kernels use 83/126/84 VGPRs and 11,136/43,652/29,312 LDS bytes. Set-bit
versions use 78/119/78 VGPRs with the same LDS. These declarations are neither
measured occupancy nor an explanation of timing.

Keep both variants isolated: the product U=V geometry shows no complete gain,
and individual samples overlap. This does not establish a model-token boundary,
TTFT improvement or release acceptance. The qualified 24,835.3024 ms TTFT and
all performance/long-context gates remain unchanged.

Evidence: [complete native record](../benchmarks/correctness/group-window-gdn-native-components-20260918.json),
566,700 bytes, SHA256
`cc035456bc8dbaa11a0975bf3101030c3aaae45cec6ff54a60a0bb702aa4e2f6`.
It pins `run-native-group-window-gdn-r1.ps1`, baiying, source, compiler, executable,
all input/table hashes and matching original/external boundaries for
`D:\models\Qwen3.6-35B-A3B`. The model is a capture reference, not loaded inference.
