# Inline probability repair experiment

This isolated component replaces the joint-context scheduler's multiple
score-selection, probability initialization and histogram passes with one
wave per query/head. Each original K32 tile repairs all possible prefix
maxima and ambiguous BF16 probabilities using the unchanged prepared K16
dot, with a full original-dot restart for unsupported inputs. The wave then
propagates original-order lower, center and upper denominator recurrences.
It stores probabilities and alphas for native PV and exact selected PV replay.

The score bound and complete Cartesian context certificate are unchanged.
An unresolved context row triggers one complete remaining-score replay using
the original 2x2 microtile. Its denominator is then recomputed, and any
remaining ambiguous numerator receives original PV replay. A row must pass
the original certificate or its exact-point fallback. No reference values
select work, no coefficient is reduced, and the runtime does not dispatch
this experiment.

This removes global lower/upper probability arrays, per-score queues and
priority arrays, repeated interval scans and the histogram. It does add
divergent scalar QK work inside the first probability traversal. Native
measurements must determine whether the removed passes outweigh this cost
and the full-row fallback fraction.

The native fixture first checks the initial probability pass independently,
before native PV or complete score fallback can hide a repair error. It then
checks the complete attention result, every causal score enclosure, BF16
probability and alpha, denominator and full original PV numerator enclosure,
all context outputs, guards, unused tails and input immutability. Eight
boundary shapes and five data families precede the captured comparison.

The q8192 component repeats the first 1023 rows of real q7169 layer-3 Q/K/V.
The original 29364224 GB10 context cells are comparison-only; repeated rows
use original arithmetic without a new external reference. One warmup and
three rotated completed-host samples include the entire candidate pipeline
and all fallback. Preparation, allocation, transfers and observers are outside
timing. Source `878d8c0` passes all 80 generated control/candidate configurations
and every captured comparison on baiying. The independent initial-pass check
also passes for every generated case and all 64 captured query slabs.

| Complete q8192 attention | Retained control | Inline repair |
| --- | ---: | ---: |
| Samples ms | 590.5098 / 587.9386 / 588.7369 | 1406.2032 / 1406.7957 / 1407.7085 |
| Median ms | 588.7369 | 1406.7957 |
| Original score evaluations | 536936448 | 535611958 |
| Original PV replays | 3127598 | 3129599 |

The first pass repairs 17280213 scores (3.2183%). After numerator repair,
130495 of 131072 rows (99.5598%) still need complete remaining-score fallback.
Final original score work is 99.7533% of the control. All rows eventually
certify with zero context differences, unresolved rows or escaped intervals.
Removing the global interval arrays and repeated scans therefore does not
remove enough original work to compensate for the added pipeline.

Keep the candidate outside runtime dispatch. The initial probability kernel
declares 130 VGPRs and no private bytes; remaining-score replay declares
122 VGPRs, 32800 LDS bytes and no private bytes. These are static resources,
not measured causes. All 269 compiled source inputs match the pinned commit.
The qualified model baseline stays 23902.4417 ms, and no model, package or
release acceptance changes.

[Commands, all comparisons and decision](../benchmarks/correctness/inline-probability-repair-components-20260918.json):
119290 bytes, SHA256
`bedb77ac17bc71d534c2fe22c527c63578a7c2d59bf2c9d01d0eb4213fb8b6d8`.
