# Adaptive strict attention-denominator refinement

Source `1f13077` revisits the strict selective-QK component with cached
probability intervals and adaptive work selection. Original native score
bounds, exact prefix maxima and BF16 probability repair precede independent
original prepared-dot replay. The candidate computes its own canonical PV
numerator from those probabilities and exact alphas. Each token/head then
refines its unrounded denominator until every BF16 context output has fixed
interval endpoints. The final round replays all remaining scores. Neither
canonical scores nor external GB10 values choose work or enter computation.

The first implementation checks 18 rounds, halving each row's maximum
probability-width threshold. Cached intervals avoid repeating table evaluation
for unchanged scores. Every selected replay updates its interval with the
exact original scalar probability. A certified row may use the lower
denominator endpoint because the complete reciprocal/output interval has
already fixed all BF16 results. This is a strict context certificate, unlike
the earlier approximate-denominator product trial that failed continuation.

All 35 generated cases pass 981520 causal score comparisons, 623812 selected
score bit comparisons, 6307840 final context cells and 80 independent CPU
dots. Cases include zero, cancellation, subnormal inputs, exponent spread and
causal tails through 129 tokens; two rows exercise complete fallback. The
q7169 capture preserves every one of 29364224 external GB10 context cells.
Its selected fraction falls from the historical uniform-budget 88.72% to
36.41%, but candidate QK/probability takes 2307.8093 ms against 333.1643 ms
for the original prepared QK and probability kernels.

Source `065b258` adds two independent comparisons. Shared-tile replay groups
selected cells into 16-query/16-key bitsets, publishes each nonempty tile
once, compacts its selected dots and reuses packed operands through two
128-feature shared windows. Original K16 carried arithmetic and exceptional
full-dot restart are preserved. A six-round schedule advances four priority
bins at a time; its final complete-replay certificate is unchanged. All four
combinations run in the same executable, with separate completed clocks for
collection, replay, probability initialization and certification.

All 140 generated reports pass 3926080 causal scores, 2499786 selected score
bits, 25231360 final output cells and 320 independent CPU dots. Tile checks
cover complete selected-cell membership, every set bit, unique tile ownership,
causal geometry and all redzones. All eight q7169/q8192 capture reports pass
every probability, alpha, denominator and output check. Each compares 29364224
external GB10 prefix cells. q8192 repeats the first 1023 captured Q/K/V rows;
all 33554432 context outputs match original arithmetic, but only the original
7169-token prefix has the external GB10 context reference. This is a component
shape, not a new real-model q8192 inference or continuation gate.

Matching schedules have identical selected-score and pending-row counts in
the scalar and tile versions. The 18-round scalar counts also exactly match
the initial source on all generated and captured cases. Count identity is
diagnostic; complete numerical and GB10 context comparisons establish this
component's correctness boundary.

| q8192 replay / rounds | Replayed scores | Original QK + probability ms | Candidate QK + probability ms |
| --- | ---: | ---: | ---: |
| Scalar / 18 | 195925421 | 433.0500 | 3001.7861 |
| Shared tile / 18 | 195925421 | 430.5287 | 4488.7601 |
| Scalar / 6 | 267244502 | 445.1504 | 2177.5395 |
| Shared tile / 6 | 267244502 | 438.1677 | 2440.9528 |

The 18/6-round schedules replay 36.4895% / 49.7721% of 536936448 causal
scores. All 131072 rows certify; both schedules have 1509 complete-replay
fallback rows. The six-round scalar stages take 265.9115 ms collection,
1089.2674 ms replay, 310.2262 ms initialization and 396.6034 ms certification,
plus 115.5310 ms bounded score production. Shared-tile replay is slower in
both schedules. Collection and initialization alone exceed the complete
original QK/probability clock in every q8192 variant. Eliminating replay
would therefore still not rescue this implementation.

These clocks include completion events. Tile replay includes mask/owner/count
clears, marking and queued arithmetic. Test downloads, comparisons, allocation
and guards are outside stage clocks; guarded process walls are attached.
Preparation and candidate canonical PV are separately reported. The retained
production approximate/corrected PV implementation is not timed here, and no
TTFT is measured.

Local C ABI and public hygiene checks pass for both revisions; no new complete
Rust/Python suite is claimed for these isolated GPU components. Both native
builds and every run pass host guards. Initial/revised builds take
13002.994 / 13235.617 ms. All kernels declare wave32. Scalar and tile repair
declare 130 VGPRs and zero private bytes; tile repair additionally declares
17412 LDS bytes. Initialization declares 35 VGPRs, marking 10, and the
certificate 16 with eight private bytes. Static resources do not measure
occupancy or explain the timing by themselves.

Keep all variants isolated and retain the current production stack. The best
component remains about 4.89 times slower than its original control. Continue
with the routed/shared MoE down-projection consumer and complete residual
boundary; the earlier gate/up pointwise audits already showed small removable
fractions. All mission and correctness gates remain unchanged. No prefix,
long-context, package, retained performance or release acceptance follows.

Evidence:

- [Initial adaptive source, native cases and captures](../benchmarks/correctness/adaptive-denominator-qk-native-20260917.json), 180479 bytes, SHA256 `f7cfb759a9de6f9893d683d3f6a90fd32d4e419d71f8def2f8dce1b7cc7df337`.
- [Four variants, full phase clocks, numerical checks and decision](../benchmarks/correctness/adaptive-denominator-qk-tile-native-20260917.json), 484664 bytes, SHA256 `f76074737cb9644dfd01c55356dbe9a2aa67160747a87eba47c7c632604f2b59`.
