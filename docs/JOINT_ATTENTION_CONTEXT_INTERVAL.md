# Joint attention-context interval

Source `ed1354c` adds an isolated BF16 output certificate for uncertain PV
numerators and attention denominators. It checks their complete Cartesian
product, rather than requiring an exact numerator before denominator
refinement. No runtime dispatcher uses it yet.

The caller must provide valid arithmetic enclosures and the original
SHA-verified reciprocal table. For denominator intervals in `[1,2^19)`, the
helper encloses possible reciprocal-table excursions with two FP32 output
steps at each endpoint. A point denominator uses its exact table result.
All four rounded numerator/reciprocal corners must produce the same finite
BF16 value. Nonfinite, reversed and subnormal product intervals decline;
nonzero underflow to zero also declines. Signed-zero point intervals preserve
their sign, while mixed signed zeros decline. Declining leaves output intact.

The host audit uses UBSan and independently enumerates representable points
near BF16 rounding boundaries. It covers 65536 intervals, all 19 reciprocal
exponents, both numerator signs, 9732096 admitted Cartesian points and
2162688 reciprocal points. Half the generated intervals admit and half
decline. Exceptional inputs and null pointers are checked separately.

On baiying/gfx1151, all 65550 native cases match the host decisions and
outputs, including 14 exceptional/signed-zero cases. The same 9732096
Cartesian points and 2162688 reciprocal points pass with zero false
admissions or escaped reciprocals. Input bytes, table bytes and all redzones
remain intact. Build/test processes complete normally with every host guard.

This qualifies only the arithmetic helper. It does not establish bounds on
model-derived numerators or denominators, reduce runtime replay, or qualify
model tokens or performance. The next complete component must construct its
own enclosures, preserve original QK/PV fallback, and compare against original
arithmetic and the GB10 context. Golden values remain comparison-only.

[Pinned source and native evidence](../benchmarks/correctness/joint-attention-context-interval-native-20260918.json):
85759 bytes, SHA256
`95c50250e514e5f98dde5b274bc35aeccbc2457b212330da019bea10874454d6`.

The qualified experimental model baseline remains 24835.3024 ms q8192 TTFT.
The 10-second boundary, retained performance, long contexts, package and
release acceptance remain open.

## Complete attention scheduler, 2026-09-18

Source `a225ea1` constructs numerator and denominator enclosures from the
candidate inputs. It retains the existing native QK bound, exact prefix-max
and BF16 probability repair, and native PV with unit-denominator scales.
Ambiguous numerator columns receive original PV replay. A single weighted
uncertainty histogram selects original QK work; strict context certificates
and complete remaining-score/PV fallback decide acceptance. Golden scores,
denominators, numerators and GB10 values never select candidate work.

The refactored arithmetic helper passes the same UBSan host audit. On
baiying/gfx1151, 80 generated configurations pass, including causal tails,
signed zeros, cancellation, subnormal and wide exponent inputs. Every q8192
attempt checks all 536936448 causal score enclosures, original BF16
probabilities and alpha values, original denominators, and 33554432 original
raw PV numerator enclosures/context outputs. All 29364224 captured GB10
context cells match. The additional 1023 rows repeat original Q/K/V and use
original arithmetic as their observer; they have no new GB10 capture.
All 131072 query/head rows certify. Input, table, prepared/transposed storage,
redzones and unused tensor tails remain intact.

| Complete q8192 attention | Current control | Joint intervals |
| --- | ---: | ---: |
| Samples, ms | 584.9506 / 589.2833 / 581.9830 | 1468.6504 / 1472.7099 / 1472.3998 |
| Median, ms | 584.9506 | 1472.3998 |
| Original score cells evaluated | 536936448 | 391202410 |
| Original PV cells replayed | 3127598 | 3129599 |

One warmup and three rotated samples run per 128-query slab. Candidate clocks
include every producer, selector, certificate and full fallback, without
intermediate host waits. Original full PV observers, allocations, resets and
checks are outside timing. Common preparation adds 3.7317 ms; common EXP
construction and full-domain verification add 9.0286 and 5.7869 ms.

Keep this scheduler outside runtime dispatch. It remains 887.4492 ms slower
than the current control and still evaluates 72.8582% of original scores.
The candidate does not materially reduce PV replay. Its fixed selected QK
kernel has 118 declared VGPRs, 32800 LDS bytes and no private allocation;
these are compiler declarations, not occupancy or timing explanations.
No new model-token run, performance acceptance or package change follows.

[Complete source and native evidence](../benchmarks/correctness/joint-context-attention-components-20260918.json):
124133 bytes, SHA256
`cec03057ccc517d012f8251fb0c84fb0709e53bf8ac1aaac37c5e07db14c606a`.
