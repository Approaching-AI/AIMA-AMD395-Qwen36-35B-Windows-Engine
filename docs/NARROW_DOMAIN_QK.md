# Exact QK specialization for a proved input domain

The isolated QK candidate classifies original query and key rows. A complete
causal matrix tile may use its simplified K256 loop only if every operand
is signed zero or a normal BF16 with unbiased exponent in [-32,32]. Other
tiles retain original per-cell eligibility, carried-value checks and complete
original-dot fallback. The classifier reads actual inputs on each call; it
does not infer eligibility from model names or past captures.

Each nonzero product has an original product exponent of at least -64. An
original K16 alignment unit is therefore at least 2^-89. Every nonzero
normalized carried value has exponent at least -89; a group containing only
zero products preserves the preceding carry. The absolute sum of 256 eligible
products stays below 2^74. Thus scales and FP32 carries remain normal/finite,
and each individual scaled product fits the original int32 conversion.
Unsigned modulo summation, sign decoding and original truncation remain.
A -89 maximum floor changes only the all-zero sum, which still returns +0.
These preconditions permit removing per-group exceptional-value checks.

The experiment compares the retained decoded 2x2 QK with three separate
fast/fallback kernel pairs: 2x2 with a K128 shared window, and 4x2 / 2x4 with
K64 windows. All groups retain ascending K16 order. Probability/native-PV and
complete exact PV replay remain the existing retained implementations.
Runtime dispatch and packaging are unchanged.

The ASan/UBSan host audit checks all 65536 BF16 encoding predicates, including
16642 admitted encodings, 266410 ordered carry states and 4262560 products
against independent original integer arithmetic. It covers zero tails,
minimum-scale cancellation, maximum growth and conservative carry endpoints.
The first host build required an explicit int16 cast in the test's Value
initializer; the arithmetic source was unchanged by that fixture fix.

Native validation is pending. The complete-attention fixture checks eight
boundary shapes and nine data families across four layouts. It includes
domain extrema, nearby excluded exponents, subnormals, cancellation and causal
tails. GPU classifications are compared with complete CPU-derived flags and
remain immutable. Every raw score and native PV surface is checked before
replay, then every complete result and candidate count is checked afterward.

The q8192 capture repeats the first 1023 rows of real q7169 Q/K/V. All original
29364224 GB10 context cells remain comparison-only; repeated rows use original
arithmetic without a new external capture. One warmup and three rotated
completed-host samples include classification dispatch checks, both QK
kernels, original fallback and the complete retained probability/PV pipeline.
One-time row classification is reported separately and must be added to a
candidate total. Allocations, transfers, resets and validation are outside
timing. No model TTFT, continuation, package or release claim follows.
