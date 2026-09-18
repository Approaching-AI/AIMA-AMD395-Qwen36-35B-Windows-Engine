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
The component comparisons leave runtime dispatch and packaging unchanged.

The ASan/UBSan host audit checks all 65536 BF16 encoding predicates, including
16642 admitted encodings, 266410 ordered carry states and 4262560 products
against independent original integer arithmetic. It covers zero tails,
minimum-scale cancellation, maximum growth and conservative carry endpoints.
The first host build required an explicit int16 cast in the test's Value
initializer; the arithmetic source was unchanged by that fixture fix.

Both native component revisions pass. The complete-attention fixture checks eight
boundary shapes and nine data families across four layouts. It includes
domain extrema, nearby excluded exponents, subnormals, cancellation and causal
tails. GPU classifications are compared with complete CPU-derived flags and
remain immutable. Every raw score and native PV surface is checked before
replay, then every complete result and candidate count is checked afterward.
All 288 generated configurations pass in each revision. Captured native
surfaces are checked before replay on warmups; complete results are checked
on every warmup and timed attempt.

The q8192 capture repeats the first 1023 rows of real q7169 Q/K/V. All original
29364224 GB10 context cells remain comparison-only; repeated rows use original
arithmetic without a new external capture. One warmup and three rotated
completed-host samples include classification dispatch checks, both QK
kernels, original fallback and the complete retained probability/PV pipeline.
One-time row classification is reported separately and must be added to a
candidate total. Allocations, transfers, resets and validation are outside
timing. No model TTFT, continuation, package or release claim follows.

| Revision / layout | Median attention ms | Classification ms | Combined ms |
| --- | ---: | ---: | ---: |
| R1 retained 2x2 | 599.7330 | 0 | 599.7330 |
| R1 narrow 2x2/K128 | 564.5672 | 1.3174 | 565.8846 |
| R1 narrow 4x2/K64 | 551.2840 | 1.3174 | 552.6014 |
| R1 narrow 2x4/K64 | 545.2129 | 1.3174 | 546.5303 |
| R2 retained 2x2 | 605.9593 | 0 | 605.9593 |
| R2 narrow 2x2/K128 | 553.3776 | 1.3104 | 554.6880 |
| R2 narrow 4x2/K64 | 552.8363 | 1.3104 | 554.1467 |
| R2 narrow 2x4/K64 | 544.8790 | 1.3104 | 546.1894 |

R1 source is `d292b0b`; R2 is `397ddc8`. R2 reuses a Q staging word before
staging becomes live, with a barrier before reuse. This removes four LDS
bytes: the 2x2 tile uses 32768 and wider tiles 24576 bytes. No candidate has
private allocation or compiler-reported spills. Static metadata does not
establish actual occupancy or the cause of the measured difference.
All 545259520 score slots and 33554432 outputs match original arithmetic;
all 29364224 original captured context values match GB10. Each variant has
3127598 exact PV candidates. Both fast and original tile paths execute.
[Complete source, commands, results and audit](../benchmarks/correctness/narrow-domain-qk-components-20260918.json):
240342 bytes, SHA256
`e4ea81a71a81fdf86ac4715f460017bda949038e1b4386d0396bb27984b63921`.

The next provider experiment uses the 2x4/K64 layout under
`QRT_CK_SM121_NARROW_DOMAIN_QK=1`, default off. It requires the existing
exact-attention and prepared-QK owners and rejects competing exponent-mask,
RZ-tree or native matrix routes in its eligible cold scope. Decode, suffix
and counts above 8192 retain existing dispatch. A mutex-owned 589832-byte
allocation stores original-input domain flags and two tile counters. Every
layer/call refreshes classification on the same stream; release frees it.
The native fixture calls the production workspace callbacks. All 288 boundary
configurations and the complete captured q8192 comparison pass again at
source `1c2770d`, including the unchanged arithmetic headers. The selected
complete attention total is 548.5977 ms versus 607.1770 ms control. The 46
explicit owner-policy cases and existing exact/fused policies pass ASan/UBSan.
[Provider build and native checks](../benchmarks/correctness/narrow-domain-qk-provider-native-20260918.json):
195055 bytes, SHA256
`8447171ec1b87c8e73f3521822032c178cd5db51d501751b57fbeda02160ead4`.

Four fresh real-model q8192/out512 processes on baiying use one CK DLL and
change only the narrow-domain flag. Whole `ddacdc9` with final query/output
liveness, MoE `9235750`, FLA `7b20c90` and CLI `24c4304` remain fixed.

| Order / mode | Load ms | TTFT ms | TPOT ms |
| --- | ---: | ---: | ---: |
| 1 / OFF | 21403.5530 | 23805.0149 | 100.776659 |
| 2 / ON | 21362.0551 | 23365.6036 | 101.032919 |
| 3 / ON | 21236.7994 | 23342.0123 | 101.322312 |
| 4 / OFF | 21619.9015 | 24369.4573 | 103.205164 |

Every original prompt, all 2048 GB10 IDs, first logit 10.375/error 0 and actual
callback pass. Nine cold attention calls activate the candidate; original q1
calls retain their prior path. The 130 dense, nine coarse OUT, 40 compact MoE
and 30 adaptive-linear correction counts are unchanged. The first analyzer
omitted ten normal q1 diagnostics; its corrected observer requires nine cold
and ten q1 calls equal to the retained baseline. Native data and numerical
thresholds were unchanged and no model rerun was needed for that correction.

OFF/ON median TTFT is 24087.2361/23353.80795 ms, a 733.42815 ms difference
(3.0449%). Both ON samples beat both OFF samples. OFF varies appreciably;
the full observations remain attached and the median difference is not a
universal savings estimate. ON becomes the next experimental control.
Its load median is 21299.42725 ms; no decode gain is established. All code
and package defaults stay off. Other prompts, prefix/long contexts, total
peak memory and release gates remain open. The 10000 ms first boundary and
4187.415605 ms retained target remain unchanged.

The 1906176-byte CK DLL is
`D:/projects/AIMA-public-narrow-domain-qk-provider-20260918-r1/build/narrow-domain-qk-provider/qrt_ck_fmha_sm121.dll`,
SHA256 `0e889459cc837c3733ef613ff789cbfecd966efecaeb1633c91f41a4410b351d`.
[Four complete model runs](../benchmarks/correctness/narrow-domain-qk-product-20260918.json):
1557593 bytes, SHA256
`fa87b27cacfcc50c67f5e89e47abfed25cce394a12d1947a1ca2193a07187273`.
