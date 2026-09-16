# Exact K16 exponent masks for QK

Component source `11ebed9` preserves each BF16 value in the high half of its
prepared word. Four low halves per K16 hold the row maximum and masks for
its top three exponents. Intersections certify the exact paired maximum;
the carry-aware arm also accepts a carry exponent that dominates the upper
bound. An unresolved group scans the original sixteen exponent pairs.
Products, ordered K16 accumulation and complete K256 exceptional replay
remain unchanged. The representation needs no additional allocation or LDS.

The host checks cover all 65,536 BF16 encodings, 327,680 maximum certificates
and 262,144 chained groups, including declined-output ownership and immutable
inputs under ASAN/UBSAN. Full local checks pass 54 Rust tests, clippy, 473
Python tests (two skipped), C/q16 ABI and public hygiene. A post-test metadata
writer used a wrong production filename; its recovery and original complete
test log are recorded. The full suite was not repeated for that record fix.

On baiying, the native build and 66 generated cases pass, with 4,224 independent
CPU dots. Full q7169 and repeated-row q8192 comparisons pass 11,565,072,576 raw
score slots and 1,452 additional CPU dots. Every warmup and rotated sample
checks all scores, unused output tails, input bytes, prepared words/masks/flags
and external guards. q8192 repeats 1,023 rows from the immutable layer3 q7169
capture; it is an operator extension, not a new model prompt or GB10 token run.

| Route | q7169 total ms | q8192 total ms | q8192 scores ms |
| --- | ---: | ---: | ---: |
| Retained prepared scalar | 274.3129 | 351.2893 | 341.8868 |
| Exact exponent masks | 239.3425 | 310.6646 | 307.0304 |
| Masks plus carry bound | 231.1551 | 295.7922 | 292.1580 |

Totals add the representation's measured query/key preparation to completed
score execution. Allocation, resets and validation are outside the timers.
Preparation order is fixed, so these measurements do not establish a separate
preparation speedup. All three candidate score samples are below all three
control samples. Both new kernels and the scalar control declare 130 VGPRs,
16 KiB LDS and zero private bytes, with wave32; these are static declarations.

The carry-aware route advances to a default-off product experiment through
`QRT_CK_SM121_EXPONENT_MASK_QK=1`. It requires the existing prepared decoded,
float-alignment and compact-PV configuration for cold calls through q8192.
The arena is refreshed on every call and the matching score producer is
selected at the same time. Single-query decode and other call ranges retain
their existing route. The option adds no workspace. Product retention requires
a same-DLL real-model comparison with the unchanged GB10 continuation gate.

Evidence: [source, commands and complete component comparisons](../benchmarks/correctness/exponent-mask-qk-native-components-20260916.json),
SHA256 `89bd3e265307310e00645aaa24a53f3b4b8c57d91d8fb17304346958b894c2ca`.
These component results do not establish TTFT, prefix, long-context or release
acceptance. The 10,000 ms gate and 4,187.415605 ms retained target remain open.
