# Reduced EXP correction experiment

The isolated representation in source `da7e3c9` reduces the derived correction
table from 82182144 to 2097152 bytes. The original 38909480-byte SM121 table
remains required. A separate 1048576-byte bitmap is used during construction.
No runtime owner or package selects this representation.

For negative inputs with magnitude in `[1,126)`, index the correction by the
exact fractional part in units of `2^-23`. Derive canonical corrections at
`-(1+fraction)` using the actual gfx1151 EXP instruction. Scan every admitted
input and mark any fraction whose correction is not reusable across integer
classes. Marked fractions, canonical escapes and exterior inputs use the
original table. Construction and verification run on the execution device;
there is no assumed SFU error bound or portable derived artifact.

The sanitized host audit checks all 58458112 admitted FP32 encodings against
the original table after integer exponent scaling, with zero differences.
On baiying, native verification checks all 328728576 original input encodings,
40 boundary/exterior cases, deliberate correction corruption, immutable
buffers and redzones. All pass. No additional cross-integer escapes occur;
57914920 admitted inputs use native correction and 543192 use original-table
fallback. The four canonical code counts are 4664556, 3658349, 433 and 65270.

Source `68a0d45` compares both tables in complete attention. The current
fused probability/native-PV consumer remains the control; QK, reduction order
and every selected exact PV replay remain unchanged. A template policy keeps
the existing correction representation as the production default.
All 80 generated configurations and every q8192 raw intermediate/output
comparison pass. All 29364224 original q7169 context cells also match the
external GB10 capture. The additional 1023 repeated rows use original
arithmetic as their reference. Golden data is comparison-only.

| q8192 component | Current table | Reduced table |
| --- | ---: | ---: |
| Completed attention samples, ms | 590.7785 / 594.3330 / 597.9503 | 595.7752 / 592.5978 / 594.4485 |
| Completed attention median, ms | 594.3330 | 594.4485 |
| Table build and verification, ms | 15.0931 | 5.7791 |
| Common preparation, ms | 3.7816 | 3.7816 |
| Preparation plus median, ms | 613.2077 | 604.0092 |

Each arm has one warmup and three rotated completed host samples per
128-query slab. Complete attention includes fallback and exact PV replay.
Table allocation, uploads, snapshots and comparison observers are outside
the timed intervals. Both compiled consumers declare 192 VGPRs, 2304 bytes
of LDS and 12 private bytes; these are static declarations, not occupancy
measurements. Each arm selects the same 3127598 exact PV candidates.

Keep the candidate isolated. Steady attention timing is effectively unchanged;
the 9.3140 ms one-time table preparation saving does not establish a
seconds-scale model improvement. No model token run or TTFT acceptance
follows. The qualified experimental model baseline remains 24835.3024 ms,
with long-context, retained-performance, package and release gates open.

[Pinned native evidence](../benchmarks/correctness/reduced-exp2-attention-components-20260918.json)
records all commands, source commits, table/capture hashes, build artifacts,
guards and comparisons. It is 185909 bytes, SHA256
`1f00674ab378e52db92b3f37e7cceabb1af99ad38751102acc99709fa22d763c`.
