# Exact native EXP correction

Sources `4fca895` and `d89c260` add an isolated representation and complete
captured-attention comparison. Runtime dispatch, package assets and retained
settings are unchanged.

## Representation and complete domain

Each negative FP32 input in the original exp2 domain owns a two-bit code.
Three codes adjust the actual gfx1151 `v_exp_f32` output word by -1, 0 or +1;
the fourth evaluates the original SHA-bound interpolated table. No assumed
SFU accuracy bound determines acceptance. The derived table is generated on
the execution device from the original source, independently of model data.
Its 82182144 bytes supplement the 38909480-byte source table.

On baiying, all **328728576** interior inputs compare bit-for-bit with the
original table twice, with no encoding or evaluation error. Counts repeat:

| Correction | Inputs |
| --- | ---: |
| -1 | 77827717 |
| 0 | 215981535 |
| +1 | 32590895 |
| Original-table escape | 2328429 |

The escape fraction is 0.708314%. The largest native output-word difference
is 8388563, so a universal one-ULP assumption would be incorrect. Escapes
preserve these inputs exactly. This is evidence for the tested device and
kernel configuration, not a portable SFU contract.

Table generation completes in 8.381 ms. Two exhaustive checks take
21.2646 / 28.5797 ms; these primitive times do not establish inference
performance. Twenty-six exterior inputs preserve original positive-plateau,
zero, NaN and underflow behavior with null table pointers. Source/derived
table immutability and guards pass. Another 120 generated attention cases
compare 216595776 probability slots and 6848256 scale slots, including the
ordered denominators, with zero mismatches and unchanged inputs/tails.

Local checks pass the related exp2 suite (five tests, one offline NumPy
builder skip), sanitized correction/escape edges, C ABI smoke and hygiene.
The native-only captured fixture also passes local C smoke and hygiene.
No new full Rust/Python suite run is claimed. Native domain build/test and
captured build finish normally in 12565.261 / 5405.159 / 13266.332 ms,
with all host guards passing.

## Complete captured attention

The same executable compares the original and replacement probability
decoders while retaining prepared exact QK, original recurrence order,
direct native PV and all original compacted PV replay. An independent
original tiled-QK control supplies arithmetic comparisons. All raw scores,
probabilities, scales, output, accumulators, denominators, error bounds and
candidate counts agree on every warmup and timed attempt. Guards, unused
tails, full operand encodings and source immutability pass.

| Captured shape | Original complete ms | Native delta complete ms | Original incl. preparation ms | Native delta incl. preparation ms |
| --- | ---: | ---: | ---: | ---: |
| q7169 | 564.3187 | 537.0660 | 566.8222 | 547.2520 |
| q8192 extension | 786.5368 | 745.8377 | 790.2496 | 757.1356 |

Each number is the median of three completed full-shape samples, with one
warmup per 128-query slab and rotated arm order. Preparation totals include
Q/K preparation, V transpose and, for the candidate, one table build.
Probability-only event sums have no invalid intervals and decrease from
123.3210–126.2806 to 81.5717–83.4544 ms at q8192. These event intervals are
nested in complete attention and must not be added to it.

Both geometries preserve all 29364224 original GB10 context cells. The
q8192 shape repeats the first 1023 captured Q/K/V rows, checks all 33554432
output cells against original arithmetic, and has external context only
for its first 7169 rows. Original PV candidate counts are 2198673 / 3127598
and are identical between arms. The 228 / 256 independent CPU QK checks
also pass. Capture processes complete in 7146.590 / 9477.714 ms with host
guards. No new model token loop, TTFT or long-context acceptance follows.

## Decision

Keep the representation isolated. Its complete q8192 component saves
40.6991 ms, or 33.1140 ms after charging preparation, which has insufficient
scope to resolve the current model TTFT gap. Continue a broader canonical
dot/replay or dataflow replacement. Future integration still requires owned
and validated derived storage plus the unchanged real-model GB10 boundary.
The 10000 ms gate and 4187.415605 ms retained TTFT target remain unchanged.

[Complete source, commands, domain, guards and captured GB10 evidence](../benchmarks/correctness/native-exp2-delta-attention-components-20260917.json),
241587 bytes, SHA256
`5aaf703794d84456d95fb54cf2ff47db22e2a61dbbda6e609900cb10c97165a2`.
