# Lossless scaled-half operands in four-score QK

Source `a97e04660162fbb74850b0fc9483b0c722d31cb2` combines the established
lossless scaled-half K16 row representation with a 32x32 score tile. Each
thread owns four independent score cells. Full integer carries keep original
per-group fallback; FP32 carries mark the entire affected dot for original
replay. All ascending K16 carries, exact mixed scalar products and canonical
normalization remain unchanged. Product dispatch is unchanged.

Both native arithmetic arms pass 131,072 raw carry-state comparisons against
the independent original integer primitive, including all 65,536 BF16
encodings and 16 controls. Together they check 4,194,304 original products.
All 72 generated configurations pass 98,544,624 score comparisons and 4,608
CPU dots. Cases include partial 31/32/33-row tiles, causal masks, signed zero,
subnormals, late unsupported inputs, overflow and an invalid operand in only
the second score owned by a thread.

At q8192, every warmup and timed attempt compares all 545,259,520 score slots
per variant. The three arms total 6,543,114,240 checked scores and 768 further
CPU dots. Complete prepared encodings, original inputs, guards and unused
tails remain intact. This geometry repeats the first 1,023 rows of an original
q7169 capture; it supplies no new model prompt or external context/token gate.

| Route | Query including encoding/replay ms | One-time preparation ms | Total ms | VGPRs | LDS bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current decoded 2x2 | 293.0037 | 9.2971 | 302.3008 | 117 | 32768 |
| Scaled-half 2x2, integer carries | 284.3689 | 0.3407 | 284.7096 | 91 | 18432 |
| Scaled-half 2x2, FP32 carries | 363.8384 | 0.3407 | 364.1791 | 49 | 18432 |

One warmup and three completed samples rotate the arms across 128-query
slabs. Candidate query encoding belongs to every slab clock; each total also
charges its key preparation. The control charges its full Q/K preparation
and key transpose once. All original fallback work is included. Allocation,
reset, reference generation and comparison remain outside timing.
All three score kernels declare wave32 and zero private storage. These
resource declarations do not establish occupancy or a timing explanation.

Keep both candidates isolated. The best complete reduction is 17.5912 ms,
with only 8.6348 ms in the query clock. The FP32 variant regresses despite its
lower register declaration. This result does not establish a seconds-scale
product route while TTFT remains above 10 seconds. No unchanged q7169 repeat
or provider integration follows. The model baseline remains 24,709.6139 ms;
package, retained-performance, long-context and release gates remain open.

Local C ABI smoke and public hygiene pass. The bounded Windows/HIP build,
generated safety and complete q8192 actions complete in 13,896.307, 1,855.798
and 6,523.304 ms with all host guards passing. No full model is loaded.

Evidence: [source, commands and native results](../benchmarks/correctness/microtile-half-qk-native-components-20260918.json),
145,111 bytes, SHA256
`8011d9cface7612176e802624e5ab5275d5d9dedc2c01a03c07df8f855e207fb`.
It pins baiying, `run-native-microtile-half-qk-r1.ps1`, all source/build/input
hashes and the `D:\models\Qwen3.6-35B-A3B` capture reference.
