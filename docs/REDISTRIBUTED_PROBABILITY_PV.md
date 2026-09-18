# Probability/PV ownership experiment

The retained fused probability/native-PV consumer declares 192 VGPRs and
12 private bytes in the qualified CK binary. This experiment compares its
32-query/256-thread layout with 16 queries/256 threads and 32 queries/512
threads. Each candidate halves the per-thread PV accumulator and bound state.
QK remains the independently parallel 2x2 producer; this does not revive the
complete QK/probability/PV fusion tested previously.

Both candidates keep original K32 max/sum recurrence, native EXP correction,
VLLM reduction order, BF16 probabilities, ordered K16 native PV, each original
16-query value mask, error bounds and the complete selective exact PV replay.
They introduce no coefficient or selection changes. The retained producer,
runtime dispatch and package are unchanged.

The native fixture checks eight boundary shapes and five data families for
all three layouts. Before replay it compares every native output, accumulator,
denominator, bound, probability, scale and score with the retained consumer.
After replay it compares all surfaces and candidate counts with independently
computed original attention. Guards, unused tails and input immutability are
part of these comparisons.

The captured comparison uses real q7169 layer-3 Q/K/V and repeats its first
1023 rows to obtain q8192. All original 29364224 GB10 context cells are checked;
the repeated rows have original-arithmetic coverage only. Each 128-query slab
has one warmup and three rotated completed-host samples per layout. Warmups
also compare native state before replay. Timed calls submit QK, the consumer
and replay together; every completed attempt is checked. Common preparation,
allocation, transfer and validation are outside timing. This component test
does not load the model or establish TTFT, token-loop or release acceptance.

Source `ab6b9ce` passes all 120 generated configurations and every captured
comparison on baiying. All 545259520 score slots, 33554432 output cells and
3127598 PV candidates match the original control on every q8192 attempt;
all original GB10 context cells match. Native state before replay also agrees.

| Layout | Complete q8192 samples ms | Median ms | VGPRs | Private bytes |
| --- | --- | ---: | ---: | ---: |
| Retained 32 queries / 256 threads | 586.8079, 590.3002, 586.1541 | 586.8079 | 192 | 12 |
| 16 queries / 256 threads | 1265.6953, 1267.8185, 1276.4234 | 1267.8185 | 256 | 228 |
| 32 queries / 512 threads | 1091.4214, 1090.3975, 1096.8114 | 1091.4214 | 256 | 276 |

Both candidates are slower and remain outside runtime dispatch. Halving the
declared accumulator state did not reduce compiler register/private allocation.
The resource metadata alone does not establish occupancy or causality. The
qualified model baseline remains 23902.4417 ms; no model run, package, target
or release state changes.

The same evidence includes a read-only audit of an already qualified model
run. Dense allocation/free totals only 26.2956 ms over 130 owners. The actual
route has 290 host count reads and FLA segment guard disabled. GPU-containing
wait clocks overlap other work and cannot be added as pure host overhead.
These observations motivate checking required exact work before another
allocation-only rewrite.

[Commands, source hashes and all comparisons](../benchmarks/correctness/redistributed-probability-pv-components-20260918.json):
120234 bytes, SHA256
`2ba70924fd1210b43f0cca882acb22e6dd4cab1dd6187dbfadd09ce03129fea5`.
