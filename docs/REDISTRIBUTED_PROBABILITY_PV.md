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

Native measurements are pending. Register declarations alone do not establish
occupancy or a performance benefit. The qualified model baseline remains
23902.4417 ms and all original performance and context targets remain open.
