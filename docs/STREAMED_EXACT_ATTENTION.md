# Streamed exact attention experiment

The isolated producer in `native/providers/ck_fmha/streamed_exact_attention.h`
keeps 32 decoded query rows in a block. Each original K32 tile feeds exact
QK, ordered softmax and native PV before the next tile begins. This combines
three complete stages and removes the global score matrix from timed calls.
The runtime does not dispatch this candidate.

The producer retains the original K16 carries, full raw-dot fallback, native
EXP correction table, VLLM reduction order, K32 denominator recurrence, BF16
probabilities, register PV rescaling and final error bound. The existing
collector and full selective exact PV replay still execute. Probabilities
and alpha are written for that replay. This is not an approximate denominator
or a reduction in the correctness boundary.

The native test compares every live probability, scale, output, accumulator,
denominator, error and candidate count with an independently computed original
QK control. Untimed diagnostic calls also compare all score bits. Timed
score-free calls verify that the score allocation remains untouched. Eight
boundary shapes and five operand modes cover zeros, signed zeros,
cancellation and inputs requiring the complete QK fallback. All owners have
redzones, and original/prepared/transposed inputs are checked for mutation.

Captured tests use the original q7169 layer-three Q/K/V and GB10 context.
The q8192 component geometry repeats the first 1023 rows; it is not a new
model prompt. A same-executable comparison rotates the retained split stack
and this complete producer, each followed by the original selective replay.
All attempts are checked. Component clocks exclude validation and common
preparation and do not establish model TTFT or inference acceptance.

## Complete fusion result

Source `f0fbc91` passes all 80 generated cases, the complete q8192 component
surfaces and all 29364224 original GB10 context cells. Both arms select
3127598 original PV replays. Completed attention grows from 636.8399 ms to
1067.5065 ms, with common preparation of 3.7360 ms outside both clocks.
The candidate is not integrated. Compiler metadata reports 192 VGPRs,
55552 bytes of shared storage and 320 bytes of private storage per work item;
this alone is not a measured occupancy or bottleneck explanation.

[Pinned native evidence](../benchmarks/correctness/streamed-exact-attention-native-20260917.json):
94708 bytes, SHA256
`3d9cba458d2df82cf68f46efe33530c22705f95afc44932d75a87c1a0490bcb6`.

The revised consumer retains the independently parallel 2x2 exact QK
producer and its complete fallback scan. Only softmax and native PV share a
kernel. This removes repeated global probability/alpha reads by the native
PV stage while preserving their writes for original exact replay. It retains
the same 32-query tile and every arithmetic/error-bound operation. The
updated test compares all original score bits on every attempt, as those
scores are now an input to the fused consumer.

Source `d56ef47` passes the same 80 generated cases and all captured surfaces.
q8192 complete attention improves from 633.5608 ms to 593.2034 ms;
q7169 improves from 456.5003 ms to 421.3265 ms. Common preparation is
3.7559 ms and 2.4290 ms respectively. Original PV candidate counts remain
3127598 and 2198673. All 29364224 GB10 context cells pass in both geometries.
Compiler allocation is 192 VGPRs, 2304 shared bytes and 12 private bytes.

[Fused consumer component evidence](../benchmarks/correctness/fused-probability-pv-native-components-20260917.json):
102600 bytes, SHA256
`9a169d62177677ac700a7be5b34c51a65f1f1ee3b914f406f961c6bbf4531b4b`.

## Default-off provider integration

`QRT_CK_SM121_FUSED_PROBABILITY_PV=1` requires the qualified combined exact
attention option for cold prefill from 2 through 8192 queries. Decode,
nonzero-start suffix calls and longer prefill retain existing dispatch.
Malformed option values and incompatible cold configurations are rejected.
The consumer reuses the validated EXP owner and caller's scratch/stream.
It adds no persistent allocation, retains original collection and exact PV
replay, and drains failures through the existing outer owner. Completed
profile stage 1 includes both softmax and native PV; stage 2 is an empty
completion boundary. The activation marker states this changed stage scope.

Actual launcher, provider and real-model qualifications are pending.

No package, default or release gate changes.
