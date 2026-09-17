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

Source `df2ea51` qualifies the actual callback and query launcher. All 80
generated cases pass. OFF/ON launcher runs both match the original 29364224
GB10 context elements and produce identical four raw output, accumulator
and denominator files. Both select the original 2198673 PV replays and
separately verify and reuse the complete native EXP owner. The CK DLL builds
on baiying and passes the real-model comparison below.

The full local suite passes 493 Python tests with two environment skips,
C/Rust/clippy and public hygiene. Integration first exposed an outdated
workspace mock signature, which was repaired and extended with owner reuse,
failure-drain and fallback checks. A CK compilation-unit name collision was
then fixed solely by explicit namespace qualification; normalization proves
the arithmetic source unchanged, targeted local checks pass and every native
build/capture is repeated for the final commit. Both failures are retained.

[Actual owner and provider evidence](../benchmarks/correctness/fused-probability-pv-provider-native-20260917.json):
260817 bytes, SHA256
`742453e878d7196f86c76c27f59692b4ef2973beff693a85b0dc64656bd37add`.

## Real-model comparison

Four fresh baiying processes run the original q8192/out512 case in
OFF/ON/ON/OFF order. Both arms use the same source `df2ea51` CK DLL,
1848832 bytes, SHA256
`64c56d84091e979a4110118e026c6eedbfd9280840f683fc261f44b496f32676`.
Whole `6e4908b`, compact MoE `9235750`, FLA `2ee6215` and CLI `24c4304`
remain fixed. Only `QRT_CK_SM121_FUSED_PROBABILITY_PV=0/1` changes;
adaptive OUT2, compact MoE1, combined exact attention and register PV stay on.

| Order / mode | Load ms | TTFT ms | TPOT ms |
| --- | ---: | ---: | ---: |
| 1 / OFF | 21265.4320 | 25311.1634 | 101.569030 |
| 2 / ON | 21373.5027 | 24818.1339 | 101.657110 |
| 3 / ON | 21324.2583 | 24852.4709 | 100.688219 |
| 4 / OFF | 21290.8780 | 25442.5373 | 100.984396 |

All original prompts, 2048 GB10 output IDs and actual callbacks match.
Every first logit is 10.375 with zero error. All ten fused attention owners
activate in each enabled process; other retained owner counts and host guards
pass. OFF/ON TTFT medians are 25376.85035/24835.3024 ms, saving
541.54795 ms (2.1340%). Both ON observations are below both OFF observations.
Retain mode 1 for subsequent experimental q8192 work. TPOT medians are
101.276713/101.1726645 ms; this limited comparison does not establish a
decode gain. All loads are below 30 seconds, but TTFT remains above 10
seconds. Code and package defaults remain off; long-context, retained
performance and release gates remain open.

[Commands, artifacts and all four correctness boundaries](../benchmarks/correctness/fused-probability-pv-product-20260917.json):
965116 bytes, SHA256
`9963b45485de6f944d2f5fadc1c5f588fa64e0c30f128359e019d3551e6a2758`.
