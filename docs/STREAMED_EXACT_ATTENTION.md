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

Native results are pending. No package, default or release gate changes.
