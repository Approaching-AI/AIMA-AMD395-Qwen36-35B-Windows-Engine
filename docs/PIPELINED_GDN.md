# Overlapping complete GDN segments

This isolated fixture uses the retained paired-score, WU, shared-arena state8
and shared-arena output kernels without changing their arithmetic. Its control
submits all four stages serially for each 1,024-token segment. A second route
keeps preparation and state on one stream while a second stream consumes
completed segments. A third route separates preparation, state and output
across three streams. HIP completion events carry the actual dependencies.
All recurrent state updates remain ordered on one stream.

Each score/WU producer accesses only its own segment. The original WU CTA
owns all rows of its V columns, retaining the U=V alias contract. State writes
separate H/Vnew spans and leaves the original FP32 state for the next segment;
output waits for that segment's completed state event, which also covers its
score producer. Every event belongs to one segment and is reused only after
the complete invocation finishes. Launch failures drain submitted work before
the fixture can release buffers. There are at most eight segments per call.

All variants use the same complete-invocation intermediate allocations in the
existing fixture. Stream/event creation, input loading, reset and observation
are outside the timer; kernel dispatch, dependency operations and device
completion are inside. This does not establish the production provider's
workspace cost: its present owner reuses one segment's scratch, whereas this
experiment keeps every segment's intermediate spans alive through completion.

The shared fixture compares every score, W/U, Vnew, BF16 checkpoint, output
and final FP32 state, including guards and nonaliased inputs, after every
warmup and rotated timed attempt. Independent CPU dots and the original GB10
q7169 capture remain observers. The extended q8192 geometry repeats 1,024 rows
after 7,168 original rows and is not a new model prompt. Safety covers six
families, partial segments and both U ownership modes.

Native results, resource declarations and any product integration must be
recorded separately. This source has no runtime dispatcher, model-token result,
claimed speedup or release qualification.

The first native build at `41231ba` succeeds, but safety stops with a score
comparison failure before any case completes. The inherited completion helper
waits on a default-stream event, which does not cover nonblocking streams.
The corrected owner records one final event on the consumer and makes the
default stream wait on it. That event transitively covers every producer and
state segment before the bounded helper finishes, observers read storage or
another attempt reuses it. The original failed run remains evidence; it is
not an arithmetic comparison or performance result.


The corrected source `1bf6755` passes all288 safety configurations, including
500465664 outputs,250232832 scores and2728230912 state/intermediate values.
Independent CPU checks cover291192 dots and684 score dots across48 cases.
Original q7169 output/W/U/Vnew,113 checkpoints and final FP32 state match GB10.
The q8192 extension preserves all original-prefix boundaries and every extended
score/output/state bit. Every warmup and rotated attempt checks all surfaces,
guards, immutable inputs and both U ownership modes.

| Captured shape / U storage | Serial ms | Two streams ms | Three streams ms |
| --- | ---: | ---: | ---: |
| q7169 / separate | 75.0543 | 68.0423 | 67.0379 |
| q7169 / U=V | 75.0875 | 67.9797 | 67.1359 |
| q8192 / separate | 86.3189 | 77.4624 | 78.2368 |
| q8192 / U=V | 86.7021 | 78.5150 | 77.3428 |

All q8192 candidate samples are below all corresponding serial samples.
The separate-U q7169 ranges overlap. Neither stream count is universally
faster. Keep this as a component building block: about9ms per captured layer
is not a measured seconds-scale model gain, and production memory/lifetime
integration remains unqualified. Runtime dispatch and the23353.80795ms model
baseline are unchanged.

Nine arithmetic source files match the retained `7b20c90` FLA provider exactly.
The same score/WU/state/output kernels declare70/78/113/77 VGPRs and
10276/10816/26180/19264 LDS bytes, with zero private bytes and spills. These
static declarations do not measure concurrency or occupancy. Streams and all
17 events are initialized outside the clock; the final join is timed.

[Complete native evidence](../benchmarks/correctness/pipelined-gdn-native-components-20260918.json):
615176bytes, SHA256
`9b56490f7150c6479fdb2ae6f43a0af0a14302738260eacd6626acd914d8289a`.
It preserves the failed first safety run, both builds, commands, all source
and executable identities, raw reports and original GB10 capture provenance.
