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
