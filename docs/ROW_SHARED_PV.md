# Exact PV candidates grouped by query row

This isolated component reorganizes the original compacted candidate queue
into one column list per query/head. It preserves every candidate identity,
the original four-lane K16 integer dot, both K16 endpoints in each K32 tile,
REGISTER rescaling and the original reciprocal. No numerical bound, native
producer or runtime dispatch changes.

Two kernels compare K128 probability staging in shared memory with wave
broadcasts from four loader lanes. Each 128-thread block owns one query/head
and handles 32 selected columns per pass. Empty rows return uniformly;
unused candidate slots still participate in all broadcasts and barriers.
Independent selected columns can be stored in any order. The input queue
is immutable. Scratch for 128 queries is 1056768 bytes: 1048576 column bytes
and 8192 count bytes. Count reset and queue scatter belong to timed replay.

The native fixture retains the original REGISTER control and independent
CPU recurrence from the prior PV fixture. Empty, full and sparse queues are
tested in reverse order, across eight causal boundary shapes and six
operand families, including signed zeros, subnormals, nonfinite values and
tiny rescaled carries. Every raw output, accumulator, denominator and error
surface, complete row membership, unused scratch tail, guard and immutable
input is checked. Both launch policies reject twelve invalid calls each.

Captured comparisons use original GB10 q7169 Q/K/V and context. The q8192
extension repeats the first 1023 captured rows and has original arithmetic
coverage for that extension. The unchanged native PV producer, original
candidate collection, extra row preparation and replay are timed together
with one warmup and three rotated completed-host samples. Common QK and
probability generation, allocation and validation are outside that component
clock; the common V transpose is also reported. This is not model TTFT.

Native validation is pending. The qualified model control remains
23353.80795 ms and all broader performance and release goals stay open.
