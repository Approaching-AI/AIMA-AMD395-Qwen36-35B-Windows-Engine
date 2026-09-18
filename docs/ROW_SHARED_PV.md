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

Source `d2a275abb440e5402336c7f75b2cca9dc096e736` completes build, 288
generated configurations and both complete captures on baiying. All 43794432
generated output positions, 768 independent CPU samples, 24 invalid-call
rejections, raw surfaces and memory checks pass. Each captured attempt
matches all 29364224 original GB10 context cells and the complete original
candidate set. q7169/q8192 retain 2198673/3127598 selected cells.

| Captured shape | REGISTER control ms | Shared K128 ms | Wave broadcast ms |
| --- | ---: | ---: | ---: |
| q7169 | 173.5761 | 229.3993 | 272.0729 |
| q8192 | 268.6239 | 334.8396 | 404.4658 |

These complete PV component medians include the common V transpose.
Both schedules regress and remain outside runtime dispatch. The shared
variant uses 102 VGPRs versus the control's 71; wave broadcast uses 92 VGPRs
and reports two SGPR spills. All three declare eight private bytes. Static
metadata alone does not identify the cause of the observed slowdown.

[All source/binary hashes, commands, samples and validation](../benchmarks/correctness/row-shared-pv-native-components-20260918.json):
288407 bytes, SHA256
`c7a443b637349f973e6075dd924eac6bd6fdf3916e4494f61044de6205f248c0`.
The command file is `run-row-shared-pv-r1.ps1`. All 336 compiler inputs are
audited against the source commit. In control reports the row-membership
flag denotes original candidate membership; only the two candidates create
row scratch. No model run follows this rejected component comparison.
The qualified model control remains 23353.80795 ms and all broader
performance and release goals stay open.
