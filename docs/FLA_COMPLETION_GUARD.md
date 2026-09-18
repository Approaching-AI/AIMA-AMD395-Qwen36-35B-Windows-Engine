# FLA completion guard

The September18 full128k-prefix run stopped during layer33 of its fifteenth
owner chunk. It exited5 after4219017.852 ms, before producing any output.
Fourteen chunks through114688 tokens completed. All14255 preceding logged
FLA intervals were finite and at most6.039 ms. The original failure message
omitted the failed interval and its operation name. The record cannot
distinguish execution delay, scheduling delay or an invalid GPU clock value.
Host, memory, boot and cleanup checks passed. See the
[original failure record](../benchmarks/correctness/long-final-pv-prefix128k-guard-failure-20260918.json).
This is a failed context qualification; it supplies no128k correctness or
performance result.

The completion wrapper now measures an independent monotonic host interval
from before recording the start event until after successful end-event
synchronization. It retains the100 ms bound. A finite, nonnegative GPU
interval within that bound remains sufficient. If that interval is invalid
or exceeds100 ms, a finite, nonnegative host interval within100 ms independently
proves that the enclosed submissions completed within the same bound.
If neither clock supplies that evidence, submission stops.

HIP creation, recording, synchronization and elapsed-time API errors retain
their failure paths. Partial submission is drained before scratch can be
released. Deferred segment stages still rely on the single outer completion.
There is no retry, extra device work, changed arithmetic, extended deadline
or altered segmentation.

Every host-clock fallback or failed bound reports the operation, raw GPU
time, enclosing host time, selected clock and completion result in
`FLA_COMPLETION_GUARD`. Profiling retains the raw GPU value and marks it
invalid when only host evidence is usable. A generic `sequence_ms` result
then uses the accepted host interval; it must not be treated as a valid GPU
profile sample without checking the adjacent diagnostic.

Host tests cover121 clock pairs, exact and adjacent100 ms boundaries,
negative/nonfinite values, monotonic interval measurement, the actual
submission wrapper and segment wrapper, deferred completion, HIP errors,
stream mismatches and cleanup. Local FLA regression runs113 tests, with
one existing Linux parent-death-signal test skipped on macOS. Native build, real-model regression
and a new long-context run remain required. No cause or successful repair of
the original128k failure is claimed from these host tests.
