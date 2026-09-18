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
one existing Linux parent-death-signal test skipped on macOS. No cause or
successful repair of the original128k failure is claimed from these host tests.

Native source `2b33665676223a4cded4b1df10437f4d1456aae9` now builds on
baiying, with all45 compiled inputs checked. Its972800-byte DLL SHA256 is
`73bd48b27ff696135b5902d1d693bc4d6a08e7f308f79b63e587c169e1ba83f7`.
Captured q7169 output and final state match the GB10 reference bitwise.
Full q8192/out512 matches every original token and callback, with first
logit10.375/error0. Load is21317.5197 ms, TTFT23327.2313 ms and
TPOT100.546425 ms. No host-clock fallback occurs in that run.

The full16384 owner first token/logit and31 cached continuation tokens match
the original owner32. Both512-token suffix requests, raw first logits,
callbacks, restoration and changed-prefix rejection also pass. Load is
21298.4029 ms; the timed suffix TTFT is8450.1209 ms, TPOT156.542066 ms and
owner continuation4897.6403 ms. One inverse-stage GPU interval is−0.048 ms;
the enclosing monotonic host interval is0.3358 ms, independently satisfying
the100 ms bound. The fallback is recorded and no invalid GPU duration is
accepted as a profile sample. This demonstrates a negative event interval.
The cause of the earlier128k failure remains unestablished.

[Native build, captured comparison and complete product evidence](../benchmarks/correctness/fla-completion-guard-native-product-20260918.json)
attach commands, source/binary identities and original numerical boundaries.
The samples establish no paired speed gain. The new128k run, larger contexts,
package qualification and performance acceptance remain open.
