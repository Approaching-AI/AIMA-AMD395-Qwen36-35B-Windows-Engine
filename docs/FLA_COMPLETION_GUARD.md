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
The samples establish no paired speed gain.

The repeated128k run also fails at layer33 of owner chunk14, after14
completed chunks through114688 tokens. Native wall is4213592.288 ms,
exit5, with no output. The failed `blackwell_output_segment` records
283.885010 ms on the GPU and284.029600 ms on the monotonic host clock.
Both exceed100 ms. Four earlier negative norm-stage GPU intervals pass
independent host bounds. All host and cleanup guards pass. This establishes
a completed delay at the repeated location; it does not identify which of
the score/value kernels is slow or exclude scheduling and memory effects.
See the [completed-delay failure record](../benchmarks/correctness/fla-completion-guard-prefix128k-delay-failure-20260919.json).
No128k numerical acceptance exists.

An opt-in diagnostic `QRT_FLA_GDN_CAPTURE_OUTPUT_FAILURE_DIR` now saves the
failed segment only after successful event synchronization and elapsed-time
query, with finite nonnegative clocks rejecting the original100 ms bound.
It writes Q/K, V-new, chunk states, cumulative gates, scores and completed
outputs to a new directory. Reads use at most1 MiB of host scratch and each
capture is below64 MiB. Existing directories are refused. Failed or partial
copies cannot publish a completion record. Original model failure is retained
regardless of capture success; no kernel is retried or reference supplied to
the model. The diagnostic is disabled in the portable profile.

All117 local FLA tests pass, with one Linux-only test skipped on macOS.
The new checks exercise partial/full segments, byte preservation, copy and
filesystem failures, invalid completion evidence and the actual submission
wrapper's observation reset. Diagnostic source
`7a33fc99e248faaeebeb9b5ec3064246c133c6e7` builds natively on baiying;
all46 compiler inputs match. Its979968-byte DLL SHA256 is
`ac234c0b416967b4c0978bd22a3d36a5bc3546287934c614ff6d6bb68c4789a1`.
The original q7169 output and final state match bitwise. With capture disabled,
q8192/out512 passes every original GB10 ID, callback and first logit
10.375/error0. Load is21345.493299 ms, TTFT23192.6479 ms and
TPOT101.249491 ms. This single run does not replace the qualified performance
median. See the [native diagnostic build and short regression](../benchmarks/correctness/fla-output-failure-capture-native-q8192-20260919.json).

The original128k owner/suffix diagnostic is running with capture enabled and
the unchanged7200-second process budget. Actual failed-input capture, delay
identification and long-context recovery remain pending. The published
portable candidate retains FLA `2b33665`; this diagnostic is not part of its
archive. Performance and release acceptance remain open.
