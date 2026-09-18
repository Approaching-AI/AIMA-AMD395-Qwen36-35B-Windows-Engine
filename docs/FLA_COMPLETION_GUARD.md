# Completed FLA stages and latency

The retained non-pipelined runtime checks successful HIP submission, end-event synchronization and
elapsed-time API status before using a completed interval. A finite,
nonnegative GPU or enclosing monotonic host interval permits continuation.
An interval above the nominal 100 ms is now reported as
`FLA_COMPLETED_LATENCY`, including both clocks and the selected source. It
does not turn already completed work into a runtime error. The normal clock
selection and invalid-GPU-clock host fallback are preserved.

HIP API failures and a pair of invalid clocks still fail. Partial submission
is drained before scratch reuse, and deferred stages require their outer
completion. Kernel arithmetic, dispatch, segment sizes and memory ownership
are unchanged. External bounded process execution remains required. The
q8192 product latency target, retained performance target, model-load limit
and every GB10 numerical boundary remain unchanged.

The strict 100 ms `evaluate` helper remains available for the standalone
failure diagnostic. Runtime dispatch uses `evaluate_completed`; its build
metadata explicitly labels 100 ms as a nominal latency and records that
completed latency is not a runtime failure. The historical failure-only
capture hook does not run for an accepted completed latency outlier.
The disabled experimental pipeline retains its separate phase guard; this
repair changes the ordinary stage and deferred-segment completion wrapper.

All 121 local FLA tests pass, with one existing Linux-only test skipped on
macOS. The actual submission and segment wrappers are exercised with slow
completed work, invalid clocks, every HIP failure path, deferred submission,
stream mismatches and cleanup. The 121 clock pairs separately retain strict
diagnostic classification and test the completed-runtime policy. These are
host tests. See the [local policy checks](../benchmarks/correctness/fla-completed-latency-policy-local-20260919.json).

Repair source `1d11bf742268924ecf5df1e0e66a0e19fb7cec15` now builds on
baiying in56377.047 ms, with all46 compiled inputs checked. The984064-byte
DLL SHA256 is
`a6b110c3482ca8bc1bb7ef1b6c8213133bb98d53afa5d3a491ba4194bc1d3a88`.
Captured q7169 output and final state match bitwise. Full q8192/out512 matches
all512 original GB10 IDs and actual callbacks, with first logit10.375/error0.
Load is21278.6683 ms, TTFT23363.176899 ms and TPOT101.670414 ms. The run has
no latency outlier or host-clock fallback. This is a functional regression,
not a paired speed comparison or a new retained performance median. See the
[native build and original q8192 boundary](../benchmarks/correctness/fla-completed-latency-native-q8192-20260919.json).
The repaired original128k owner/suffix run is active under the same7200-second
process deadline, with capture disabled and all original comparisons retained.

The policy repair follows three preserved failed 128k owner/suffix runs.
Each completed 14 owner chunks through 114688 tokens and produced no output:

| Provider | Failure layer | Completed GPU / host interval | Native process wall |
| --- | --- | --- | --- |
| Original `7b20c90` | 33 | Failed interval not recorded | 4219017.852 ms |
| Dual-clock `2b33665` | 33 | 283.885010 / 284.029600 ms | 4213592.288 ms |
| Capture `7a33fc9` | 8 | 297.485138 / 297.613700 ms | 3832371.554 ms |

The [original failure](../benchmarks/correctness/long-final-pv-prefix128k-guard-failure-20260918.json),
[dual-clock repeat](../benchmarks/correctness/fla-completion-guard-prefix128k-delay-failure-20260919.json)
and [captured failure](../benchmarks/correctness/fla-output-failure-prefix128k-capture-20260919.json)
retain the commands, binary identities, host checks and raw artifact hashes.
All completed before the unchanged 7200-second process deadline, with normal
cleanup and no numerical acceptance. The changed layer weakens the earlier
fixed-input-location hypothesis.

The capture preserves 54657024 bytes across seven surfaces: Q/K, V-new,
chunk states, cumulative gates, scores and completed output. All values are
finite. Static eligibility rejects no prior-state dot cells and 380928 of
4194304 local-value dot cells; this alone does not explain the delay. Copies
use at most 1 MiB of host scratch, publish a final record only on completion,
and never retry a kernel or provide captured inputs to model inference.

Standalone replay source `5ed1f233ea68fce8c25dd56cf9987d0fa215eec2` builds
on baiying. The 823808-byte executable SHA256 is
`2ffedaf75d1863041481fa1acf1c290ee91317003a4cc6d85f1de216d35e7aa4`.
The same captured layer8 segment reproduces every score and output bit in
3.990500 ms. Separate launches measure scores at 0.606600 ms and output at
2.879790 ms, again with no bit mismatch. Immutable inputs, redzones and all
completion/process checks pass. This small-memory component process does
not reproduce the original delay; it does not identify scheduling, paging
or another root cause, and native self-comparison is not a GB10 oracle.

A separate original GB10 layer33 window, from position114688 for8192 tokens,
passes the seeded key-major native interface with provider `7a33fc9` and
independently pinned tool `37c8504`. All33554432 BF16 output cells and524288
FP32 state cells match the reference bitwise. Repeating the original seed
is bitwise stable; a zero-state negative control differs in13660175 output
cells. Completed operator clocks are91.288704,88.080154 and89.094551 ms.
This covers the earlier layer33 failure location, not the captured layer8
segment or the full128k model. See the [native diagnosis and independent component evidence](../benchmarks/correctness/fla-completed-delay-native-diagnosis-20260919.json)
and [original GB10 window](../benchmarks/correctness/gb10-prefix128-layer33-window-20260919.json).

Historical short-context qualification remains attached to its exact source.
Provider `2b33665` passes captured q7169, full q8192/out512 and the original
16k prefix-owner/suffix transactions. The16k run demonstrates a negative
GPU interval with a valid0.3358 ms enclosing host interval. See the
[dual-clock product evidence](../benchmarks/correctness/fla-completion-guard-native-product-20260918.json).
Capture provider `7a33fc9` also passes captured q7169 and every q8192/out512
ID, callback and first logit10.375/error0, with capture disabled. Its load is
21345.493299 ms, TTFT23192.6479 ms and TPOT101.249491 ms. That single sample
is not a replacement performance median; see the [diagnostic provider regression](../benchmarks/correctness/fla-output-failure-capture-native-q8192-20260919.json).

The unpublished R6 portable candidate retains FLA `2b33665`. The current
policy repair has not been put in that archive. Full128k recovery, product
performance and release acceptance remain open.
