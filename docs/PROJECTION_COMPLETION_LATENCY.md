# Completed projection latency

The projection correction launcher previously returned
`hipErrorInvalidConfiguration` after successful stream synchronization when
a dispatch exceeded100 ms, a device/matrix window exceeded250 ms, or the
complete correction exceeded10000 ms. These internal latency thresholds could
reject already completed numerical work, as the separately observed FLA
failure did. No projection failure at those thresholds is claimed here.

The corrected launcher first checks successful HIP submission and stream
completion. Both enclosing monotonic host intervals must be finite and
nonnegative. A completed interval beyond its original nominal threshold emits
`hawkeye_completed_latency` and permits further work. Invalid clocks and HIP
failures still stop the call. The original strict timing classification
helpers remain for diagnostics; build metadata explicitly identifies the
completed-latency policy and nominal thresholds.

Candidate admission, scratch capacity, bounded dispatch quanta, candidate
ordering, arithmetic and cleanup are unchanged. The recursive local include
comparison against whole-provider source`ddacdc9` finds86 unchanged files;
only the provider wrapper and dispatch policy differ among88 local files.
This is a source include inventory, not a compiler dependency trace. External
process deadlines, real q8192 TTFT below10000 ms, the retained4187.415605 ms
target and every GB10 token/logit requirement remain unchanged.

Five relevant host tests pass. The actual production launcher is compiled
with16/8/4 replay lanes under ASan/UBSan. A deterministic test clock advances
at mocked synchronization, covering101/251/10001 ms ordinary and device
completions,16 slow resident-matrix windows, negative elapsed time, HIP
failures, deferred cleanup, allocation and candidate bounds. Existing output
and input checks remain. The runtime still uses `std::chrono::steady_clock`.

The [local verification](../benchmarks/correctness/hawkeye-completed-latency-policy-local-20260919.json)
binds commands, changed sources and the include comparison. It follows the
[independent failed-FLA-output comparison](../benchmarks/correctness/fla-failed-layer8-gb10-output-comparison-20260919.json)
without transferring that numerical evidence to a new projection build.
Source`50a5718cc588d38826d458abd2f5d09450c109f2` builds on baiying in95247.781 ms.
The13454848-byte DLL SHA256 is
`305b925419de180943beb707c3907843b5f29bd9b067148643237d9f4db17614`.
All494 compiled GPU kernels preserve the control's executable sections and
complete AMDHSA metadata. This comparison excludes host code and device
non-executable storage; actual numerical acceptance comes from the GB10 run.

The original q8192/out512 regression passes every output ID, actual callback
and first logit10.375/error0. Load is21273.614 ms, TTFT23241.4512 ms and
TPOT101.286092 ms. This single functional sample does not replace the retained
performance median. The run observes no projection latency outlier. See the
[native build, compiled comparison and q8192 boundary](../benchmarks/correctness/hawkeye-completed-latency-native-q8192-20260919.json).

The completed128k FLA repair run used the earlier whole-provider DLL and fails
the initial suffix token sequence; it does not qualify this new whole build.
Package, larger-context, performance and release acceptance remain open.
