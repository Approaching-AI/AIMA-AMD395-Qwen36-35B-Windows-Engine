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
Native build and real-model regression are pending. The ongoing128k FLA
repair run still uses the earlier whole-provider DLL. No package, performance
or release acceptance follows from these host tests.
