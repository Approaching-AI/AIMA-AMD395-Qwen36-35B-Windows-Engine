# Real-model performance

Updated September 22, 2026. The current corrected Windows runtime passes the
short functional cases below, but its q8192 TTFT remains above the required
10-second boundary. No new release is qualified. All measurements use the real
BF16 model on baiying, Ryzen AI Max+ 395 (`gfx1151`), at batch size 1.
Model and engine loading are measured separately from TTFT.

## Current runtime

Whole provider and CLI source `35607853eb03487b443ddfed8529b8da5539de86`
resolve packed gate midpoint rounding and retain the cold final one-input repair. The ordinary route and four opt-in
native-MTP controls reproduce all 1,120 original GB10 outputs, their callback
sequences and first logits with zero first-logit error. These are individual
functional runs, not a new paired performance comparison.

| Case | Output IDs and callbacks | Load ms | TTFT ms | TPOT ms |
| --- | ---: | ---: | ---: | ---: |
| Ordinary q8192 | 512 / 512 | 21517.6512 | 23272.0441 | 101.032956 |
| Native-MTP q7169 | 32 / 32 | 21467.8422 | 25719.9802 | 256.743619 |
| Native-MTP q8191 | 32 / 32 | 21484.663701 | 29573.382701 | 284.850855 |
| Native-MTP q8192 | 512 / 512 | 21438.5739 | 24491.133999 | 252.530466 |
| Native-MTP q8193 | 32 / 32 | 21471.6946 | 25659.4965 | 221.823958 |

Each run binds its command, original prompt, model path, source, actual output
IDs, first logit, host checks and completed cleanup in the
[native evidence](../benchmarks/correctness/packed-gate-midpoint-products-20260922.json).
Native MTP remains opt-in. The
[cold-tail diagnosis](COLD_PREFILL_TAILS.md) records the earlier failures and
the scope of the repair.

The latest correctness-attached paired q8192 median remains 23353.80795 ms
for the earlier narrow-QK control. Its same-DLL OFF/ON/ON/OFF comparison
improves 24087.2361 to 23353.80795 ms, with all 2,048 original output IDs
passing. That result does not substitute for a new356 comparison.
[Original paired evidence and component bindings](NARROW_DOMAIN_QK.md).

## Immutable targets

| Requirement | Target | Current evidence |
| --- | ---: | --- |
| First q8192 performance boundary | TTFT below 10000 ms | Open; latest ordinary control is 23272.0441 ms |
| Retained q8192 TTFT | At most 4187.415605 ms | Open |
| Retained prefill throughput | At least 1506.407263 tok/s | Open |
| Retained TPOT | At most 35.502151 ms/token | Open |
| Retained decode throughput | At least 28.167926 tok/s | Open |
| Model plus engine load | At most 30000 ms | Passes the five individual controls above |

The complete context and prefix-cache curves remain required. Their targets
are not recalibrated to current measurements. Product acceptance requires
original prompt IDs, the GB10 first token and first logit within 0.125;
declared continuations must match token-for-token. Engine self-hashes are
diagnostics. Component correctness or timing alone does not qualify inference.
[Correctness and current context scope](EVALUATION.md).

## Measured work and active experiments

The [Linux core Windows prototype](LINUX_CORE_WINDOWS_PROTOTYPE.md) evaluates a
complete compute-core replacement after the component alternatives below proved
slower. Its pinned source, OS adapters and local controls are prepared. It has
no Windows build or product timing yet and does not replace any retained result.

The last completed q8192 phase profile has the following top-level scopes.
Its extra instrumentation makes its 27321.9776 ms TTFT diagnostic; it does
not replace the uninstrumented median.

| Profile scope | Total ms | Included work |
| --- | ---: | --- |
| Linear core | 7165.56 | Recurrence 3327.918, output projection 1970.7119, convolution 740.014 |
| Nine full-query attention calls | 5477.5099 | QK 2317.5885, probability/native PV 1446.7542, exact PV 1539.6902 |
| MoE | 4968.8112 | Routed and shared intervals overlap |
| Linear projections | 4362.125 | Input projection and its correction work |

Included values must not be added again to their enclosing scopes. Dense,
coarse-OUT and adaptive-OUT correction clocks overlap these totals. The
[full profile](../benchmarks/correctness/narrow-stack-completed-profile-20260918.json)
preserves the commands, outputs and invalid negative event clocks.

The structural fixtures at source `5d63d5c` consume the complete original
q8192 reference with zero repeated rows. The [projection experiment](PARTIAL_WAVE_PROJECTION.md)
passes all240 generated GPU configurations and all83,886,080 original BF16
operator outputs, but complete QKV preparation/replay increases41.2941 to641.18ms
and OUT increases152.258 to598.795ms. Its implementation remains outside the
runtime. OUT's comparator is the original midpoint correction, not retained
coarse-OUT. [Completed native evidence](../benchmarks/correctness/partial-wave-projection-native-20260921.json).

The independent [QK experiment](WAVE_MATRIX_QK.md) passes560 generated GPU
cases, five reference-extent fault controls and all33,554,432 original GB10
context cells per variant. Its complete component times are518.7543ms for the
retained narrow control,975.0799ms for the compact queue,1028.6810ms for the
wave owner,1485.3155ms for exact remainder and1614.9949ms for partial coefficients.
All include common preparation and classification. The narrow runtime route
remains selected. [Completed native comparison](../benchmarks/correctness/partial-wave-matrix-qk-native-20260921.json).
These isolated results require a broader performance redesign; they do not
change the whole-model median or the immutable targets.

Existing slower candidates and failed attempts remain available in the
[historical record](PERFORMANCE_HISTORY_THROUGH_20260921.md). Those results
inform route selection without prohibiting a materially different retry.

## Context and package boundaries

The original 131072-prefix plus 1024-suffix case passes both 512-token
continuations, restored owner32, first logits, callbacks, restoration and
changed-prefix rejection on whole source `32a1b96`. Its instrumented hit
TTFT is 62731.6067 ms and TPOT is 588.789589 ms, so the retained prefix targets
remain open. [Exact source and original-model boundary](../benchmarks/correctness/rope-single-round-prefix128k-product-20260919.json).

The preceding whole `8612387` 256k run completes all owner chunks but first
differs at suffix output index 124: 8984 instead of 4980. The completed c268
rerun reproduces all512 of those outputs. Its1257 same-history surface
comparisons first differ in layer5's incoming recurrent state, head13. The
90-case GPU replay on original reference operands passes both state layouts.
The earlier cause is now traced to a BF16 midpoint in B/head13 at263238.
The [repair](PACKED_GATE_MIDPOINT.md) passes actual original-operand GPU
components and the five short model controls. The complete256k rerun is
active and remains unqualified. The preceding c268 run's native wall is
22598962.712 ms with clean host checks. No256k or performance acceptance is
claimed. [Completed run and actual state evidence](../benchmarks/correctness/single-tail-q1-prefix256k-step124-native-20260921.json).
[Prior failure](../benchmarks/correctness/retired-reference-mode-prefix256k-20260921.json)
and [qualified same-history reference](../benchmarks/correctness/gb10-prefix256-step124-reference-20260921.json).

The c268 server builds on Windows and passes all 54 Rust tests. The
[new package preparation](../benchmarks/correctness/packed-gate-midpoint-package-prepared-20260922.json)
retains that binary's actual source after verifying all25 build inputs are
unchanged in356, and binds the repaired whole provider and CLI. Its archive,
actual HTTP matrix, three native retirement cases,13-case cold matrix and
one-hour soak remain open. The older unpublished
[R6 package](CURRENT_PACKAGE_R6.md) has its own source inventory and service
checks. Neither its observations nor Linux release results qualify the new
artifact. [Linux release review and Windows scope](SIBLING_REVIEW.md).

## q8192-neighbor continuity gate

The downloadable v1.0.1 archive was published on August 15 from source
`2bf04571dbd17122bd24fe7b8b7d207153458f7e`. The later August 16 continuity
results and 72-request sweep belong to source `09bd96fd`; they must not be
attributed to that archive. The original archive's neighbor limits also
differ from the later tightened limits. Those historical short-output checks
do not qualify the current corrected 512-token runtime.
[Preserved archive identity, numerical scope and individual measurements](PERFORMANCE_HISTORY_THROUGH_20260921.md#q8192-neighbor-continuity-gate).

The complete earlier performance document is preserved without body changes
in [the history through September 21](PERFORMANCE_HISTORY_THROUGH_20260921.md),
including published results, rejected experiments and their original evidence
links. Current qualification is stated above and in [evaluation](EVALUATION.md).
