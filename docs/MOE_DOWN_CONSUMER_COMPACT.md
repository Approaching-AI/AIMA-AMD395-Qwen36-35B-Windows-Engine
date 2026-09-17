# Fused MoE down consumer collection

The default-off `QRT_QWEN36_MOE_DOWN_CONSUMER_COMPACT=1` experiment merges
the existing [complete consumer certificate](MOE_DOWN_CONSUMER_FILTER.md)
and candidate collection. Runtime source `2f29c36` is unchanged through
native qualification source `9235750`; subsequent commits repair the native
test fixture's event and explicit-stream ownership.

For each token/channel, one thread encloses all eight weighted routed
contributions and the completed shared merge. A fixed final combined BF16
endpoint permits omission of the original selected contributions, preserving
the full unrounded residual. Otherwise every selected candidate keeps the
original exact K16 replay. The empirical projection envelope is unchanged.
No external reference or control output supplies computation.

Warp publication and a bounded CTA queue emit the retained indices directly
into the existing window queue. This removes the old full-shape omission
bitmap and repeated selector scan. Expert permutation, exact replay and
production combine retain their original arithmetic. The per-invocation
owner uses 1056 bytes for counters and guards, waits on its own recorded
shared event, and checks completion identities before freeing its storage.
The experiment requires the qualified q8192 fused merge, staged replay and
expert ordering. Audit/filter experiments cannot be enabled with it. Other
logical lengths retain existing dispatch.

## Native qualification

Full local checks pass 491 Python tests with two environment skips, Rust,
C/ABI, clippy and hygiene checks. The actual owner passes 39 injected API
failures and eight corrupted reports under address/undefined-behavior
sanitizers, including event ordering and drain-before-free.

On baiying/gfx1151, 24 generated cases at 1/3/17/33/129/257 tokens compare
the original route, previous filter, standalone fused collector and actual
runtime owner/launcher. Each standalone fused window is checked against the
old filter's complete queue. All raw route values, complete F32 residuals,
counter identities, input immutability and guards pass. Twenty-one invalid
calls reject without writes. Each fixture computes 154 periodic CPU dot
references and reuses them across output positions; the report's
`independent_cpu_dots` field counts positions checked against these cached
references, not independent CPU computations.

The full 8192 generated case spans 256 experts with original staged operands,
row classification, expert ordering and 16384-block windows. Values and
selection distributions are synthetic. All 134217728 original down cells
are selected; all filtering variants omit 67549152 and replay 66668576.
Every output is checked after one warmup and each of three rotated samples.

| R6 variant | Median completed ms | Samples ms |
| --- | ---: | --- |
| Original unfiltered | 632.9772 | 632.9772, 533.1953, 634.4430 |
| Previous filter | 270.1754 | 266.5575, 270.1754, 277.4230 |
| Standalone fused collector | 257.8840 | 269.7204, 257.8840, 257.5535 |
| Actual runtime owner/launcher | 254.7320 | 255.7966, 254.1534, 254.7320 |

The clock includes certificate, collection, permutation, replay and combine;
the actual runtime variant also includes prepare/finish validation. Common
allocation, upload/reset, staged preparation, shared-event completion and
readback are outside. The 15.4434 ms saving against the old filter is a
component observation and cannot be multiplied across model layers as a
measured TTFT reduction.

Four failed revisions remain in the evidence. R2/R3/R4 time out while
querying the stream after recording an event, before comparison kernels.
Direct event polling resolves that boundary in R5. Its first dense
cross-window case then exposes a fixture queue reset issued on the default
stream while replay uses a nonblocking stream. R6 puts all fixture transfers
on its owned stream; all cases pass. Production numerical code is unchanged
through these diagnostic revisions.

[Source-bound native evidence](../benchmarks/correctness/moe-down-consumer-compact-native-20260917.json):
588918 bytes, SHA256
`73c1f340e9c20a1121629c315410317ff9342f9492cd66e3c33b2a2910ea3bab`.
Real-model GB10 and net product timing remain separate gates. This component
evidence grants no inference, retained-performance, package or release
acceptance.

## Real-model comparison

Four fresh baiying processes use `D:\models\Qwen3.6-35B-A3B`, the same
new MoE DLL from `9235750`, whole `6e4908b`, CK `cf0f889`, FLA `2ee6215`
and CLI `24c4304`. Adaptive linear OUT mode 2, combined exact attention,
register PV, dense PPB1000 and MoE PPB512 remain fixed. Only the compact
consumer flag changes. The MoE DLL is 914432 bytes, SHA256
`716379525ead68d583b36380261b467e8bfc37cd427f6577a24ee4519cda9891`.

| Order / compact | Load ms | TTFT ms | TPOT ms |
| --- | ---: | ---: | ---: |
| 1 / Off | 21361.2329 | 25635.9938 | 100.631598 |
| 2 / On | 21329.9096 | 25332.2134 | 103.321543 |
| 3 / On | 21282.4474 | 25426.6205 | 101.305620 |
| 4 / Off | 21391.7412 | 25585.5306 | 101.253438 |

All original prompt IDs, all 2048 GB10 output IDs and actual callbacks
match. Every first logit is 10.375, within the unchanged 0.125 tolerance.
Each ON process activates all 40 owners, omits 126904922 of 158740988
original selected replays (79.9446%) and executes 31836066. The counts match
the old consumer certificate. All130 dense, ten coarse full-attention OUT
and 30 adaptive linear OUT counts match across arms; these are diagnostics.

OFF/ON TTFT medians are 25610.7622/25379.41695 ms, a 231.34525 ms reduction
(0.9033%). Both ON samples are below both OFF samples. Retain mode 1 only
as the default-off experimental q8192 prefill baseline. TPOT medians are
100.942518/102.3135815 ms; no decode or total-request speedup is claimed.
The compact owner does not activate at decode length 1. Two observations per
arm are limited evidence, and TTFT remains above 10 seconds. No prefix,
long-context, package, retained-performance or release acceptance follows.

[Source-bound real-model evidence](../benchmarks/correctness/moe-down-consumer-compact-product-20260917.json):
884564 bytes, SHA256
`8ae8646fb5c93866c7e7c3588e26290c0fd5223c07bee104d01d208a0f1649cf`.
