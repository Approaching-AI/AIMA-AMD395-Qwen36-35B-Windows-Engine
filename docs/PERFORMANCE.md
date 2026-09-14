# Real-model performance

## Current unreleased measurements, 2026-09-14

Shared-expert prevalidated exact replay at `1958c2c` passes both complete
q8192/out512 runs on baiying with `D:\models\Qwen3.6-35B-A3B`. The same MoE
DLL OFF/ON preserves all 512 GB10 outputs and actual callbacks, first token
144 and raw logit 10.375. Whole `a50f15d`, CK `4a5a5b0`, FLA `2ee6215`
state 8, CLI `a797b62` and all ten AOT files are common.

Actual callback TTFT is 37867.3011 / 37527.5887 ms OFF/ON, a 339.7124 ms
reduction in this single pair. Enabled TPOT is 101.594291 ms and model plus
engine load is 21368.4516 ms. Use the enabled route for the next experiment;
this does not establish repeatability or retained performance. The immutable
4187.415605 ms TTFT and 35.502151 ms TPOT targets remain open.

Eight native shared comparisons preserve 22658560 BF16 cells per variant,
original norm bits and eligibility flags, including integer fallback. Six
routed regressions preserve raw output bits and norms. Dedicated shared
metadata uses 77824 bytes; the q8192-only option remains off by default.
Selectors, arithmetic, K16 order, PPB bounds and short-shape routes are
unchanged. Full local checks pass. No package, prefix, HTTP or release gate
is inferred from these runs. A synchronized enabled profile is now pending
to choose the next structural replacement.
Command: `run-native-shared-prevalidated-product-r1.ps1` with the recorded
OFF/ON arguments. Evidence: `benchmarks/correctness/shared-prevalidated-product-20260914.json`,
SHA256 `42ba61141f4852a9cb01d4cc0c7e825b29c1348c20dda6b32ce6c84e26358d46`.

The complete QKV interval-filter experiment at `be477fa` preserves all
58728448 GB10 BF16 cells and the original 3791742 selected identities.
It certifies only 56913 candidates (1.50 percent), leaving 3734829 for exact
replay. All replayed FP32 values match the control by identity, all 37 native
safety cases pass, and source/output/partition guards remain intact.

Total preparation plus filtering, compaction, count read and replay costs
458.8400 ms versus 51.3805 ms for unchanged exact replay. This 407.4595 ms
regression rules out product promotion on current evidence. The earlier
92.2 percent unweighted sample did not predict reduction in actual candidates.
Command: `run-native-projection-interval-filter-r1.ps1 -Kind filter|real -Action build|test`.
Evidence: `benchmarks/correctness/projection-interval-filter-20260914.json`, SHA256
`3baa42e6bad722e55c6dda32950babeb596728b5f5f20d4b8dc2f2e14ff8ecf2`.

The existing same-run q8192 GB10-qualified profile has 40 complete nonnegative
MoE detail reports. Routed gate / up / down correction totals are
1714.231794 / 2052.170724 / 1571.298102 ms. Shared-stream intervals total
4898.652916 ms and overlap routed work; do not add these scopes together.
Source inspection finds shared projections still on the original sixteen-lane
integer replay, while routed projections use validated four-lane float replay.
An opt-in implementation now extends that exact path to q8192 shared
projections with 77824 bytes of dedicated stream metadata and unchanged norms,
selectors, K16 order and short-shape dispatch. Full local checks pass 378 Python
tests (two skips), 47 Rust tests, C smoke, clippy and hygiene. Allocation-failure,
drain-before-free and shared/routed pointer ownership tests also pass.
Native safety and the same-artifact product pair are now complete as recorded
above; the option remains off by default.

Prior interval/carry probes remain in `projection-interval-20260914.json`,
`wmma-scalar-carry-envelope-20260914.json` and
`wmma-projection-envelope-20260914.json`. All narrower-coefficient
counterexamples remain attached; no product or release gate is relaxed.

The hybrid exact-integer/scalar-float QK experiment at `824a07f` passes
128 native generated cases, 257735040 score comparisons and 8192 independent
CPU dots. Eight original q7169 variants each preserve 418496528 raw scores
and 228 CPU dots; all memory guards and immutable operands pass. Full local
checks pass 377 Python tests with two skips, 47 Rust tests, C smoke, clippy
and hygiene. The independent sanitized arithmetic test covers 500000 groups
and all three exact-integer, scalar-float and original-integer branches.

The same-run scalar-float control takes 325.1009 ms plus 6.5245 ms transpose.
The best hybrid takes 538.6084 ms plus 0.4851 ms key encoding, including
query encoding. It improves its matching nonhybrid matrix control at
582.9409 ms but remains slower than scalar float. Keep it outside product
dispatch. CPU samples predict 2176 exact-integer and 1472 scalar-float K16
groups; these are diagnostic CPU classifications, not GPU counters.

A separate compiler resource check reports no scratch or register spills.
Scalar QK uses 65 VGPRs; hybrid layouts use 83–94. Their compiler occupancy
is 16 waves/SIMD, which is not a measured occupancy result. The evidence
does not support register spilling as the cause of this performance gap.
Investigate the seconds-scale dense projection replay surface using the
completed whole-stack attribution below. Product arithmetic, admission
bounds and all mission targets remain unchanged.
Command: `run-native-matrix-float-qk-r1.ps1 -Kind qk -Action build|test|capture`.
Evidence: `benchmarks/correctness/matrix-float-qk-20260914.json`, SHA256
`bf667a8b18fd696450a8469f93569d4ce819d5937bea4b8876d4f4cf9132c659`.

Earlier exact matrix-consumer layouts (`395c9ee`) and decoded QK windows
(`7d25cfd`) remain component evidence only. Their full comparisons are in
`matrix-consumer-qk-20260914.json` and `decoded-window-qk-20260914.json`.
No token-loop or product performance qualification follows from these runs.

Dense and MoE candidate grouping at source `a50f15d` passes both complete
original q8192/out512 product runs on baiying with
`D:\models\Qwen3.6-35B-A3B`. Both OFF and ON preserve all 512 GB10 outputs
and actual callbacks, first 144 and exact raw logit 10.375. The same dense
and MoE DLLs, four lanes, one MoE staged K16 group, ten AOT artifacts, CK
`4a5a5b0`, FLA `2ee6215` state 8 and CLI `a797b62` are common.

Callback TTFT is 37810.5655 / 38176.3825 ms OFF/ON. Grouping is 365.8170 ms
slower in this single pair, despite its component gains. Keep both grouping
options at 0 for subsequent work. The qualified OFF control has TPOT
101.637313 ms and model plus engine load 21296.3550 ms. This does not
establish repeatability or retained performance; the immutable 4187.415605 ms
TTFT and 35.502151 ms TPOT targets remain open.

All 48 native permutation/capacity cases, six MoE controls, four actual
dense correction shapes and both original real QKV runs pass. Full local
checks and extended address/undefined-behavior dispatch checks pass. Both
product runs select identical dense candidate counts; completed dense
correction totals are 7310.218 / 7346.269 ms. Those inclusive correction
clocks do not fully attribute TTFT. The completed synchronized OFF profile
below now directs the next structural investigation. Original arithmetic,
FP64 norms and admission bounds remain unchanged; range normalization
remains a component experiment.

The same-artifact synchronized OFF profile preserves all 512 GB10 outputs
and callbacks, first 144 and raw logit 10.375. Completed host full-attention
core time is 14284.6 ms; its nested CK event total is 9812.025 ms. The thirty
completed linear projection/core totals are 6571.388 / 7800.228 ms, with
output projection included in linear core. Forty MoE event totals sum to
7533.152 ms. These intervals overlap by scope and must not be added together.
Nine residual/postnorm event intervals are negative and excluded.

CK `9bbd5d0` now passes native HIP build and the complete q8192 GB10
boundary with `QRT_CK_SM121_PROFILE_COMPLETED_STAGES=1`. All ten attention
calls, 640 query batches and 3200 completed checkpoints pass phase coverage
and disjoint interval accounting. QK is 4666.2065 ms, probability generation
1449.7858 ms, approximate PV 1767.9649 ms, collection 110.8707 ms and exact
PV 1887.8618 ms. Entry wait, preparation and dispatch remainder account for
the rest of the 9949.2487 ms completed attention total. No interval is negative.

The diagnostic defaults to 0 and leaves Q1 synchronization unchanged.
Instrumented callback TTFT is 39229.4652 ms, TPOT 104.926239 ms and load
21328.6543 ms. These do not replace the uninstrumented control. Prioritize
cooperative QK operand decoding in bounded K windows, with the original
K16 arithmetic and fallback. Complete raw QK and external BF16 comparisons
must precede a product decision. Command:
`run-native-ck-completed-profile-q8192-r1.ps1 -CkCompletedProfile 1 -ProfileStages 1`.
Evidence: `benchmarks/correctness/completed-attention-phases-20260914.json`,
SHA256 `8a12277736733c4f94d8d8bdcc1abdd5428d7e1872773dac4caa26aa548bb460`.

The recorded instrumented callback TTFT of 39026.1248 ms is not comparable
to uninstrumented performance. Command:
`run-native-partition-product-q8192-r1.ps1 -PartitionReplay 0 -ProfileStages 1`.
Evidence: `benchmarks/correctness/current-off-stack-profile-20260914.json`,
SHA256 `da00667bec7c1b0276ab83da135d40520514bab6a26d54d1dd3a99bc222d2bdf`.

The qualified 64k stack retains its earlier sixteen-lane MoE identity.
No new prefix, Windows 128k/256k, package, HTTP, soak or release acceptance
follows. Command: `run-native-partition-product-q8192-r1.ps1 -PartitionReplay 0|1`.
Evidence: `benchmarks/correctness/dense-moe-candidate-partition-product-20260914.json`,
SHA256 `3999da7941991baa25bc488981d1589b7e5887346f1d3ae250bdb284f68f9378`.

The preceding MoE work-ownership experiment at source `cb0f266` passes a complete same-source
q8192/out512 pair on baiying with `D:\models\Qwen3.6-35B-A3B`. Both sixteen
and four lanes preserve all 512 GB10 outputs and actual callbacks, first
144 and exact raw logit 10.375. Whole `27cfc32`, CK `4a5a5b0`, FLA `2ee6215`
state 8 and CLI `a797b62` are common. Both native safety suites pass, and
all ten AOT artifacts are identical. This revisits the older integer-era
experiment under current prevalidated-float replay and registered metadata.

Callback TTFT is 39379.8088 / 38091.7431 ms for sixteen lanes with four
staged K16 groups / four lanes with one group, a 1288.0657 ms reduction in
this single pair. Four-lane TPOT is 101.798454 ms and load is 21296.2683 ms.
Use four lanes and one group for subsequent q8192 candidates; source build
defaults remain unchanged. Numerical bounds remain dense/MoE PPB 1000,
full-attention OUT PPB 10000 and routed radius 512. This does not establish
repeatability or retained performance. The qualified 64k stack below still
uses the prior sixteen-lane MoE. Broader contexts, current package/HTTP,
soak and release remain open. Command:
`run-native-moe-prevalidated-ownership-product-r1.ps1 -MoeLanes 16|4`.
Evidence: `benchmarks/correctness/moe-prevalidated-ownership-product-20260914.json`,
SHA256 `c478df7f4139e5fe855c12bed7b61bd77627aa65818ca3d1aef93cb7cbc557f1`.

Exact candidate partition at public source `156badd` passes all 24 native
permutation cases and all original QKV checks: 58728448 GB10 BF16 cells,
3791742 unrounded FP32 values, candidate identities, class order and memory
guards. Counts are 3028041 floating and 763701 integer. Full local checks pass.
With unchanged four-lane arithmetic, completed preparation plus replay is
50.8645 / 43.2025 ms for original / grouped ordering. The range-arithmetic
variants take 48.3263 / 39.7011 ms. GPU partition and raw capture writes are
included; these are single component observations.

The component result motivated the dense/MoE product comparison above;
its complete GB10-valid product regression now determines the option choice.
Range normalization stays outside product dispatch during that comparison.
No new product performance or release acceptance follows. Command:
`run-native-replay-partition-r1.ps1 -Kind partition|real -Action build|test`.
Evidence: `benchmarks/correctness/replay-candidate-partition-20260914.json`,
SHA256 `d550d9627c3426916e4fa44a8c97f70742f97ebbf811686c3b793248495be65e`.

Range-certified projection at `8b0462e` preserves all 28 native dot controls,
all 58728448 original GB10 QKV BF16 cells and 3791742 unrounded candidate
values. The finite-range proof also passes 28835840 independent host
normalization comparisons; full local checks pass. Four-lane preparation
plus replay is 52.7459 / 47.9062 ms for original / certified normalization,
with 3028041 certified candidates. This single component gain does not
establish a seconds-scale product benefit; retain the original dispatcher.

Read-only inspection finds that the 112 old-ineligible weight rows contain
229376 tiny normal BF16 values, with biased exponents 4 through 51. All
8192 weight rows differ bytewise; the first-layer input has 257 unique rows.
Do not generalize that embedding-only repetition to later layers. This motivated the candidate grouping experiment recorded above. Source bounds, original dot order and PPB 1000 remain unchanged.
Command: `run-native-range-projection-r1.ps1 -Kind projection|real -Action build|test`.
No new token-loop or release acceptance follows. Evidence:
`benchmarks/correctness/range-certified-projection-20260914.json`, SHA256
`f5db34aee02363f5bd6432743e49a2cfd5c6b2be7de29092ecd6f076c5e2391b`.

The component-only FP32 K16 carry experiment at `cb0f266` preserves all
tested raw bits, but does not justify product promotion. Native checks cover
2097152 ordered K16 groups, 24 generated projection cases, 42 generated QK
cases and 1255489584 original q7169 QK score comparisons. All six QKV
variants preserve 58728448 external GB10 BF16 cells and 3791742 unrounded
candidate FP32 values, with unchanged PPB 1000 and passing memory guards.

Completed QK time is 341.5882 / 315.7412 / 314.1448 ms for the qualified
scalar-float carry / direct FP32 bits / explicit round-toward-zero variants.
QKV preparation plus replay is 52.8446 / 52.8659 / 52.5744 ms with four
lanes and one staged group; the corresponding sixteen-lane, four-group
variants take 106.928 / 109.552 / 108.020 ms. These single component clocks
include raw QKV capture writes and exclude allocation, uploads and CPU
verification. Keep the new carry outside product dispatch and investigate
current MoE work ownership separately. All local checks and native builds
pass; no new token-loop or release acceptance follows. Command file:
`run-native-f32-carry-r1.ps1 -Kind core|projection|qk|real -Action build|test|capture`.
Evidence: `benchmarks/correctness/f32-carry-components-20260914.json`, SHA256
`c8d57159d6bdb18e55b4d63da6bb03b492b4cfbdeb4a3ada48b1d6d68b14e2cd`.

The current stack passes the original 65536-prefix plus 1024-suffix GB10
boundary on baiying with `D:\models\Qwen3.6-35B-A3B`: all 512 outputs and
actual callbacks match, first token 3709 and exact raw logit 5.9375. All eight
owner chunks complete with first token 16/logit 24.25. State restoration and
changed-prefix rejection pass. Whole `27cfc32`, CK `4a5a5b0` with long direct
PV operands enabled, MoE `9f00db5`, FLA `2ee6215` state 8 and CLI `a797b62`
are the qualified components for this case. Command:
`run-native-long-direct-pv-product-r1.ps1 -LongDirectPvOperands 1`.

Warm callback TTFT is 52683.7323 ms, 6604.4922 ms below the 59288.2245 ms
qualified CK `d705a68` baseline in this single pair. TPOT is 355.022114 ms;
load is 21209.1856 ms, within 30000 ms. The CLI seed phase, including the
initial continuation, is 2576915.3558 ms; native wall is 2832783.961 ms and
host guards pass. Only CK identity and the long direct operand option differ
from the baseline. Long calls retain the original per-group envelope and
add no allocation. Use the enabled option for subsequent long-context
candidates; its source default remains 0.

The unchanged 64k targets are 5432.415542 ms TTFT and 46.658882 ms TPOT.
Performance, saved checkpoints, Windows 128k/256k, current package/HTTP,
one-hour soak and release remain open. Evidence:
`benchmarks/correctness/long-direct-pv-prefix64k-20260914.json`, SHA256
`6c3595aa3ea69bfc37cf553021ba4f0f10496279bbc30fe00207aaf45fbf3cfe`.
The complete preceding 64k baseline remains in
`benchmarks/correctness/prefix64k-current-stack-20260914.json`, SHA256
`a113d05400d2469ecc181c32d4244273af4eddcbfe8c94a886fe91a779cfd3ad`.

The opt-in long-history direct PV provider at `4a5a5b0` passes 42 native
kernel cases and the original 16k+1024 layer-3 component. All 4194304 BF16
cells and same-DLL raw FP32 outputs agree between OFF/ON, including repeat,
same-address refresh and negative checks. Four-call process walls are
2966.147 / 2253.658 ms; these are component diagnostics. Long calls retain
the original per-group error envelope, and the new option defaults to 0.

Its original q8192 control matches all 512 outputs/callbacks and exact first
144/logit 10.375. Callback TTFT is 39203.4717 ms, TPOT 101.690611 ms and load
21282.1297 ms. The long option is inert at q8192; no speed change is inferred
from this single control. The complete real 64k result is recorded above.
Evidence: `benchmarks/correctness/long-direct-pv-controls-20260914.json`,
SHA256 `0d2deb5fba42aa21c2fba9225fd6309b4596324fac402605d9be21c345d147fb`.

CK `d705a68` corrects a long-history application timeout without changing
attention arithmetic. The first 64k attempt completed 24576 original inputs
but stopped in layer 15 of the next chunk: 8128 of 8192 queries had drained
successfully at 20.081793 seconds, exceeding the old flat 20-second budget.
No output or callback was emitted; that run has no inference acceptance.
Each query window now receives a bounded budget based on its query count
and ending key-history extent. q8192 keeps its original 20-second execution
bound, and the long process retains a separate 3600-second hard timeout.
Performance targets are unchanged.

The rebuilt CK passes the original q8192/512 control with whole `27cfc32`,
MoE `9f00db5`, FLA `2ee6215` state 8 and CLI `a797b62`. All 512 outputs and
callbacks match GB10, first 144 and exact logit 10.375. Callback TTFT is
39324.1539 ms, TPOT 102.038706 ms and load 21327.8670 ms. Mathematical kernel
and external CK source hashes are unchanged. All local checks pass, including
injected-clock progress, failure-drain and overflow cases. The real 64k
rerun is recorded above; current package, HTTP, soak and retained-performance gates
remain open. The refreshed Linux release is still `v1.5.1-native-vl.7`, with
unchanged notes. Command: `run-native-attention-history-deadline-control-r1.ps1`.
Evidence: `benchmarks/correctness/attention-history-deadline-20260914.json`,
SHA256 `6ad4a9f91986bb0d3d3cd2fa3ab362dbf3748b2fb66593418451c8dda7818ab3`.

Whole and CK source `27cfc32` extend exact attention capacity from 65536 to
131072, covering a 65536-token prefix, 1024 suffix inputs and the resident
decode tail. The three fixed attention allocations grow by 306184192 bytes
(292 MiB). Model position limits and numerical dispatch parameters stay
unchanged. Local checks pass 371 Python tests with 2 skipped, 47 Rust tests,
C smoke, clippy and hygiene; both native Windows builds pass.

The new q8192 control on baiying with `D:\models\Qwen3.6-35B-A3B` preserves
all 512 original GB10 outputs and callbacks, first 144 and exact logit 10.375.
Callback TTFT is 39498.9559 ms, TPOT 101.624535 ms and model plus engine load
21503.0635 ms. MoE `9f00db5`, FLA `2ee6215` state 8, CLI `a797b62` and all
qualified numerical flags remain common with the 39620.1231 ms reference.
This single capacity control does not establish a speed improvement. The
separate real 65536+1024 prefix result is recorded above; 128k/256k plus suffix need
larger capacity and their own qualification. Performance and release gates
remain open. Command: `run-native-prefix64k-capacity-control-r1.ps1`.
Evidence: `benchmarks/correctness/prefix64k-capacity-q8192-control-20260914.json`,
SHA256 `0981bd909d6495e5a9e305767d7adc8912d289376b4f211537a31b186a1eb160`.

Scalar float GDN state at FLA source `2ee6215` passes same-DLL q8192
OFF/8-column controls on baiying with `D:\models\Qwen3.6-35B-A3B`. Both
match all 512 original GB10 output tokens and actual callbacks, first 144
and exact raw logit 10.375. Callback TTFT is 40531.9227 / 39620.1231 ms,
a 911.7996 ms reduction in this single pair. Enabled TPOT is 100.946664 ms
and model plus engine load is 21304.6899 ms. Use
`QRT_FLA_GDN_SCALAR_FLOAT_STATE=8` in subsequent candidates; source default
remains 0 pending broader qualification. Scalar WU/output 1, tiled KKT 2,
whole/MoE `9f00db5` registered metadata 1 and both prevalidated replay
options 1, CK `6c0c54c` scalar QK 1 and CLI `a797b62` remain common.
Command file: `run-native-fla-scalar-state-product-r1.ps1 -StateColumns 0|8`.

Each state CTA retains complete columns, reuses losslessly packed BF16
operands and row eligibility, and computes the original decay once per
token/chunk. Ordered K16 carry, integer fallback, BF16 residual/checkpoint
boundaries, final FMA and FP32 state layout are unchanged. Both 4-column
and 8-column versions preserve every output and final-state bit in q64,
q65 and the original GB10 q7169 GDN capture, including sync/async parity.
Full `make check` passes 364 Python tests with 2 skipped, 47 Rust tests,
C smoke, clippy and public hygiene. WU/output lane markers now report the
actual scalar lane count 1. State dispatch has its own explicit marker;
checkpoint-capture calls retain their original kernel.

Completed state event diagnostics fall from 1952.21936 to 1009.60123 ms
across 240 segments. These measurements are inside the linear-core wall;
actual callback TTFT remains the product clock. Continue structural work
on the attention and projection walls. q8192 still exceeds 10 seconds and
the immutable 4187.415605 ms target. New long-context, checkpoint, package,
HTTP, soak and release acceptance remain open.
Evidence: `benchmarks/correctness/fla-scalar-state-product-20260913.json`,
SHA256 `758caabf0fbba86e772691902bd052ad2c98f06d1267d36d8fe062593a28d9c1`.

The prevalidated float PV component at source `08326fd` passes all 348
native comparisons (51904512 output positions), including original raw
output/accumulator/denominator equality, CPU metadata checks, candidate
ownership and memory guards. Original, 1-thread and 4-thread routes each
match all 29364224 GB10 q7169 attention cells and all four qualified replay
files, retaining 2198673 candidate cells. Completed query plus V preparation
is 645.9263 / 689.9965 / 638.8247 ms in these single observations. The
1-thread route is slower; the 4-thread difference is only 7.1016 ms and
does not establish a seconds-scale product gain. Keep the product PV
dispatcher unchanged and investigate broader dense projection correction,
which takes 7359.256 ms in the current qualified state-8 product run.

This experiment reuses the consumed score slab for row eligibility,
preserving separate probabilities, scales and candidate storage. It retains
K16 rounding, K32 rescaling and the original fallback. Full local checks
pass; it has no product environment dispatch and no new token-loop,
long-context, package, HTTP, soak or release acceptance. Command file:
`run-native-float-pv-replay-r1.ps1 -FloatPvLanes 0|1|4`.
Evidence: `benchmarks/correctness/prevalidated-float-pv-20260913.json`,
SHA256 `3334d2061034d39818dabf8635ae724c0a999180da95868c7604c0cbd24a27d0`.


The tiled dense projection component at source `6eb2eba` passes all 56
native CPU/GPU comparisons and all 58728448 original GB10 q7169 QKV cells.
Both 64-row and 128-row layouts preserve the 3791742 candidate identities,
raw output bits, CPU eligibility, bitmap ownership and memory guards.
Completed eligibility plus bitmap/replay is 49.4104 / 93.5343 / 142.6540 ms
for original prevalidated4 / 64-row / 128-row, respectively. Each candidate
adds a 7341056-byte bitmap; both are substantially slower in these single
observations and remain outside the product dispatcher. Full local checks
pass 365 Python tests with 2 skipped, 47 Rust tests, C smoke, clippy and
public hygiene. No new product or release qualification is claimed.

The experiment reuses original BF16 K64 slabs across a two-dimensional
candidate tile and falls back to the original dot for dense tiles. Continue
with structural changes to K16 arithmetic and reusable row metadata.
Command file: `run-native-tiled-projection-whole-r1.ps1 -Action reference-test`.
Evidence: `benchmarks/correctness/tiled-dense-projection-20260913.json`,
SHA256 `b596dcc248a3125402b3f6680f36ba89784f489487504729c73e6fde87f8eb50`.


Reusable K16 row bounds at source `0fae245` also preserve all 58728448
GB10 QKV cells and all 3791742 candidate identities. CPU checks cover all
65536 BF16 encodings and 2097152 ordered groups; 20 native comparisons
check 902732 complete carry states, metadata, guards and immutable inputs.
Full local checks pass 367 Python tests with 2 skipped, 47 Rust tests,
C smoke, clippy and public hygiene. The candidate is slower:
52.7489 / 61.7995 ms including preparation and replay. It remains outside
the product dispatcher, with no new token-loop or release acceptance.

A separate untimed accounting pass records 97753728 ineligible fallback
groups out of 485342976. Original input words all satisfy the old range;
229376 QKV weight words do not, including 216676 with biased exponent 4.
Continue investigating exact scalar alignment across this wider range and
fallback control flow. Command file:
`run-native-bounded-projection-whole-r1.ps1 -Action reference-test`.
Evidence: `benchmarks/correctness/float-row-bounds-projection-20260913.json`,
SHA256 `7b65cbe49c208b2c1361058357bd5b59784c0c87cb06ed17ea6b9c5c401c8591`.


The exponent-aligned scalar component at source `d13d36a` supports all
normal BF16 source exponents while preserving original K16 arithmetic. It
passes 2228224 exhaustive aligned-term combinations, 2097152 CPU carry
steps, and 30 native cases with 1354098 complete carry comparisons. Both
component variants preserve all 58728448 GB10 QKV cells, all raw output
bits and all 3791742 candidate identities. The new route uses scalar
arithmetic for every one of the 485342976 replayed groups, including the
original tiny weights, with the same 61444-byte row-flag workspace.

Completed preparation plus replay is 49.3512 / 46.9501 ms for original/new
in one observation. The 2.4011 ms component gain does not establish a
seconds-scale product improvement, so this helper remains outside the
product dispatcher. Full checks pass 369 Python tests with 2 skipped,
47 Rust tests, C smoke, clippy and public hygiene. Continue broader
full-model replay experiments with the original GB10 token/logit boundary.
Command file: `run-native-scaled-projection-whole-r1.ps1 -Action reference-test`.
Evidence: `benchmarks/correctness/scaled-significand-projection-20260914.json`,
SHA256 `ecc9711752b25a52ee4353be7d34549dd68d69219215230f228cfe1a337363a8`.

Four same-DLL q8192/512 admission experiments fail the original GB10
continuation boundary. Combined quarter, dense-only quarter, MoE-only
quarter and combined half PPB settings match 114 / 115 / 50 / 58 of 512
positions; their first mismatches are at zero-based indices 2 / 2 / 1 / 8.
All finish with passing host guards, 512 actual callbacks, first token 144
and first-logit error at most 0.0625. This first-token agreement does not
qualify the failed continuations. Observed callback walls
31976.7090 / 35359.5935 / 35793.3683 / 34521.9848 ms are diagnostic only.

Complete environment comparisons verify only the declared selector PPBs
changed. Keep the qualified dense/MoE PPB 1000 and full-attention OUT PPB
10000, with the original GB10 tolerance 0.125 and all 512 expected tokens.
The product dispatcher and defaults are unchanged. Continue broader exact
arithmetic replacements; these observations do not constrain materially
different designs. Command files: `run-native-replay-admission-product-r1.ps1`,
`run-native-replay-admission-isolation-r1.ps1` and
`run-native-replay-admission-half-r1.ps1`. No new release qualification.
Evidence: `benchmarks/correctness/replay-admission-product-failures-20260914.json`,
SHA256 `01d73e7965f15ffc95d63f8bd0152222cd8b57362fcd05b1312438443541e0aa`.

Complete-row exponent maxima at source `dfd681c` preserve all 58728448
original GB10 QKV cells, raw output bits and 3791742 selected identities.
Forty native cases also preserve 11592 outputs and 1805464 complete K16
carry states, with independent CPU metadata, production/diagnostic parity,
unaligned inputs and memory guards. Full checks pass 371 Python tests with
2 skipped, 47 Rust tests, C smoke, clippy and public hygiene.

The candidate retains the original 61444-byte row metadata allocation and
certifies 181901255 of 485342976 real K16 groups without a product maximum
reduction. Another 205687993 groups retain that reduction and 97753728 use
the original ineligible-row fallback. Completed preparation plus replay is
55.0315 / 56.6621 ms for original/candidate, a 1.6306 ms increase in this
single observation. Keep it outside the product dispatcher and pursue a
broader compute replacement. No token-loop or release qualification follows.
Command file: `run-native-row-max-projection-whole-r1.ps1 -Action reference-test`.
Evidence: `benchmarks/correctness/row-maximum-projection-20260914.json`,
SHA256 `86ab77ef43c721c7d847595a73281227c0d825684323fd83bafac5c0962a4b61`.

The earlier scalar float GDN WU/output matrices at FLA source `186da01` pass same-DLL
q8192 OFF/ON controls on baiying with `D:\models\Qwen3.6-35B-A3B`. Both
match all 512 original GB10 output tokens and actual callbacks, first 144
and exact raw logit 10.375. Callback TTFT is 41738.0364 / 40707.8780 ms,
a 1030.1584 ms reduction in this single pair. Enabled TPOT is 101.156079 ms
and model plus engine load is 21207.7142 ms. Use
`QRT_FLA_GDN_SCALAR_FLOAT_MATRICES=1` in subsequent candidates; source
default remains 0 pending broader qualification. Tiled KKT 2, whole/MoE
`9f00db5` registered metadata 1 and both prevalidated float options 1,
CK `6c0c54c` scalar QK 1 and CLI `a797b62` remain common. Command file:
`run-native-fla-scalar-matrices-product-r1.ps1 -ScalarMatrices 0|1`.

The new kernels reuse lossless packed BF16 rows and eligibility in shared
memory, assigning one thread to each result. They preserve ordered K16
carry, original fallback, BF16 boundaries and output FMA order. A WU CTA
captures all V rows in its eight columns before writing U, retaining U=V
ownership. q64, q65 and original GB10 q7169 comparisons preserve every
output and final-state bit, with synchronous/asynchronous parity. Full
`make check` passes 363 Python tests with 2 skipped, 47 Rust tests, C smoke,
clippy and public hygiene. The enabled product records 240 WU and 240 output
scalar-matrix launches. Its legacy `cooperative_lanes=4` field describes
the outer selected route; separate scalar-matrix markers identify the
actual WU/output dispatch.

Completed existing event diagnostics show WU 827.83775 / 445.71661 ms and
output 1778.16570 / 1131.34616 ms across 240 segments. Its enabled persistent
state took 1967.73891 ms before the state replacement recorded above. These scopes
are inside the linear-core wall and do not replace callback TTFT. q8192
remains above 10 seconds and the immutable 4187.415605 ms target; no new
long-context, checkpoint, package, HTTP, soak or release acceptance follows.
Evidence: `benchmarks/correctness/fla-scalar-matrices-product-20260913.json`,
SHA256 `9cff463f8495e0da995516f12f0812b3101a65a5233eaa08a48cf65377b508b7`.

Tiled scalar KKT at FLA source `62d5fd4` passes complete same-DLL q8192
OFF/ON controls on baiying, `D:\models\Qwen3.6-35B-A3B`. All 512 original
GB10 output tokens and actual callbacks match, with first 144 and exact raw
logit 10.375. Callback TTFT is 42180.7213 / 41931.5687 ms, a 249.1526 ms
reduction in this single pair. Enabled TPOT is 100.911523 ms and load is
21291.7237 ms. Use `QRT_FLA_GDN_TILED_KKT=2` in subsequent candidates;
source default stays 0 pending broader qualification. Whole/MoE `9f00db5`
registered metadata 1 and both prevalidated float options 1, CK `6c0c54c`
scalar QK 1, and CLI `a797b62` remain common. Command file:
`run-native-tiled-kkt-product-r1.ps1 -TiledKkt 0|2`.

Each 8-query by 32-column tile reuses the original rounded beta*K rows and
K columns. One thread owns each output; the ordered K16 carry, fallback,
upper triangle and later gate kernel are preserved. All 18 native controls
match 39845888 raw outputs and 4608 independent CPU dots. Generated q8192
original / tiled integer / tiled float takes 19.1582 / 8.0214 / 5.1945 ms;
only the complete token runs establish product gain. Full FLA q64, q65 and
original GB10 q7169 comparisons preserve every output and final-state bit,
with synchronous/asynchronous parity. Full `make check` passes 362 Python
tests with 2 skipped, 47 Rust tests, C smoke, clippy and public hygiene.

Existing completed stage events report KKT 561.36287 / 171.05978 ms across
240 segments; enabled persistent-state and output stages still take
2014.55534 and 1767.07512 ms. These diagnostics sit inside the linear-core
scope and do not replace callback TTFT. Continue broader GDN matrix work.
q8192 remains above 10 seconds and the immutable 4187.415605 ms target;
long-context, prefix checkpoint, package, HTTP, soak and release remain open.
Evidence: `benchmarks/correctness/tiled-kkt-product-20260913.json`,
SHA256 `e0010a7c206f72178d32c52ebb5569721705c1b17b8762dcba2523c6cca0e9ff`.

Registered immutable MoE weight metadata at whole/MoE source `9f00db5`
passes same-DLL q8192 OFF/ON controls on baiying with the real model
`D:\models\Qwen3.6-35B-A3B`. Both match all 512 original GB10 output tokens
and actual callbacks, first token 144 and exact raw logit 10.375. Callback
TTFT is 43272.7319 / 42224.5708 ms, a 1048.1611 ms reduction in this single
pair. Model plus engine load is 20024.5016 / 21329.8927 ms, both within
30000 ms; enabled TPOT is 101.249185 ms. Use
`QRT_QWEN36_MOE_REGISTERED_WEIGHT_METADATA=1` in subsequent candidates;
source default remains 0 pending broader qualification. Both prevalidated
float replay options remain 1, CK `6c0c54c` scalar QK 1, FLA `8f436db`, and
CLI `a797b62` are common. The command file is
`run-native-registered-metadata-product-r1.ps1 -RegisteredWeightMetadata 0|1`.

Registration runs the original FP64 weight norm and eligibility scan during
loading and owns 240 MiB for 40 layer pairs. The enabled run records one
complete table before request entry, 80 matching metadata copies into the
original scratch, and explicit invalidation after the final decode token.
Invalidation clears every source identity before raw weights are released;
new registration drains old readers before rebuilding. Partial failures
never publish a table and retain ownership for drained cleanup. Full
`make check` passes 361 Python tests with 2 skipped, 47 Rust tests, C smoke,
clippy and public hygiene; sanitized lifecycle checks cover failures,
address reuse and deferred frees. Source `9f00db5` also fixes the legacy
prepared-operands marker to report encoded operands only. No new long-context,
checkpoint, package, HTTP, soak or release qualification follows. q8192 is
still above 10 seconds and the immutable 4187.415605 ms target.
Evidence: `benchmarks/correctness/registered-weight-metadata-product-20260913.json`,
SHA256 `06d156a6156fa29b462eb9a29a3a88daede41bbb10f92b03c67df4cd3c280bb6`.

Source `d2f283f`'s synchronized q8192 profile still matches all 512 GB10
tokens/callbacks and exact first logit. Completed host walls are 6607.706 ms
for all 30 linear input projections, 10841.934 ms for their linear cores
including output projection, 10502.727 ms for all 40 MoE calls, and
14292.800 ms for the full attention pipeline. These scopes are nested and
must not be summed. Some GPU subintervals remain negative, so their sums do
not establish a reliable additive breakdown. Instrumented callback TTFT
45053.0266 ms does not replace the uninstrumented 43324.5935 ms result.
Evidence: `benchmarks/correctness/prevalidated-float-wall-profile-20260913.json`,
SHA256 `46cc00d8851eaf96bcb872cf6521d969db239c3566877a9d594df75767350b15`.

The cooperative scalar-float QK component at source `ae2256d` is correct
but slower. All 171 generated cases match 292498128 raw score slots; each
of the nine variants also matches 418496528 original real q7169 score slots
and 228 independent CPU dots. Guards, input immutability and fallback pass.
The same executable measures original integer QK at 453.4480 ms, current
scalar float at 344.6427 ms, and the fastest new cooperative variant at
491.0003 ms. Keep the product's scalar QK setting; these eight cooperative
layouts remain component diagnostics. The next investigation targets broader
linear-attention matrix work. No new product or release acceptance follows.
Evidence: `benchmarks/correctness/float-subgroup-qk-20260913.json`,
SHA256 `8f93aed84d81cf4c51574f922e72be405d2ce489b622450338f382ef0420d04a`.

Row-prevalidated cooperative float replay at whole/MoE source `d2f283f`
passes complete same-DLL q8192 controls. Baseline / both enabled / dense only
match all 512 original GB10 output tokens and actual callbacks, with first
144 and exact raw logit 10.375. Callback TTFT is
44254.7018 / 43324.5935 / 43506.9730 ms; dense correction host wall is
8104.932 / 7328.709 / 7322.577 ms. Dense only saves 747.7288 ms and adding
MoE saves another 182.3795 ms in this single comparison. Retain
`QRT_QWEN36_HAWKEYE_PREVALIDATED_FLOAT_REPLAY=1` and
`QRT_QWEN36_MOE_PREVALIDATED_FLOAT_REPLAY=1` for subsequent candidate runs;
source defaults remain 0 pending broader qualification. Original per-pair
float replay and prepared MoE stay 0. CK `6c0c54c` scalar QK 1, FLA `8f436db`
and CLI `a797b62` remain common. Both-enabled TPOT is 101.887785 ms and
load is 20043.4378 ms. q8192 remains above 10 seconds and the immutable
4187.415605 ms target; no long-context, package, HTTP, soak or release
qualification follows.

Dense replay keeps only row flags; MoE produces its flags in the original
FP64 norm scan with 2359296 additional bytes. Native checks pass 24 core
cases, two bit-identical norm scans, four routed gate/up/down controls,
four dense correction cases and all 58728448 real captured GB10 QKV BF16
endpoints. Guards and immutable inputs pass. The legacy prepared-operands
marker also follows flags-only storage in this source; the analyzer checks
its exact byte count and requires separate prevalidated-float markers.
Original raw evidence remains unchanged. The test-only follow-up `29fe4cb`
keeps native/build sources identical and passes full `make check`: 47 Rust
tests, 360 Python tests with 2 skipped, C smoke, clippy and public hygiene.
The completed phase profile of this token case is recorded above.
Evidence: `benchmarks/correctness/prevalidated-float-product-20260913.json`,
SHA256 `f9e7b6c1fc38765b9ef1a55984375a283d220755dd3a98936aa428fd795b8cdd`.

The component comparison at source `75d5e56` retains provenance in
`benchmarks/correctness/scalar-projection-20260913.json`: all 12 generated
cases and all captured QKV endpoints match. Original prepared4 / scalar
row-major / scalar transposed / prevalidated cooperative4 preparation plus
replay is 54.6064 / 54.7826 / 304.4780 / 50.3057 ms. Those timings exclude
allocation, upload and candidate collection; only the complete product
controls above establish a q8192 gain.

Cooperative scalar float replay at whole/MoE source `9112d0b` preserves
correctness but is slower. Three real q8192 controls match all 512 GB10 output
tokens and callbacks, with first token 144 and exact raw logit 10.375.
Baseline / both enabled / dense only callback TTFT is
44229.9633 / 45788.2217 / 44865.9197 ms. Dense only adds 635.9564 ms;
adding MoE to dense adds 922.3020 ms. Dense correction host wall is
8097.165 / 8699.474 / 8706.942 ms. Keep both
`QRT_QWEN36_HAWKEYE_FLOAT_REPLAY` and `QRT_QWEN36_MOE_FLOAT_REPLAY`
at 0. All three settings use the same whole/MoE DLLs, CK `6c0c54c` scalar
QK 1, FLA `8f436db`, and CLI `a797b62`; load remains about 20 seconds.
This is one observation per setting, without release qualification.

Native checks also pass all 3563520 intermediate K16 endpoints, generated
dense correction and routed gate/up/down cases, and every one of 58728448
real captured GB10 QKV BF16 cells. Guards, immutable inputs and host checks
pass. The derived analyzer's duplicated prepared-marker assertion was fixed
to allow preparation to be skipped when float replay is active; original
native evidence is preserved. The next investigation targets the amount of
exact replay required by error admission and attention PV, while preserving
the GB10 boundary. Evidence:
`benchmarks/correctness/float-replay-product-20260913.json` and
`benchmarks/correctness/float-subgroup-20260913.json`.

Exact scalar float QK at CK source `6c0c54c` passes complete same-DLL
q8192 OFF/ON validation: all 512 GB10 outputs and callbacks match, with exact
first token 144 and raw logit 10.375. Callback TTFT is 46064.2854/44535.0777 ms,
TPOT 101.936076/101.197382 ms, and load 20034.3533/20021.0454 ms. Only
`QRT_CK_SM121_FLOAT_ALIGNMENT_QK=0|1` differs. The single pair saves
1529.2077 ms without extra workspace; use 1 in subsequent candidate runs.
The source default remains 0 pending wider validation. Whole `4fec3ea`,
FLA-MoE `8f436db`, CLI `a797b62`, selective QK 0 and direct PV 1 remain common.
All host guards pass. q8192 is still above 10 seconds and the immutable
4187.415605 ms target; long-context, package, HTTP, soak and release gates
remain open. The cooperative dense/MoE follow-up and its negative performance
result are recorded above.
Evidence: `benchmarks/correctness/float-alignment-product-20260913.json`.

Scalar FP32 products with canonical integer alignment at source `f15ea53`
pass all 1048576 intermediate K16 endpoints and 65536 dots on gfx1151,
including original fallback. Fourteen generated QK cases match 32182304
score slots and 896 CPU dots; real q7169 Q/K match 418496528 slots and
228 CPU dots. Guards and inputs pass. Query time is 338.3151 ms against
447.4125 ms original, with the same 5.1814 ms key transpose. This is a single
component observation, not retained product performance. The product implementation and complete same-DLL q8192 validation are
recorded above; no release qualification follows yet. Evidence:
`benchmarks/correctness/float-alignment-qk-20260913.json`.

The compact 18-bit core at source `609f468` is exact but remains slower.
Its native primitive enumerates all 262144 core encodings and matches
1048576 independent integer dots and all three raw matrix partials. Fourteen
QK cases match 32182304 score slots and 896 CPU dots; original real q7169
Q/K match 418496528 slots and 228 CPU dots. Guards and inputs pass. The
132-byte row reduces generated replay work, but completed QK plus key
preparation takes 554.0810 ms against original QK plus transpose 446.3313 ms.
Keep this component outside product dispatch. No model load or token run is
claimed. The scalar FP32 follow-up and its component measurements are recorded above.
Evidence: `benchmarks/correctness/wide-integer-qk-20260913.json`.

Nonnegative three-part integer reconstruction is exact but slower. The
primitive source91b069b passes1048576 independent CPU int64 dots with zero
raw partial differences. Prepared QK source31d2b91 then passes28 generated
cases/64364608 score slots and1792 CPU dots; original q7169 Q/K match
836993056 score slots and456 CPU dots. All buffers/immutable inputs pass.
Prepared positive QK plus key encoding takes631.0199ms, versus original
scalar QK plus transpose440.4953ms. Four-IU8 on the same148-byte extended
row layout takes597.7987ms. This component stays outside product dispatch.
No model is loaded and no product tokens or performance are accepted.
The wider-core follow-up and its measured decision are recorded above.
Evidence: `benchmarks/correctness/positive-integer-qk-20260913.json`.

The signed three-BF16 integer-core candidate is rejected at source1738c5f.
Native1048576 cells contain566483 raw partial differences and263108
reconstruction differences; four-IU8 reference partials, guards and inputs
pass. Source dca442c isolates the effect: all196608 core conversions are
exact, but mixed-sign BF16 WMMA, FP16 WMMA and packed BF16 DOT2 each differ
on67345 of262144 cells, with maximum observed error0.015625. Scalar FP32
and the five other tested sign/pattern families are exact. Observed nearest-
integer agreement is diagnostic; no snapping or general error bound is
accepted. The current runtime remains unchanged. Next investigate a
nonnegative third partial with exact integer centering correction.

The probe also repairs a real runner issue at source182492c: omit an empty
Start-Process ArgumentList while preserving quoting and job ownership.
Four Windows argument cases pass, and the unchanged no-argument GPU test
then starts and records its numerical failure correctly. No product timing
is accepted from these component diagnostics. Evidence:
`benchmarks/correctness/integer-matrix-contract-20260913.json`.

Routed MoE prepared replay at source7af2c3c passes both complete q8192
controls: all512 GB10 tokens/callbacks and exact first144/logit10.375. Only
`QRT_QWEN36_MOE_PREPARED_REPLAY=0|1` changes. Disabled/enabled callback TTFT
is45905.0608/46197.6074ms, TPOT101.531248/100.843159ms, and load
20036.5836/20084.2981ms. Enabled preparation adds1143209984 bytes and
292.5466ms in this single pair. Keep it disabled; no performance gain or new
release qualification follows. Native fused scans preserve original FP64 norm
bits across663040 encoded cells; four routed cases through1025 tokens preserve
gate/up/down bits, including unsupported-row fallback and compaction windows.
The next investigation targets the repeated exact K16 matrix core. The prior
floating mantissa counterexample remains evidence; native arithmetic must be
verified before any product dispatch. Evidence:
`benchmarks/correctness/moe-prepared-replay-20260913.json`.

Completed workspace and matrix attribution at wholea9f0d0f/5feb7e7 keeps
CKe9673f8 selective QK0/direct PV1, FLA-MoE8f and CLIa797. Both real q8192
runs pass all512 GB10 outputs/callbacks and exact first144/logit10.375.
The170 dense workspace allocations/frees total47.7403/48.6635ms; reuse of
this measured surface cannot recover seconds. The apparent8001.004ms wait
before linear QKV includes previous queued work. Separate completed intervals
find6361.2797ms predecessor wait and1590.4194ms for its30 matrix calls.
All110 profiled matrix calls total3925.7338ms, with2.7024ms plan setup.
The extra profile fences can alter stream overlap; these are diagnostic
measurements, with no retained-performance acceptance. Callback TTFT is
46105.0172/46045.1689ms, still far above the immutable target. Next evaluate
reusable routed-MoE replay operands produced during required norm scans,
with the completed negative result recorded above.
Evidence: `benchmarks/correctness/dense-producer-wall-20260913.json`.

Output-margin denominator refinement at sourcea25b1c3 passes all35 generated
cases /6307840 outputs and all29364224 originalq7169 GB10 BF16 cells.
The output margin chooses work only; strict interval admission and complete
row fallback remain authoritative. Replay drops from98.0597% to88.7196%
(364827389 of411213840 scores), but repair/probability still takes2408.1079ms,
against original QK/probability431.5555/87.0005ms. This strict component stays
outside product dispatch. Runtime helper math is unchanged and its previously
failed approximate denominator remains disabled. The next structural check
measures repeated prefill workspace allocation/free before choosing reuse.
No model or product tokens are submitted by this component. Evidence:
`benchmarks/correctness/denominator-output-budget-20260913.json`.

The approximate-denominator product trial at sourcee9673f8 fails the GB10
continuation boundary. On baiying with the real model, the enabled path returns
244 instead of255 at output index1 and matches118/512 positions. First144 and
raw logit10.3125 satisfy the first-token tolerance, but cannot qualify this run.
All512 actual callbacks complete and host checks pass; native exit6 records
the token-contract failure. Keep `QRT_CK_SM121_SELECTIVE_QK_PROBABILITY=0`.
The disabled same-DLL control matches all512 outputs/callbacks and exact
first144/logit10.375, with callback TTFT46169.357ms. The enabled elapsed time
is diagnostic only. Component checks establish exact probabilities/alpha and
agreement with canonical PV at the same approximate denominator;1726 captured
BF16 context cells differ from GB10. Strict denominator correctness remains
necessary for the next route. Evidence:
`benchmarks/correctness/selective-qk-probability-20260913.json`.

Selective QK source90609e4 now constructs its candidate from original Q/K,
without reading baseline scores. All35 generated cases /6307840 output cells
and all29364224 originalq7169 GB10 BF16 cells pass. Probability and alpha
endpoints agree, all score/denominator intervals hold, and guards pass. The
strict denominator repair nevertheless recomputes403235056 of411213840
scores (98.0597%). Native bounded QK takes99.5748ms and the repair/probability
pipeline2512.3431ms; original QK/probability take431.8602/86.9274ms.
Candidate PV is canonical for every cell in this component. This route remains
outside product dispatch. The subsequent approximate-denominator product
trial is recorded above. Evidence:
`benchmarks/correctness/selective-qk-20260913.json`.

The QK numerical decomposition at sourcee373d47 keeps the product unchanged.
On originalq7169 layer3 Q/K/V, canonical attention matches all29364224 GB10
BF16 cells. Native QK changes45648 outputs. Exact prefix maxima and maximum
anchors reduce this to24345; isolated probability and denominator changes
affect22722 and1853 outputs respectively. All six variants are finite and
all whole-allocation guards and immutable inputs pass. Exact QK constructs
the diagnostic hybrids; these counts do not demonstrate avoided score work
or product performance. The two earlier invalid-domain diagnostic attempts
remain attached as failures. The resulting selective score repair and
denominator/output fallback are recorded above. Evidence:
`benchmarks/correctness/qk-probability-decomposition-20260913.json`.

The subsequent native HIP wait diagnostic finds no millisecond fixed completion
penalty in its one-stream component workload. Source548385b observes flags1 in
the untouched process; default, explicit Spin and explicit Auto have shortest
kernel/event medians around32us. Blocking waits are slightly slower. All60
cases pass final output samples and redzones across1920 completed submissions.
Host intervals include submission. No model is loaded, and this does not
establish product timing or all-stream API equivalence. Runtime flags and
completion semantics remain unchanged. Evidence:
`benchmarks/correctness/hip-completion-wait-20260913.json`.

The current corrected runtime has not recovered the retained performance
targets. The following runs use the actual model on baiying, with startup
excluded from the first streamed callback clock. Each listed case generates
all 512 GB10 tokens and an exact first-token raw logit. Both prefix runs also
verify their fallback transaction and complete owner-state restoration.

| Build / route | Actual prompt shape | Callback TTFT ms | TPOT ms | Load ms |
|---|---|---:|---:|---:|
| whole 730a855 / CK 5816925, bound 10000 | 32768 prefix + 1024 suffix | 25597.6857 | 229.372641 | 20055.0067 |
| whole/CK 9871ef2, chunking enabled | cold 17408, chunks 8192+8192+1024 | 170613.1127 | 162.553605 | 20066.476999 |
| same new binaries, q8192 control | cold 8192, ordinary path | 62478.683599 | 110.904104 | 20046.251001 |
| same new binaries, chunked owner | 16384 prefix + 1024 suffix | 17733.562599 | 161.906225 | 20033.961899 |
| MoE 6278fe0, original 1024-block window | cold 8192, ordinary path | 62231.9179 | 112.049735 | 20085.2361 |
| same binary, 16384-block window | cold 8192, ordinary path | 59348.223499 | 111.265862 | 20039.383501 |
| MoE 25c3693, DPP integer reductions | cold 8192, ordinary path | 58243.9345 | 112.070807 | 20050.7749 |
| whole/CK/FLA 930955a, same DPP MoE | cold 8192, ordinary path | 57617.9907 | 107.136582 | 20077.7027 |
| CK d028182, globally compacted PV replay | cold 8192, ordinary path | 56604.0518 | 106.860717 | 20074.636 |
| MoE 8a7a8dc, staged K64 operands | cold 8192, ordinary path | 55453.937 | 105.957922 | 20116.758099 |
| whole/CK/FLA/MoE 8f436db, compact normalization | cold 8192, ordinary path | 53818.8428 | 102.263044 | 20068.490699 |
| MoE a012e01, exact dot tile certificate (not retained) | cold 8192, ordinary path | 56050.9289 | 101.508477 | 20076.2551 |
| whole 57b3306, original dense dispatch | cold 8192, ordinary path | 53953.8059 | 100.925442 | 20031.092701 |
| same binary, device-count dense replay | cold 8192, ordinary path | 53534.7813 | 100.702816 | 20018.5822 |
| whole 584588a, original matrix producer | cold 8192, ordinary path | 53982.562999 | 100.835152 | 20032.911 |
| same binary, hipBLASLt matrix producer | cold 8192, ordinary path | 52110.692099 | 101.46877 | 20017.0856 |
| CK 42448fe, same binary / 32 queries | cold 8192, ordinary path | 52167.1323 | 101.064893 | 20041.6208 |
| same binary / 128 queries | cold 8192, ordinary path | 50764.6481 | 101.346222 | 20096.24 |
| CK 9a7eaf4, 128 queries / V transpose disabled | cold 8192, ordinary path | 50888.006 | 101.364914 | 20059.2105 |
| same binary / V transpose enabled | cold 8192, ordinary path | 49691.0931 | 101.291451 | 20017.5312 |
| same 9a7eaf4 source, lossless compact exp2 table | cold 8192, ordinary path | 49519.006 | 101.043432 | 20072.6491 |
| whole 8cc97d6, prepared operands disabled | cold 8192, ordinary path | 49840.7445 | 101.842764 | 20028.7697 |
| same binary, prepared operands enabled | cold 8192, ordinary path | 48202.0546 | 102.12639 | 20006.0793 |
| whole 1a7f621, absolute bound disabled / prepared enabled | cold 8192, ordinary path | 48111.7024 | 101.159201 | 20065.8768 |
| whole 2e49049, absolute bound disabled / prepared enabled | cold 8192, ordinary path | 48017.509 | 101.468183 | 20041.4914 |
| whole 2e49049 / CK 6520982, final PV envelope disabled | cold 8192, ordinary path | 48223.2078 | 101.959194 | 20074.458 |
| same binaries, final PV envelope enabled | cold 8192, ordinary path | 47785.5537 | 101.329395 | 20064.6161 |
| whole 4fec3ea / CK 126cdcb, shared PV operand loading | cold 8192, ordinary path | 47770.6281 | 101.6759 | 20008.4729 |
| same binaries, direct PV operand loading | cold 8192, ordinary path | 46055.3727 | 101.615746 | 20023.7468 |

Direct PV operands remove per-K16 shared publication/retirement barriers while
keeping the original BF16 inputs, matrix calls, ordered carries, final error
envelope and exact replay. Both complete product runs match all512 GB10 outputs
and actual callbacks with exact first144/logit10.375. Only
`QRT_CK_SM121_DIRECT_PV_OPERANDS=0|1` changes in the resolved environment;
whole prepared operands stay enabled with the original layout. This pair
observes1715.2554ms lower callback TTFT. Keep the option enabled for subsequent
qualified-stack experiments, with default0. No retained target is recovered.

Native24 cases /4423680 cells compare four shared/direct and original/final
envelope variants. Original output/carry/denominator bits agree, direct bounds
and candidate sets equal their shared controls, and all17694720 canonical BF16
comparisons pass. Originalq7169 attention matches every29364224 GB10 BF16 cell
and all four qualified replay files in both modes. Completed query plus V
preparation is773.9421/913.7159ms; this component comparison excludes common
upload, allocation, key transpose and safety validation. Neither pair is a
sustained-throughput claim, and other contexts/package/HTTP/soak remain open.
Evidence: `benchmarks/correctness/direct-pv-operands-20260913.json`.

Direct integer QK does not share that gain. Source6142e45 passes28 native
cases /64364608 raw-score comparisons and1792 CPU sampled dots. Both direct
and shared integer schedules match836993056 originalq7169 raw scores and456
CPU dots. With all operand preparation included, direct/shared completion is
687.154/542.3215ms, against original scalar plus key transpose446.0553ms.
Keep this component experiment outside product dispatch. The scalar QK and
integer-core arithmetic are unchanged; no model or token result is submitted.
Evidence: `benchmarks/correctness/direct-integer-qk-20260913.json`.

The optional final PV envelope preserves native matrix arithmetic and enlarges
the existing error envelope. Both q8192 runs above match all 512 GB10 outputs
and actual callbacks with exact first token 144 / raw logit 10.375. The sole
resolved environment change is `QRT_CK_SM121_FINAL_PV_BOUND=0|1`; dense
absolute-product admission remains disabled. This one pair observes a
437.6541 ms callback reduction. It does not establish a sustained speedup or
recover the retained targets, and does not qualify other contexts or release.

On the original q7169 attention capture, enabled/disabled completed query
time plus V preparation is 886.8647 / 927.6509 ms, with all 29,364,224 GB10
BF16 cells matching. Candidates increase from 2,181,900 to 2,198,673.
The first implementation was slower at 1096.6938 / 903.2955 ms because its
volatile metadata generated extra private-memory traffic. Explicit FP32
register operations remove that overhead while preserving all captured output
file hashes. The revised native checks cover 4,210,688 envelope comparisons
and 4,423,680 kernel cells without shrinking bounds or candidate sets.
Keep the route optional and move to changes capable of removing seconds from
dominant attention/dense work. Evidence:
`benchmarks/correctness/final-pv-bound-20260913.json`.

The resident hipBLASLt follow-up also leaves absolute-product admission
disabled in the selected stack. It passes110430 native double-reference
cells,62930951 integration endpoints and all58728448 original QKV cells.
The QKV bound matrix costs52.629ms and total correction117.598ms. However,
the actualq8192/out512 candidate, source2e49049, fails the GB10 continuation:
first mismatch is zero-based index115, expected271 versus actual196;
only130 of512 positions match. Its first144/logit10.3125 passes the0.125
first-logit tolerance, which does not qualify the failed continuation.

The same-DLL bound-disabled control matches all512 GB10 outputs and actual
callbacks with exact144/logit10.375 at48017.509ms. No candidate performance
result is retained. A conservative absolute-product sum on the tested cells
does not establish that the existing empirical projection-error coefficients
remain valid with tighter admission. The subsequent admission audit localizes
the first BF16 difference to layer0 linear OUT: the absolute-product magnitude
bound is conservative, but its unchanged1000PPB multiplier misses the
producer/canonical difference. Retain the Cauchy configuration; the completed
phase profile is below. Evidence:
`benchmarks/correctness/absolute-product-hipblaslt-20260913.json`.

The refreshed synchronized profile uses this restored2e49049 control and
passes all512 GB10 outputs/callbacks with exact first144/logit10.375.
Completed host observations are18334.4ms for the full attention pipeline,
10690.656ms for MoE,6890.107ms for linear projections and10894.705ms for
the linear core including its output projection. Coarse buckets and nested
observers are not additive. GPU subphase intervals still include negatives;
their raw values are retained without clamping or a validated additive
attribution. The instrumented49635.3398ms callback is diagnostic and does
not replace the48017.509ms uninstrumented observation. The final PV envelope
above addresses part of this work; attention and dense execution still need
broader changes while retaining their GB10 boundaries. Evidence:
`benchmarks/correctness/prepared-stack-wall-profile-20260913.json`.

The bounded absolute-product experiment has not passed its product request.
Source1a7f621 uses WMMA to compute tighter per-cell admission metadata while
retaining midpoint radii, PPB and exact arithmetic. Twelve native windows
cover110430 cells with no underestimate against independent double sums;
62930951 integration outputs and all58728448 original QKV BF16 cells pass.
On the QKV capture, candidates fall3792033→2908868, but bound generation costs
96.820ms and completed correction wall rises to158.23ms.

On baiying, the actualq8192 candidate stops in layer3 before any token or
callback: a9216-row QKV bound window completes in228.204ms, beyond the100ms
dispatch limit. The process returns5 and releases its allocation pool.
There is no candidate TTFT or product correctness result. The same-DLL
bound-disabled control passes all512 GB10 outputs and callbacks with exact
first144/logit10.375 at48111.7024ms. Only the bound flag differs. Keep the
bound disabled in the selected stack; the resident backend follow-up above
also fails product correctness. Evidence:
`benchmarks/correctness/absolute-product-admission-20260913.json`.

Dense projection replay now optionally prepares a lossless16-bit operand view
once per correction call. A row containing an excluded BF16 value uses its
original operands and exact dot. K16 order, all admission bounds and endpoint
rounding remain unchanged. The same-DLL q8192 pair differs only in
`QRT_QWEN36_HAWKEYE_PREPARED_OPERANDS=0|1`; both pass all512 original GB10
tokens and actual callbacks with first144/logit10.375 exactly. The observed
callback reduction is1638.6899ms. Completed dense-correction host wall,
including preparation, falls from9581.693 to8157.930ms; allocation and cleanup
remain included in the actual callback clock. This is one run per setting,
without a statistical repeatability claim.

The native component checks cover6987 raw dots against independent CPU and
original GPU arithmetic,12883520 prepared cells,62930951 integration outputs,
mixed row fallback, immutable inputs and redzones. The originalq7169 QKV
capture also matches all58728448 external BF16 cells. The optional allocation
is bounded by the admitted shape and freed before return. Enable prepared
operands in subsequent experiments with whole8cc97d6 and the existing compact
CK9a7eaf4/FLA-MoE8f436db stack. Default remains disabled; this does not qualify
other contexts, a new package or retained performance. Evidence:
`benchmarks/correctness/prepared-projection-operands-20260913.json`.

A separate QK experiment lets one thread reuse a key operand across two
independent query rows. Fourteen native cases check32182304 raw scores and896
independent CPU dots with no differences. Same-executable originalq7169
attention runs match all29364224 external BF16 cells: query plus V-preparation
host wall is920.561ms for one row and893.9971ms for two. The single-pair
26.5639ms component difference is not a product performance result. Keep the
existing provider and prioritize reducing the amount of exact replay work;
the new shared API defaults to one row and has no provider integration.
See `benchmarks/correctness/paired-query-qk-20260913.json`.

The same-binary window pair changes only
`QRT_QWEN36_MOE_COMPACTION_WINDOW_BLOCKS`. The wider collection and bounded
persistent replay save 2883.694401 ms in this measured pair, using 15 MiB more
scratch; both complete GB10 boundaries pass. This is one paired measurement,
with no statistical repeatability claim. The default remains 1024 blocks;
16384 is retained for further experiments. Its separate synchronized profile
also passes all 512 outputs and callbacks, with 13997.512 ms across 40 MoE host
calls. Raw GPU intervals include invalid negative values and do not support
an additive kernel breakdown. See
`benchmarks/correctness/moe-wide-compaction-20260913.json`.

The DPP experiment changes integer lane transport, preserving the exact
arithmetic and every correction bound. Both new q8192 runs pass all 512 GB10
tokens and callbacks with exact first token/logit. Native masked reductions
check 5,505,276 lane results, and 36,891 BF16 dots match the original scalar
algorithm bitwise. Disassembly confirms eight shuffle exchanges become eight
DPP operations. The full configuration is retained for further experiments;
the observed 1730.232799 ms difference is a comparison of single runs, without
a statistical repeatability claim. Build defaults remain disabled. Evidence:
`benchmarks/correctness/dpp-exact-reductions-20260913.json`.

The compacted PV experiment collects uncertain attention outputs across each
32-query batch, then replays their original ordered K16 arithmetic with four
lanes per output. All 512 GB10 tokens and callbacks pass with exact first
token 144 and logit 10.375. Its 56604.0518 ms callback is 1013.9389 ms below
the preceding DPP run; each configuration has one measurement. The original
q7169 component matches all 29,364,224 external BF16 cells, selecting
2,181,900 cells for correction. Completed host operator time is 1058.14 ms,
versus 1200.62 ms for the ordinary path and 1521.55 ms for the same-build
per-query replay control. Twenty native safety cases preserve raw outputs,
inputs and redzones. This becomes the next experimental baseline through
`QRT_CK_SM121_COMPACT_PV_REPLAY=1`; its default remains disabled. Evidence:
`benchmarks/correctness/attention-compact-pv-20260913.json`.

Staging four K16 operand groups before routed correction preserves the original
16-lane arithmetic and admission rules. The complete q8192/out512 boundary
passes with exact first token/logit at 55453.937 ms callback TTFT, an observed
1150.1148 ms reduction from the compact-PV baseline. All 73,782 native dot
comparisons, including partial tiles, match independent CPU arithmetic.
Emitted code confirms K64 operand loads without scratch spills. One negative
microtiming remains invalid; the product comparison contains one run each.
Use four staged groups in subsequent experiments; the build default remains
one. See `benchmarks/correctness/moe-staged-dot-20260913.json`.

The shared integer normalization now uses one magnitude alignment and a
combined truncating shift, preserving the original internal 26-bit grid.
All 4,194,304 wide CPU reference cases and 73,782 native dots pass. The full
q8192/out512 run matches every GB10 token, callback and exact first logit at
53818.8428 ms callback TTFT, 1635.0942 ms below the staged-MoE baseline.
This single-run comparison supports enabling compact normalization in further
experiments; its build default remains disabled. Broader contexts and release
acceptance remain open. See `benchmarks/correctness/canonical-normalize-20260913.json`.

An exact tile certificate skips repeated normalization only when every K16
intermediate retains the original sign and exponent. Its 86,079 native dots
match independent CPU arithmetic, exercising 161,050 admitted tiles. The full
q8192 run passes all 512 GB10 tokens and callbacks with an exact first logit,
but callback TTFT rises to 56050.9289 ms, an observed increase of 2232.0861 ms
over compact normalization. Keep `CertifiedDotTiles=0` and the 8f436db MoE.
These single runs establish no repeatability or real-model admission density;
one negative native microtiming is invalid. Evidence:
`benchmarks/correctness/certified-dot-20260913.json`.

Dense correction can now collect and round in one pass, then read its candidate
count on the GPU and replay bounded four-million-cell windows. Both same-binary
q8192 runs pass every GB10 token, callback and exact first logit. The observed
419.0246 ms difference accompanies correction host wall of 9512.093 versus
9103.645 ms and eliminates 330 host count reads. Native tests compare 8,874,156
output cells, full candidate windows, short split plans and all redzones.
The larger completed window has a 250 ms deadline, exercised at 101.385 ms for
maximum K and dense selection. This remains opt-in; subsequent producer
experiments use compact normalization with device replay disabled. One paired
measurement does not establish repeatability or qualify immutable performance.
See `benchmarks/correctness/dense-device-replay-20260913.json`.

The existing hipBLASLt BF16-input/F32-output plan now optionally produces the
70 QKV/Z matrices before their unchanged norm bounds and exact correction.
Both same-binary q8192 runs pass all 512 GB10 outputs and callbacks with exact
first token 144/logit 10.375. Callback TTFT changes from 53982.562999 to
52110.692099 ms, an observed 1871.8709 ms reduction in one paired measurement.
The original q7169 capture matches all 58,728,448 corrected BF16 QKV values;
eight native shapes check 49,158 BF16 endpoints and preserve buffers.
Use whole 584588a with `QRT_QWEN36_PREFILL_HIPBLASLT_PRODUCER=1`, qualified
8f436db CK/FLA/MoE and device replay disabled for further experiments. The
shipped producer default remains disabled. This adds no dependency and does
not qualify immutable performance or the pending broader release matrix.
Evidence: `benchmarks/correctness/hipblaslt-producer-20260913.json`.

CK `42448fe` widens the independent query slab from32 to128 while preserving
all QK, online probability and selective PV arithmetic. The existing scratch
already accommodates this q8192 shape, so allocated memory does not grow.
Twenty-nine native cases compare4325376 output cells with no raw-bit mismatch,
including candidate ownership, redzones and unchanged inputs. Both original
q7169 replays match all29364224 external BF16 cells; completed query-dispatch
host wall improves1071.21 to994.928ms, with identical output, accumulator and
denominator files.

The same CK binary at q8192/out512 gives callback52167.1323ms with32 queries
and50764.6481ms with128, a1402.4842ms improvement in this single paired run.
Both pass all512 original GB10 outputs and actual callbacks, with exact first
token144/logit10.375. Candidate TPOT is101.346222ms and load20096.24ms.
128 queries becomes the next experimental configuration; the shipped default
remains32. No immutable performance or broader release gate is qualified.
Evidence: `benchmarks/correctness/attention-wide-queries-20260913.json`.

An optional completion observer then isolates originalq7169 layer3 attention
with128 queries per slab. Completed host phases are QK437.1211ms,
probability135.1112ms, approximate PV244.2869ms, collection6.5589ms and exact
PV179.7545ms. All57 samples per phase complete; all29364224 external BF16
cells and the prior diagnostic artifact manifests match. The observer adds
synchronizations, so its1006.2578ms outer component wall is diagnostic and does
not replace the q8192 performance result. See
`benchmarks/correctness/attention-completed-phases-20260913.json`.

CK `9a7eaf4` adds an optional 8 MiB transposed V view for compacted exact PV.
The arithmetic and admission bounds remain unchanged; the view refreshes
under the existing workspace lease for every layer/call and is released with
the provider. Fifty-eight native cases compare 8,650,752 output cells with
zero raw-bit differences and preserve inputs, redzones and candidate ownership.
Both original q7169 modes match all 29,364,224 external BF16 cells. Completed
query wall changes from 993.2327 to 965.8722 ms, including 0.1259 ms for the
new transpose but excluding allocation, upload, safety checks and common K
transpose. An instrumented run shows less exact-PV time and more probability
time; the cause of the latter is not established.

Both same-DLL q8192 runs pass all 512 original GB10 outputs and actual
callbacks with exact first token 144/logit 10.375. Callback TTFT changes from
50888.006 to 49691.0931 ms, an observed 1196.9129 ms reduction in one paired
measurement. Candidate TPOT is 101.291451 ms and load is 20017.5312 ms.
Enable `QRT_CK_SM121_COMPACT_PV_TRANSPOSE_VALUE=1` for subsequent experiments
with 128 queries; the shipped default remains disabled. Decode and key
extents beyond 8192 retain the original layout. No repeatability or release
qualification is claimed. Evidence:
`benchmarks/correctness/pv-transposed-view-20260913.json`.

The existing lossless exp2 encoding now passes the full q8192/out512 gate
with that stack: all 512 GB10 tokens and actual callbacks match, with exact
first token 144/logit 10.375. Callback TTFT is 49519.006 ms. The 172.0871 ms
difference from the comparable preceding run does not establish repeatable
acceleration. Retain the format for its exact table storage reduction from
183,174,448 to 38,909,480 bytes. The same decoder was checked over all
328,728,576 covered values and 100,000 domain inputs; the current q7169 replay
again matches all 29,364,224 external BF16 cells. Only CK binary/table paths
change in the resolved product environment. This remains an experimental
build option; package savings, broader contexts and release gates are pending.
See `benchmarks/correctness/compact-exp2-stack-20260913.json`.

Four-lane prepared QK (`f2b21b1`) preserves all 32,160,016 native FP32 scores
and 640 independent CPU dot comparisons, including partial/causal tiles,
raw fallback and input/redzone checks. Both same-executable q7169 modes
match all 29,364,224 external BF16 cells. However, completed query wall with
V preparation increases from 906.5995 to 1090.3751 ms. Keep one lane; this
experiment is not integrated into the provider or selected for product use.
No q8192 result is attributed to it. See
`benchmarks/correctness/subgroup-tiled-qk-20260913.json`.

An instrumented q8192 run with whole584588a and CK/FLA/MoE8f436db, before
wider query slabs and the V transpose, passes the same full GB10 boundary.
Completed host clocks show 21470.5 ms for the attention pipeline,
11298.44 ms for linear core including output projection, 10792.946 ms for MoE
and 7340.268 ms for linear projection. Its 53787.850699 ms callback includes
profiling and does not replace the uninstrumented result.

The profile also corrects earlier MoE timing labels. The prior 16546.228 and
13785.066 ms values came from synchronized GPU events; the corresponding host
clocks are 16764.390 and 13997.512 ms. Original callback comparisons and GB10
acceptance remain valid. Canonical historical records are preserved, with the
field correction and current signed event anomalies documented in
`benchmarks/correctness/hipblaslt-producer-wall-profile-20260913.json`.

Parallelizing probability generation across eight K32 waves preserves the
original sequential denominator recurrence and all 216 native numerical
cases. Both full q7169 comparisons match every external BF16 cell, but completed
host operator time changes only 1069.18 to 1063.79 ms. No speedup is retained
and no full-model run selects this mode. Subsequent experiments retain compact
PV mode 1. See `benchmarks/correctness/attention-parallel-probability-20260913.json`.

Removing the three fixed routed midpoint bands fails 384 of the 512 original
continuation tokens, despite an exact first token and logit. The absolute-only
selector experiment is rejected and its timing is excluded from performance
acceptance. Further experiments retain all three fixed radii at 512, alongside
the original norm-scaled bound. Full failed tokens, callbacks and command
provenance are in `benchmarks/correctness/moe-absolute-selector-20260913.json`.

The chunked route bounds activation carriers but shows no speedup and remains
opt-in. The 10000ppb correction bound is an empirically qualified admission
setting for the declared cases, with unchanged exact-dot arithmetic and
GB10 tolerance. The targets remain q8192 TTFT <= 4187.415605 ms,
TPOT <= 35.502151 ms and model+engine load <= 30000 ms. The new builds still need
broader contexts and renewed package, HTTP and soak qualification. Commands,
component hashes, raw outputs, callback timing and external oracle bindings
are in `benchmarks/correctness/prefix32k-admission-product-20260913.json` and
`benchmarks/correctness/cold-prefill-chunks-20260913.json`.

## Historical published measurements

The following are historical native Windows results for Qwen3.6-35B-A3B BF16 on Ryzen AI
Max+ 395 (`gfx1151`), batch size 1. Startup is measured separately from TTFT.
Every retained product row was accepted only with its matching external BF16
correctness boundary. These recorded profiles and boundaries do not qualify
the current corrected runtime above.

## Product matrix

| Mode | Prompt shape | Total tokens | Load ms | TTFT ms | Prefill tok/s | TPOT ms | Decode tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| cold | q8192 | 8,192 | 19,940.245 | 3,852.909 | 2,126.186 | 32.732 | 30.551 |
| cold | q16384 | 16,384 | 19,948.525 | 8,027.368 | 2,041.018 | 32.457 | 30.810 |
| cold | q32768 | 32,768 | 19,909.139 | 17,274.313 | 1,896.921 | 33.935 | 29.468 |
| cold | q65536 | 65,536 | 19,924.188 | 41,381.599 | 1,583.699 | 39.778 | 25.139 |
| cold control | q130560 | 130,560 | 19,938.847 | 124,594.960 | 1,047.875 | 47.173 | 21.198 |
| cold | q131072 | 131,072 | 20,051.963 | 108,563.050 | 1,207.335 | 42.762 | 23.385 |
| prefix | 16,384 + 1,024 | 17,408 | 19,919.603 | 1,529.093 | 11,384.526 | 36.488 | 27.406 |
| prefix | 32,768 + 1,024 | 33,792 | 19,920.588 | 1,930.620 | 17,503.185 | 38.509 | 25.968 |
| prefix | 65,536 + 1,024 | 66,560 | 19,933.010 | 3,065.768 | 21,710.708 | 36.922 | 27.084 |
| prefix | 129,536 + 1,024 | 130,560 | 20,035.182 | 7,899.540 | 16,527.544 | 42.745 | 23.394 |
| prefix | 131,072 + 1,024 | 132,096 | 20,023.448 | 8,592.158 | 15,374.019 | 42.890 | 23.315 |
| prefix | 262,144 + 1,024 | 263,168 | 20,045.888 | 9,278.300 | 28,363.817 | 55.555 | 18.000 |

The last historical prefix row used a 263,680-token qualification service
limit. The public HTTP profile defaults to a stricter 262,144-token total
context; operators must not infer that the larger row changes the released
default contract.

## q8192 retained target

The confirmed retained target is 1,506.407 prefill tok/s and 4,187.416 ms TTFT.
The accepted row reached 2,126.186 tok/s and 3,852.909 ms. Model plus engine
load remained below the 30-second product bound.

## q8192-neighbor continuity gate

The v1.0.1 repair replaces fixed-q8192-only CK-FMHA and fused-GDN calls with
dynamic q8191/q8193 entries while leaving the retained q8192 entry points
unchanged. The isolated gfx1151 provider smoke produced:

| Provider | q8191 | q8192 fixed | q8193 | Neighbor/fixed ratio | Numerical gate |
|---|---:|---:|---:|---:|---|
| CK-FMHA | 19.888 ms | 19.983 ms | 20.067 ms | 0.995 / 1.004 | no value above `1e-5`; no nonfinite |
| fused GDN, decay | 8.546 ms | 8.773 ms | 8.709 ms | 0.974 / 0.993 | max output error `7.63e-6`; async exact |
| fused GDN, log-g | 8.731 ms | 8.727 ms | 8.703 ms | 1.000 / 0.997 | max output error `7.63e-6`; async exact |

The q8191 CK comparison had 267 bit-level differences at the final compared
token, with maximum absolute error `2.98e-8`; none exceeded the declared
`1e-5` component tolerance. q8193's q8192 prefix was bitwise exact. The packed,
fixed q8192, and q262144 tile-regression checks were also exact. These numbers
remain synthetic component evidence rather than product inference evidence.

The native Windows real-model publication gate passed on `baiying` from a
clean build at source commit `09bd96fd2d85a0715f1501d16fb6391ce199d0f1`.
It used the model at
`D:\models\Qwen3.6-35B-A3B`, the public GB10 oracle, `max_tokens=1` and `2`,
and three cold-prefix repetitions per shape. All 18 returned token sequences
matched GB10 exactly.

| Prompt | max_tokens=1 median TTFT | max_tokens=2 median TTFT | All-six range |
|---|---:|---:|---:|
| q8191 | 3,948.904 ms | 4,027.895 ms | 3,911.132–4,046.155 ms |
| q8192 | 3,878.767 ms | 3,885.607 ms | 3,858.763–3,903.565 ms |
| q8193 | 3,904.950 ms | 3,892.132 ms | 3,846.939–3,905.346 ms |

The worst neighbor/q8192 median ratio is now `1.036619`; the worst positive
residual is `142.288 ms`. Both pass the tightened `1.10x` and `500 ms`
limits, and every q8192 sample beats the retained `4,187.416 ms` TTFT target.
The former q8193 5.5-second fallback is no longer present.

## Wide prompt-length continuity gate

The neighbor check above is supplemented by a 72-request cold-prefill sweep
covering eight independent length boundaries from q4096 through q16384. Each
boundary tests `q-1`, `q`, and `q+1` three times with a unique leading token
to prevent prefix-cache reuse. Every AMD-generated token ID matched the GB10
BF16 authority.

| Center | q-1 median TTFT | q median TTFT | q+1 median TTFT | Local max/min |
|---:|---:|---:|---:|---:|
| 4,096 | 2,265.420 ms | 2,193.201 ms | 2,187.632 ms | 1.035558 |
| 6,144 | 3,310.968 ms | 3,254.327 ms | 3,384.861 ms | 1.040111 |
| 8,192 | 3,956.523 ms | 3,889.966 ms | 3,890.921 ms | 1.017110 |
| 9,216 | 4,614.217 ms | 4,601.238 ms | 4,620.542 ms | 1.004196 |
| 10,240 | 5,134.645 ms | 5,120.694 ms | 5,143.162 ms | 1.004388 |
| 12,288 | 6,037.402 ms | 6,023.924 ms | 6,028.683 ms | 1.002237 |
| 14,336 | 7,504.772 ms | 7,396.619 ms | 7,675.859 ms | 1.037752 |
| 16,384 | 8,456.237 ms | 8,047.645 ms | 8,089.251 ms | 1.050772 |

All eight local gates remain below `1.10x` and `500 ms`; the worst observed
values are `1.050772x` and `408.592 ms`. Median throughput ranges from
`1,807.612` to `2,105.931 tok/s` across all 24 cohorts, a global max/min ratio
of `1.165035`. The split-tail path now overlaps its two independent
transactions; this raises the q6144 neighborhood without changing the retained
direct-q8192 route. The result distinguishes normal shape-efficiency variation
from a length-rounding cliff: no tested exact boundary or adjacent non-boundary
length takes a separate multi-second fallback.

The machine-readable q8192 product record, wide continuity record, provider
smoke, verifiers, and GB10 oracle are under `benchmarks/` and `scripts/`.

## Correctness attachment

At q8192 the external BF16 authority and native runtime selected token 16. The
authority first-token logit was 24.750 and native was 24.875, an absolute
difference of 0.125 (the declared tolerance boundary).

Long continuation checks generated 512/512 matching tokens:

| Case | Input digest | Output digest | Match |
|---|---|---|---:|
| cold q131072 | `ffef18d340fb4fe8` | `4813300b562d057b` | 512 / 512 |
| prefix q131072 + 1024 | `9a681295dde809d4` | `fd69ca7327c912bf` | 512 / 512 |
| prefix q262144 + 1024 | `6aa7d9579bb8427c` | `7a8c5d8ec398878b` | 512 / 512 |

Digests are compact publication identifiers, not standalone correctness
authority. Acceptance was token-for-token against the external BF16 service.

## Interpretation

TTFT excludes model/engine startup but includes real prompt prefill and the
first generated token. Prefix throughput divides the full effective prompt by
the measured reused-prefix request time and should not be compared as cold
compute throughput. Results are specific to the stated model, hardware,
toolchain, runtime profile, and batch size.

Machine-readable rows and hashes are under `benchmarks/performance/`.
