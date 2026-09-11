# Correctness and evaluation

## Correctness authority

The numerical authority is a separate BF16 Qwen3.6-35B-A3B service. Product
acceptance binds real prompt token IDs, the first generated token, and the
first-token logit within 0.125. Decode and prefix continuation are compared
token-for-token. Engine self-hashes are diagnostic only.

The unreleased arbitrary-length gate additionally freezes the failing q7169
fixture in `contracts/arbitrary_q7169_gb10_oracle.json`: prompt hashes, first
token 82, raw logit 9.25 at the unchanged 0.125 tolerance, and all 32 reference
tokens. This is captured GB10 authority, **not** Windows acceptance. The default
q8192 contract and its targets are unchanged. `scripts/capture_gb10_q7169_oracle.py`
is the byte-identical fixture adapter used for the September 9 capture; stage it
beside the two existing q8192 capture modules in the opt-in BF16 reference
container. It changes fixture constants only and does not alter model math.

The published OpenAI acceptance additionally submits deterministic raw-token
prompts across the continuous-length matrix and requires every returned text
and finish reason to match a frozen authority response. It separately tests
the exact maximum accepted input and context-limit rejection.

The production `engine/runtime.env` scopes BF16-window high-ID arbitration to
the two continuous maximum-context provider shapes that require it. The formal
MMLU run explicitly enabled the same policy globally, as recorded in the
summary (`max_ulps=2`, eligible count 2, count-three shape mask 12) and in
`benchmarks/eval/mmlu-pro-runtime-overrides-v1.0.0.env`. Keeping global
evaluation arbitration explicit prevents a score-calibration tie break from
changing ordinary OpenAI completions; production HTTP acceptance always runs
with the global override disabled.

## Unreleased correctness diagnostics (updated September 12)

Whole and CLI `77877edb8cb757149fd6cfc2343bea387119a082`, built with
`-HawkeyeReplayLanes 4`, MoE
`5710b891787bde7c5ed641a76619b374ca8911d8` and FLA
`13d77b63ea705ea8f4d4daee8d2ac30a581b7bfb` with
`QRT_FLA_GDN_COOPERATIVE_EXACT=1`, plus CK `49a660b`, pass both complete
512-token cold continuations and all six 32-token controls on `baiying`,
using real model `D:\models\Qwen3.6-35B-A3B`. Every frozen prompt,
first-token/logit, complete output and streaming boundary passes in these
eight cases: 1,216 generated tokens. The ordinary q8193 request now passes
through the bounded cold prefill split. All eight
complete with successful host checks. The requests use the strict prefill arithmetic
profile, with the candidate bound derived from actual prompt length. The
[complete product records](../benchmarks/correctness/bounded-prefill-product-20260912.json)
bind source, clean Windows whole/CLI builds, component hashes, commands,
all outputs and the unchanged GB10 oracle. Performance remains unqualified.

| Complete frozen request | First token / logit | Load ms | Actual callback TTFT ms | TPOT ms |
|---|---|---:|---:|---:|
| q7169 out512 | 82 / 9.25 | 20,012.6960 | 54,119.3108 | 112.901665 |
| q8192 out512 | 144 / 10.375 | 20,025.2229 | 62,837.3352 | 115.219774 |
| q8191 out32 | 168589 / 11.375 | 20,000.5564 | 63,083.9095 | 119.652074 |
| q8193 out32 | 220 / 9.75 | 20,237.8517 | 63,115.6430 | 109.239029 |

The dense absolute-error selector previously missed the closer BF16 rounding
boundary below a power of two. It now uses the shared nearest-boundary helper.
All 282 signed power-of-two host regressions change from missed to selected;
four native correction/guard cases pass, including the new positive and
negative boundary case. Both 512-output requests and q8191 retain every
captured GB10 operator, forty-layer carrier and KV comparison, plus 554 / 806
state boundaries. The other passing controls are q7168, q7169, q7170 and q8192.

Before the split, whole `9b6f145` emits `64,57,82` at q8193 instead of
`220,220,196`; its remaining 29 tokens agree.
Its first logit 9.75 is within tolerance, which cannot substitute for the
required token match. The run takes 45,579.7602 ms to its first callback.
It records zero exact-attention calls, versus twenty at q8192: the CK
provider's selection and scratch allocation stop at 8192 although the exact
kernel supports 16384. The next repair binds dispatch and all scratch/launch
extents to that kernel capacity. CK `49a660b` builds successfully on Windows
in 37,030.338 ms and now executes nine full-prefix and ten terminal exact
calls at q8193. Its [product result](../benchmarks/correctness/attention-extent-product-20260912.json)
still has the identical three token errors, with 63,529.5056 ms callback TTFT.
The dispatch omission is fixed, but another numerical difference remains.
The GB10 capture processes this request as 8192 tokens followed by one token.
The [native saved-prefix diagnostic](../benchmarks/correctness/prefill-split-product-20260912.json)
matches all 32 outputs and the first logit 9.75, including two repeated hits,
rollback and an unrelated-prefix rejection before provider invocation. Three
suffix executions consume the actual prompt token 63 at position 8192. Internal
layer-0 QKV/Z and downstream carriers differ from the original prefill
transaction; these diagnostics do not reject a GB10-valid output. The initial
seed/fallback takes 66,900.9132 ms, including its unstreamed 32-token continuation.
Warm hit callbacks of 172.6169 / 161.4405 ms are **not cold TTFT**.

The ordinary core now executes a cold 8192 prefix followed by the bounded
1..1024-token suffix when the existing resident tail can hold the request.
The intermediate seed is not published. Both callback and report clocks include
the cold seed. A tagged optional extension in the existing prefix result binds
the first logit to output zero before later decode/rollback; old provider results
leave it unavailable. Local tests cover clocks, cancellation, failures, suffix
and output limits, hidden seed and stale-logit rejection. Windows whole and
CLI builds pass in 83,063.055 / 11,826.273 ms. All eight ordinary frozen
requests pass; both 512-output state traces and q8191 retain their captured
GB10 operator/carrier/KV comparisons. The q8193 cold report includes the seed
and records 63,115.4979 ms provider TTFT beside the actual 63,115.6430 ms
callback. Arbitrary new prompts, general partial checkpoints, long contexts,
packaged HTTP, performance and release qualification remain open.

The [FP32 recurrent checkpoint producer](../benchmarks/correctness/fla-fp32-checkpoints-20260912.json)
at `872a042` builds on `baiying` in 37481.343 ms. Its optional FLA export
saves up to three unrounded states on existing 64-token boundaries. Real
q7169 snapshots at 64, 1024 and 7168 each match all 524,288 FP32 cells of
separately executed prefixes from the qualified old exact provider. All
29,364,224 full outputs, final state, prefix outputs, inputs and guards
remain exact. q64 compatibility and q65 checkpoint/tail controls pass.
Local C/ABI, Rust, Clippy, q16, 312 Python tests (two skips) and hygiene
pass. This is the recurrent-state producer only: complete convolution,
KV and hidden capture, atomic publication, restoration and model-level
prefix/GB10 acceptance remain open.

The [integer operand prepacking trial](../benchmarks/correctness/attention-integer-prepack-20260912.json)
at `31f3e7f` preserves all 29,364,224 captured q7169 attention BF16 values
and passes input/encoding/redzone checks. Its 64,253,536-byte workspace moves
KV preparation outside repeated matrix tiles, but total GPU time remains
3433.81 ms versus the selected layout's 1284.12 ms. The included KV preparation
takes 0.6574 ms. Keep this replay mode outside the product path; the next
attention experiment changes the exact product reduction mapping.

The [strided pair attention trial](../benchmarks/correctness/attention-strided-pair-20260912.json)
at `681167c` splits each K16 dot across two XOR-16 lanes while retaining
coalesced key/value reads. All 29,364,224 full q7169 BF16 outputs match in
three combinations, and the sparse tail and buffer/input guards pass. QK
alone takes 765.893 ms versus 578.045 ms; PV alone takes 1017.52 ms versus
705.237 ms. The combined route costs 1769.12 ms versus 1283.37 ms. None is
enabled. Local C/ABI, Rust, Clippy, q16, 311 Python tests (two skips) and
hygiene pass; the native replay build takes 8153.577 ms.

The preceding [cooperative FLA configuration](../benchmarks/correctness/fla-cooperative-product-20260912.json)
reduces actual callback TTFT by 5819.2376 / 5071.3091 /
6058.9563 ms for q8192 / q7169 / q8191 against the [preceding dense-replay
configuration](../benchmarks/correctness/dense-subgroup-product-20260912.json).
Its q8192 W/U, persistent-state and output intervals total 5290.60454 ms.
All recorded GB10 operators, forty-layer carriers, KV and 554 / 806 state
boundaries remain exact. Keep cooperative FLA, dense replay at four lanes,
and routed MoE at sixteen lanes. The immutable q8192 performance gate and
release acceptance remain open.

The explicit `QRT_FLA_GDN_BATCHED_EXACT=1` route batches independent
KKT/WU/output chunks and retains four recurrent value rows per CTA through
each bounded 1024-token segment. The original K16 arithmetic, BF16
checkpoints, FP32 carry, logical tails and final FMA are preserved. Native
q64/q65 and real GB10 q7169 component inputs match every output and final
state bit against the old exact provider, with exact synchronous/asynchronous
parity. The new FLA build passes in 32,433.351 ms. Both complete 512-token
requests retain all captured GB10 operators, carriers, KV and state-boundary
comparisons. Before routed compaction, its [full product controls](../benchmarks/correctness/fla-batched-exact-product-20260911.json)
improve q8192 by 1357.4566 ms against the [prior strict records](../benchmarks/correctness/q1-embedding-norm-product-20260911.json);
the gain is insufficient for performance
acceptance.

The [embedding-norm origin](../benchmarks/correctness/q1-embedding-norm-origin-20260911.json)
locates the remaining long-decode error at positions 7277 and 8426.
Incoming layer-0 recurrent state and convolution history are exact, but
one BF16 input-normalization value differs in each request. The original
CUDA one-row normalization reproduces the old native result; the actual
original two-row transaction and its FP32 variance/inverse reproduce the
repair. The native implementation now uses the original eight-warp,
eight-adjacent-values reduction and existing SM121 inverse correction for
both the normal embedding entry and device-top1 prefetch. The first attempt
covered only the normal entry; 502 prefetched steps bypassed it, leaving
both complete output sequences unchanged. That failed integration is
retained in the evidence. No token, position or expected-output special
case is introduced.

After repairing both entries, all captured operators and all forty carriers
at both origins are exact. All 3,726,336 layer-19 K values and V values
through position 7277, and all 4,314,624 through position 8426, match GB10.
The subsequent 806 / 554 observed layer-0 state boundaries are exact;
these hashes diagnose the repair while the complete token/logit results
provide acceptance. q8191's forty carriers and full-attention operators at
8196 remain exact. GB10 captures r25/r26/r27 each pass all eight unchanged
frozen cases. Local C/ABI, Rust, Clippy, q16, 297 Python tests (two skips)
and public hygiene pass; the Windows build passes in 80,649.669 ms.
The actual kernel is also exercised with resident BF16, host F32 and
indirect device-top1 inputs against both real norm fixtures.

The prior [recurrent output reduction repair](../benchmarks/correctness/q1-recurrent-output-order-20260911.json),
[K head normalization repair](../benchmarks/correctness/q1-key-normalization-product-20260911.json),
[final normalization](../benchmarks/correctness/q1-final-norm-product-20260911.json)
and [short gated normalization](../benchmarks/correctness/q1-continuation-origin-20260911.json)
remain active. Before the embedding repair, daa054d first differed at
q7169 output index 148 and q8192 index 255; both now pass all 512 outputs.

These strict timings remain above the immutable performance limits.
The [prefill arithmetic comparison](../benchmarks/correctness/prefill-arithmetic-comparison-20260911.json)
rejects both attempted matrix/backend substitutions on actual token errors.
The old fast AITER profile takes 4204.9695 ms for q8192 but differs at output
index 1 (244 instead of 255); q7169 takes 3958.422 ms and emits first token
220 instead of 82. Keeping original FLA/scalar endpoints while replacing
prefill midpoint corrections with hardware matrix outputs passes q8192's
32-token control in 46864.6811 ms. Its complete q8192 continuation first
differs at index 115 (248046 instead of 271), and q7169 again fails its
first token. All five requests complete with successful host and streaming
checks. None supplies a new retained performance result. The arithmetic-preserving
FLA schedule above is retained as a qualified correctness route. The
[fused CK prefix experiment](../benchmarks/correctness/prefill-ck-fused-product-20260911.json)
takes 59620.999 ms for q8192 but differs at output index 8 (82 instead of
220); q7169 takes 53133.6612 ms and again emits first token 220 instead of
82. Both complete with successful host and streaming checks. At q8192's
first decode step, layers 0–3 and all captured layer-0 operators/state remain
exact; the first differing completed carrier is layer 4. This attention
substitution also remains unqualified.

The [packed projection experiment](../benchmarks/correctness/hawkeye-packed-product-20260911.json)
uses whole `25d34e7559691948c59b8e85704882bbeb7894fa` with the same
batched FLA and strict arithmetic profile. Independent packed K16 lanes and
transposed weights retain all three complete frozen outputs and captured
GB10 boundaries. q8192 callback TTFT nevertheless rises to 84055.422 ms;
q7169 takes 72740.7119 ms and q8191 takes 84446.9134 ms. The same
350,029,910 q8192 candidate dots take 19850.035 ms versus 15016.761 ms
in the prior wave16 correction sequences. The new Windows build and all
298 local Python tests (two skips), C/ABI, Rust, Clippy, q16 and hygiene
checks pass. Keep the wave16 profile selected. These correction totals are
host wall time around completed submissions, including waits.

The [shared-memory WMMA experiment](../benchmarks/correctness/wmma-staging-product-20260911.json)
at whole `d59e049e5bc67a3e392faf298c851a89d8e6d3c6` preserves every
K16 update and shares a K64 operand slab across eight waves. All three
complete frozen requests, seventy staged projection calls per request,
captured GB10 boundaries, streaming and host checks pass. A separate
native component compares every FP32 output, including 67,108,864 cells
at q8192, with zero differences against the original kernel. Its large
matrix interval improves from 72.7421 to 51.0145 ms, but actual product
callback TTFT is 79974.2864 / 71029.3772 / 80540.2278 ms for q8192 out512 /
q7169 out512 / q8191 out32. All exceed the prior strict baseline in this
cohort. Local checks pass 299 Python tests (two skips) and the complete
C/ABI, Rust, Clippy, q16 and hygiene suite. Both Windows builds pass, as
does the native three-shape comparison. Keep global WMMA operands
selected; use the existing MoE subphase events to resolve overlapping
upstream waits before the next structural replacement.

The first [MoE event diagnostic](../benchmarks/correctness/moe-profile-event-failure-20260911.json)
streams first token 144 but fails at the first decode step with an invalid
event handle. All forty prefill intervals completed: MoE totals 18299.103 ms,
including 11344.459 ms in routed gate/up and 4572.362 ms in down; shared
work overlaps routed work. These incomplete-request timings have no
continuation or performance acceptance. The exact SM121 router skipped the
input-end and projection-end events consumed by the legacy profile flag.
The fix records both boundaries on the router stream, retains event-error
handling, and also consumes them when SM121 is selected independently of
the legacy flag. All 300 local Python tests (two skips), C/ABI, Rust, Clippy,
q16 and hygiene pass. The actual branch, event helper and callback are
executed with a host HIP recorder across enabled/disabled profiling,
cached/uncached control, early/terminal layers and launch/event faults.
Whole `649f595fc1183ab0bc212e23d4f253e7e3e63545` then passes the
Windows build (81730.553 ms) and the [complete profiled q8192 request](../benchmarks/correctness/moe-subphases-correct-continuation-20260911.json):
all 512 frozen tokens, first logit 10.375, prompt, streaming and host checks
pass. Captured operators, all forty carriers, full KV and all 554 later
state boundaries remain exact. The forty MoE event spans total 18185.966 ms;
gate/up takes 11260.171 ms and down 4571.755 ms. Profiled callback TTFT is
82523.5147 ms, load 20300.1233 ms and TPOT 116.241378 ms. These diagnostic
timings do not replace the unprofiled strict result.

The retained optional route, `QRT_QWEN36_MOE_COMPACT_ROUTED_HAWKEYE=1`,
compacts routed candidate indices across 262,144-cell windows before
replaying the original wave16 exact dots. It uses 1,048,580 bytes of scratch,
with no host counter read or synchronization. Gate, up and down share the
retained selectors; up finalizes after all window corrections. The original
local schedule remains available in the same binary. All 301 local Python
tests (two skips), C/ABI, Rust, Clippy, q16 and hygiene pass. The threaded
execution of actual selectors and scheduling covers dense/sparse/empty
windows, partial tails, debug outputs, redzones and submission faults under
ASan/UBSan. The Windows MoE build passes in 17944.130 ms; the component
build passes in 5714.362 ms and all eight native controls complete in
827.025 ms. All four new routed comparisons have zero raw-bit differences,
intact redzones and unchanged inputs on a nonblocking stream. The three
complete product requests and captured GB10 boundaries pass, as summarized
above. Component sparse timing improves from 7.594600 to 6.139460 ms;
dense timing is nearly unchanged and is not a product estimate.

The next attention experiment is `QRT_CK_SM121_NATIVE_PRODUCTS=1`,
restricted to the existing transposed split QK/PV path. Normal BF16 products
are represented exactly with FP32 multiplication; exponent edits and signed
integer truncation preserve the original K16/26-bit alignment. Subnormal,
exceptional and extreme-exponent products use the original integer path.
The original ordered normalization and all softmax/PV boundaries remain.
All 200,000 host K16 groups and 3,200,000 products match the original integer
representation and aligned sum, including every BF16 operand encoding and
random carries. All 302 Python tests (two skips), C/ABI, Rust, Clippy, q16
and hygiene pass. The [native captured-Q/K/V comparison](../benchmarks/correctness/attention-native-products-20260911.json)
at source `f5faceed881ec08af3694355b9fedcb07641e4c3` matches all
29,364,224 reference BF16 outputs and all original native FP32 bits. The
Windows build and both component requests pass with healthy host checks.
The component interval nevertheless increases from 1367.15 to 2352.06 ms
(QK 573.904 to 848.447 ms; PV 791.727 to 1502.4 ms). Keep integer-packed
products selected; the optional native-product path is a numerical diagnostic.
Its slower component does not warrant a product performance run.
The [four-BF16-partial matrix experiment](../benchmarks/correctness/attention-mantissa-wmma-failure-20260911.json)
at `98cd10fddba52795dd50c3ab5dad67577cdae71a` fails a captured real
65-query boundary: 70 of 266,240 BF16 outputs differ. Both the original
path and the existing split-softmax control match every native FP32 bit.
A separate native matrix probe at `f18f39b` finds 6,835 partial mismatches
among 11,962 eligible cells; its host and device compensation agree, but
3,864 compensated sums differ from the original integer reference. The
FP32 representability proof does not guarantee these native WMMA partials.
The next opt-in `QRT_CK_SM121_MANTISSA_WMMA=1` route instead represents
normal BF16 rows as signed 16-bit integers and combines four signed/unsigned
byte matrix products without floating-point accumulation. Rows spanning
more than seven exponents fall back to the original integer path. It keeps
ordered softmax/PV rescaling and bounded owned workspace. All 304 local
Python tests (two skips), C/ABI, Rust, Clippy, q16 and hygiene pass. Native
integer [matrix and captured-input checks](../benchmarks/correctness/attention-integer-wmma-20260911.json)
at `a9e46b7` now pass: all 29,364,224 BF16 outputs and native FP32 bits
match, as does the 65-query tail. The component is slower, 4663.77 versus
1363.4 ms, so it remains a diagnostic. Compiled kernels use 148/185 VGPRs
for QK/PV without register spills; the next experiment shares preparation
across eight matrix tiles. The prepared-row implementation shares both
original operands and packed byte fragments, retains the numerical fallback,
and passes 304 local checks plus the native arithmetic and full q7169
[shared-preparation replay](../benchmarks/correctness/attention-integer-wmma-tiled-20260911.json).
The component remains slower: 4467.32 versus 1367.34 ms. Keep the selected
packed scalar route. [Earlier fast CK output analysis](../benchmarks/correctness/attention-fast-error-distribution-20260911.json)
finds 65,059 BF16 differences across 33,681 query/head rows. A 256-ppm
head-peak midpoint window selects 20,654,619 outputs yet misses 67 differences.
These windows are diagnostics, not correctness thresholds. Replay
layouts 6/7 preserve the reference K32 online-softmax ordering and isolate
native PV with either exact or native QK. The [full q7169 native matrix
replays](../benchmarks/correctness/attention-native-mma-20260912.json)
at `809c0ca5f6e602390c7c25f743404aa76e8d557c` still differ at 29,491 /
64,959 BF16 cells. Batch 32 reduces their component times to 967.4 / 500.782 ms,
with the same numerical differences as batch 8. Reference online-softmax
ordering alone does not repair native matrix rounding. Neither layout has a
qualified product result.

The optional CK build switch `-Sm121InterpolatedExp2` selects a lossless
re-encoding of the original SM121 exp2 table: 183,174,448 to 38,909,480 bytes.
An integer chord plus packed residual reconstructs every original FP32 bit.
The [exhaustive repack record](../benchmarks/correctness/exp2-interpolated-20260912.json)
checks all 328,728,576 table cells and 100,000 random argument bit patterns,
with zero differences. Each backend validates its own layout and SHA before
upload, and the default CK build keeps the original format. [Native q7169
replays](../benchmarks/correctness/exp2-interpolated-native-20260912.json)
at `2b4e00f` match all 29,364,224 GB10 BF16 outputs and the original native
FP32 bytes for both formats and query batches 8/32. Batch 8 changes from
1368.57 to 1344.57 ms, batch 32 from 1283.23 to 1240.41 ms. This establishes
exact native reconstruction and reduced table storage, but is insufficient
to address the dominant product wall; no compact-table product is qualified.

The optional MoE build parameter `-RoutedReplayLanes 4` or `8` groups
multiple BF16 products per lane while preserving every K16 carry and the
original candidate selectors. It changes only compacted routed replay; the
local control and default build retain 16 lanes. This exposes more independent
dots per wave and reduces subgroup reductions. [Native arithmetic and routed
controls](../benchmarks/correctness/moe-subgroup-20260912.json) at `254f0a1`
pass, including signed/subnormal values, unaligned rows, tails and immutable
inputs. The 4-lane complete q8192/512 request matches every frozen token and
first logit, streaming and captured GB10 state/KV boundaries, but callback TTFT
is 87643.8764 ms versus the selected 77000.9007 ms. Keep the selected product
components. A correctness-attached profile locates the added time before
gate, in an interval that also contains input/weight L2 work. Those kernels
and native route-layout machine code are unchanged. A current compacted
16-lane control also concentrates time in that nominal interval, so its label
alone does not establish exact replay cost. It nevertheless confirms the
product regression: the fully correct profiled 16-/4-lane requests take
79956.5816 / 89658.8676 ms actual callback TTFT, and total MoE intervals are
15592.472139 / 27955.314637 ms. Retain 16-lane MoE for the product.

The same proven subgroup helper is now optional for dense prefill projection
correction via `-HawkeyeReplayLanes 4` or `8`. Its host launcher derives both
candidate capacity and grid geometry from the selected subgroup size. All
three sizes pass index transport, partial windows, dense selection and failure
cleanup tests. The Windows whole build passes in 83534.787 ms, native safety
build in 83697.142 ms, and all three correction cases in 680.18 ms with zero
BF16 mismatches and intact redzones. The complete frozen products above
qualify the four-lane dense configuration as an improvement. All 305 Python
tests (two skips), C/ABI, Rust, Clippy, q16 and hygiene pass.
The optional `QRT_FLA_GDN_COOPERATIVE_EXACT=1` schedule applies the same
four-lane K16 dot to batched W/U, output scores, output values and recurrent
state. Shared K-contiguous tiles reuse V/checkpoint/key operands; each state
CTA still owns four complete value rows through a bounded segment. Original
rounding, chunk checkpoints, final FMA and U=V ownership remain unchanged.
Threaded execution of the actual kernel bodies passes tail, alias, checkpoint
and nonzero-state transport under ASan/UBSan. The clean Windows build passes
in 36496.782 ms. Native q64/q65 and real q7169 replay match every output and
final state bit, with exact synchronous/asynchronous parity. All three
complete frozen products and captured GB10 boundaries pass as reported above.
All 306 Python tests (two skips), C/ABI, Rust, Clippy, q16 and hygiene pass.

Attention replay layout 8 applies four-lane exact dots to QK and stages K32 V
slabs for cooperative PV. It preserves the original online probabilities,
alpha rescale, K16 finish points and reciprocal. Scratch and submission-failure
checks cover the new layout. The [native captured-input replay](../benchmarks/correctness/attention-cooperative-20260912.json)
matches all 29,364,224 BF16 cells and the qualified native FP32 output at both
eight and thirty-two queries per batch, but takes 2229.53 / 2076.33 ms versus
1363.33 / 1286.46 ms for layout 4. The 65-query offset tail also matches every
BF16 cell; its negative PV stage interval is invalid timing evidence. This
schedule is rejected for product performance, and layout 4 stays selected.

The integer WMMA diagnostic now tries a row-exponent bound before loading the
sixteen original operand pairs for each cell. Four exact integer matrix
partials suffice when alignment discards no product or carry bits. Otherwise
the original compensated or packed K16 path runs. All 400,000 host groups
match the independent wide canonical accumulator, including 46,408 accepted
groups whose conservative bound differs from the actual paired maximum,
44,292 with nonzero carry and 12,555 beyond the signed-32-bit magnitude bound.
The [native range replay](../benchmarks/correctness/attention-integer-range-20260912.json)
passes all 16,384 generated cells and every captured q7169 BF16/native FP32
output, but takes 3191.52 versus 1288.02 ms for layout 4. It stays diagnostic.
Sampled real QK groups show only 65,911 of 262,144 eligible for the initial
bound. The next version uses the greatest common binary unit of each row and
four packed words of trailing-bit counts. Their bytewise sum proves product
divisibility without sixteen pair loads. The sampled eligible count rises
to 176,056, before accounting for actual GPU execution cost. Both original
and expanded branches pass 400,000 independent host groups each; the expanded
branch also checks the compensated fallback and 100,000 packed predicates.
The [expanded native branch](../benchmarks/correctness/attention-integer-units-20260912.json)
again matches every generated cell and captured BF16/native FP32 output, but
takes 3439.59 versus 1282.81 ms for layout 4. Better arithmetic eligibility
does not improve this GPU schedule; it stays diagnostic.

The [RDNA3 ISA guide](https://docs.amd.com/api/khub/documents/UkT_UPQL21KfKAMUBFnZTw/content)
specifies round-to-nearest-even for floating WMMA, so changing the general
FP32 rounding mode is not a supported way to obtain the reference arithmetic.
The native mantissa probe can instead test FP16 matrix instructions with
exactly representable four-bit parts, separately from the failing BF16
partial controls. This is an arithmetic experiment, not a model precision
change or an accepted provider. The [native FP16 control](../benchmarks/correctness/attention-fp16-parts-20260912.json)
has zero conversion mismatches but repeats the BF16 result: 6,835 differing
partials and 3,864 differing compensated sums among 11,962 eligible cells.
Changing the input encoding does not solve the matrix arithmetic boundary.

The next diagnostic uses packed unsigned sixteen-bit multiplication to
construct two original BF16 products at once. Packed integer masks handle
both exponents, zero/subnormal significands and signs; the K16 alignment and
normalizer stay unchanged. All 65,536 input encodings against sixteen
independent controls and randomized companion halfwords pass 2,097,152 host
product comparisons. `QRT_SM121_PAIRED_PRODUCTS=1` enables it in transposed
QK, serial PV and four/eight-lane subgroup dots; the default remains off.
The [native paired-product checks](../benchmarks/correctness/attention-paired-products-20260912.json)
confirm the actual `v_pk_mul_lo_u16` instruction, 2,097,166 exact products,
intact guards, immutable inputs and exact four/eight/sixteen-lane dots at
K16/K512/K2048. Every captured q7169 BF16/native FP32 output matches.
Attention takes 1410.7 / 1518.95 ms at batches 32 / 8 versus 1279.37 /
1364.54 ms for the original. QK improves modestly but PV regresses, without
register spills. Keep the original product arithmetic.

Dense projection correction had a separate selector omission: it checked
only the midpoint inside the current BF16 binade. Immediately around a
power of two, the neighboring lower midpoint is closer. Dense correction
and its diagnostic now use the same nearest-boundary predicate as routed
MoE, retaining the existing error coefficient and exact correction math.
The actual dense predicate passes 282 positive/negative boundary cases with
both absolute-sum and L2 addressing; the common helper passes 130,552 finite
cell-boundary checks. A GPU regression seeds valid error intervals whose
correct endpoints cross the formerly missed boundary. Native and complete
frozen-product validation of this correctness repair are pending.

Other prompt lengths, longer context
targets, true partial-prefix restore and packaged API acceptance remain
open. The observed speedup does not meet the product performance limits or
qualify a release.

The [reference autotune controls](../benchmarks/correctness/reference-autotune-controls-20260911.json)
explain why captures r15/r16 are excluded: they fail the first original
q7169 control and select different autotune configurations in three prefill
kernels. Reusing all eight byte-identical timing records from qualified r13
restores all eight complete frozen cases in r17, with unchanged cache hashes.
No oracle or model math changes. The observer can now select up to three
continuation offsets independently of the actual accepted MTP cache row;
the final generated history still qualifies every selected row.

Both frozen 32-token cold requests pass on baiying with the real
`D:\models\Qwen3.6-35B-A3B`, whole 8d8af14, MoE a17a9ed, FLA 7698db3 and
CK `ac0fb7d68ea44c9093d9b4cd90ac3c23912a5423`. Each run has zero first-logit
error at tolerance 0.125, all 32 oracle tokens, and 32 matching streaming
callbacks before return. Exit 0 and all host checks pass. CLI and AITER still
use retained f544cbe components; the cases retain different arithmetic profiles.

| Frozen prompt / configuration | First token / logit | Load ms | Actual callback TTFT ms | TPOT ms |
|---|---|---:|---:|---:|
| q8192, whole 8d8af14 plus FLA 7698db3, fast profile | 144 / 10.375 | 20,097.591300 | 4,135.594501 | 32.711306 |
| q7169, whole 8d8af14 plus FLA 7698db3, 1000 ppb profile | 82 / 9.25 | 20,078.009799 | 69,278.586000 | 34.441339 |
| Earlier retained q8192: f544cbe components plus whole 7c2f170 | 144 / 10.375 | 20,254.383200 | 4,163.038700 | 33.007665 |

The [logical-tail records](../benchmarks/correctness/fla-logical-tail-20260911.json)
retain the actual valid length through KKT masking, triangular inversion,
W/U, recurrent-state update and GDN output. q8191 now emits all 32 tokens;
its first token 168589 and logit 11.4375 satisfy the frozen first-token
boundary (reference 11.375, tolerance 0.125). The seventh output differs:
82 instead of 220. This run remains numerically unqualified. The two original
32-token controls and all 80 complete q7169 norms pass, and actual q8192
TTFT, load and TPOT satisfy the unchanged retained targets.

The subsequent [positive-exp2 repair](../benchmarks/correctness/exp2-plateau-product-20260911.json)
matches all 80 complete q8191 prefill normalization tensors to GB10, as
well as all 80 q7169 tensors. Actual AMD scan output identifies a positive
roundoff at layer 4, head 12, token 564; one invalid exponential spreads
into 866 inverse entries starting at token 562. GB10 exhaustively verifies
all 855,638,016 nonnegative FP32 inputs below 2^-25 produce exactly one.
FLA 41236de uses that verified plateau without another runtime artifact.
The same saved-input component now has finite output/state and exact
sync/async parity. q8191 emits the exact first logit 11.375, but output
index 6 remains 82 instead of 220, leaving decode correctness open.
Both original 32-token controls pass. This run's q8192 callback TTFT is
4,240.562900 ms, above the unchanged retained target; the earlier
4,135.594501 ms result remains retained. Local checks pass C/ABI, Rust,
Clippy, 286 Python tests (two skips), q16 and public hygiene.

The [matched-history runtime capture](../benchmarks/correctness/runtime-boundaries-20260911.json)
reproduces all eight frozen GB10 cases and their complete first-logits
tensors. q8191 matches the first nine complete normalization boundaries;
its first differing boundary is layer-4 post-attention normalization. At
each of the three first divergent decode tokens, native layer-0 completed
carriers already differ from the reference under the same input history.
Selected MTP rows retain actual positions, input IDs and logits indices;
unaccepted draft rows are excluded from comparisons. The old global Q1
BF16 residual-norm flag fails q7169's final variance handoff and moves
q8192's first divergence to output index 3, so it is not retained.

The [original GDN stage capture](../benchmarks/correctness/runtime-gdn-stages-20260911.json)
also reproduces all eight frozen GB10 cases. The complete layer-4 q8191
input norm and QKV/Z/A/B projections are byte-exact. The first observed
downstream difference is GDN core output: head 12 jumps above 1e30 at token
562, while all other heads remain below 0.032. The final 64 gated rows
differ in exactly that head. The native stage request stops at its aggregate
capture limit, so these saved tensors do not qualify a completed model run.
Native post-convolution Q/K diagnostic rows are already normalized; the
reference core-input rows are raw BF16 and cannot be compared directly.
The subsequent [raw Q1 comparison](../benchmarks/correctness/q1-projection-boundary-20260911.json)
confirms genuine BF16 QKV/Z projection differences. The same model weights
and input norms replay exactly against all 37,056 GB10 QKV/Z/A/B values
with the K16 width-26 accumulator.

The [native Q1 projection tests](../benchmarks/correctness/q1-sm121-product-20260911.json)
use whole aab009d and the opt-in `QRT_QWEN36_Q1_F32_PROJECTION_SM121=1`.
Layer-0 input norm and all projections match GB10 at the first q8191/q8192
decode steps and the q8192 first divergent extended position. Recurrent
core output still differs. q8192 passes all 32 tokens; q8191 first diverges
at output index 2, q7169 out512 at 32, and q8192 out512 at 109. The first
two diverge earlier than the previous route. The q7169 saved position 7288
is excluded from operator comparisons because its preceding input history
is already different. All requests complete and host checks pass; this
route remains diagnostic. The q8192 32-token callback TTFT is 4162.961901 ms
with selected decode tracing enabled; it does not replace the retained
unprofiled result.

The [original decode state and arithmetic evidence](../benchmarks/correctness/q1-gdn-arithmetic-20260911.json)
qualifies all eight frozen GB10 cases with convolution history and the actual
accepted recurrent state slots. At the first decode step, native q7169 and
q8191 have exact initial recurrent states and BF16 convolution history.
Their old convolution and recurrent updates differ. The fast q8192 profile
already differs in 508,846 initial FP32 state values and 163 BF16 history
values; fixing decode alone cannot qualify that prefill state.

Replaying the original SM121 operator establishes FP32 sigmoid beta and
normalized Q/K, BF16 convolution products, and the original four-K reduction
and FMA order. The portable implementation matches all core outputs, next
states and internal observations for two independently captured cases. The
actual AMD HIP probe also matches all 4,096 core outputs, 524,288 state
values and 16,480 internal FP32 observations using baiying's real q7169
inputs. Host baiying, source `af744045a56af44876affdf26475bb179da5d7da`,
model `D:\models\Qwen3.6-35B-A3B`, command
`run-native-q1-gdn-probe-r1.ps1` (SHA
`bad7138ff5b7ec8112a6a19694ff9997578d2c70072281b1f38bf20b59811c93`);
build and probe complete with all host checks passing. These are operator
checks, not token-loop or performance acceptance.

The opt-in `QRT_QWEN36_Q1_SM121_GDN=1` now connects that convolution and
recurrent implementation, plus exact gated normalization, to resident Q1
decode. Its table paths use separate `QRT_QWEN36_Q1_SM121_*` variables so
enabling it does not change prefill arithmetic. It preserves cache-frontier
checks and supports both existing recurrent layouts.

The [whole-model Q1 tests](../benchmarks/correctness/q1-gdn-product-20260911.json)
at whole 99148b3 pass the Windows HIP build and complete all three product
requests with successful host checks. First-decode convolution, all recurrent
state values, core and gated norm are now exact for q7169 and q8191. Token
equality still fails at output indices 6 (q8191), 38 (q7169 out512) and
115 (q8192 out512). Both 512-token runs match their original first 32 tokens.
Fast q8192 retains its prefill-state differences and 862 core mismatches.
The retained profile is unchanged; these failed continuation runs do not
qualify performance.

The next exact boundary is the linear-attention output: the old Q1
projection differs in five BF16 values for each of q7169/q8191, and the
following postnorm differs in 378/382 values. The same-input K4096 CPU
projection with K16 width-26 accumulation matches all 4,096 reference
values. Norm replay then matches both complete rows with the rounded BF16
numerator and the variance of the unrounded residual addition. Using the
rounded variance or unrounded numerator fails these controls. The opt-in
`QRT_QWEN36_Q1_SM121_OUTPUT=1` connects this projection and existing exact
residual norm to all 30 Q1 linear-attention layers, requiring the exact GDN
route. It also publishes the optional BF16 MoE input without changing the
original norm result.

The [native output-boundary tests](../benchmarks/correctness/q1-output-product-20260911.json)
at whole 2b90db9 qualify every first-decode layer-0 linear-attention boundary
for q7169 and q8191, including the output projection and post-attention norm.
The first completed layer still differs after MoE. All three product runs
complete with successful host checks, but continuation remains incorrect:
q8191 first differs at index 6, q7169 out512 at 38, and fast q8192 regresses
to index 1. The last case's callback TTFT of 4174.093001 ms does not qualify
performance. The original retained benchmark remains unchanged.

The reference observer now copies the original first-layer decode MoE input,
router, selector, shared expert, routed/shared output tuple and next residual
inputs while returning every original result unchanged. Native Q1 adds
bounded optional MoE files using its existing diagnostic reads, plus live
shared gate/down/activation and next-norm endpoints. It caps files at 128 KiB
each, 64 files and 4 MiB total per process.

The [MoE boundary captures](../benchmarks/correctness/q1-moe-boundary-20260911.json)
pass all eight original GB10 controls. Native whole 23a2e9c reproduces the
q7169/q8191 MoE input, router logits and expert IDs exactly, but shared
activation differs in 142/148 BF16 values and routed output in 1109/1218.
The next input norm differs in 1072/904 values. Both native requests complete
with successful host checks and the same token failures as whole 2b90db9.
CPU replay of all 22 observed rows matches the original router and shared
projections with K16 width-26 arithmetic, the scalar gate's cuBLAS reduction,
and BF16 activation/scaling. Original CUDA exp plus its reduction/division
order matches all 176 routing weights; host exp changes 59. These are
operator controls, not full-model or performance acceptance.

The opt-in `QRT_QWEN36_Q1_SM121_MOE=1` connects these rules to all Q1 layers:
exact router, selected/shared projections, BF16 weighted contributions and
route sum, and the rounded norm numerator with unrounded residual variance.
It requires the existing exact Q1 GDN/output path and next-layer norm handoff.
It uses Q1-specific table bindings and has no effect on prefill.

The [native Q1 MoE product tests](../benchmarks/correctness/q1-moe-product-20260911.json)
at whole 9460249 pass the Windows build and complete q8191 out32/q7169 out512
with successful host checks. Every observed first-layer MoE endpoint now
matches GB10 exactly, including the next norm. Complete layer-0 and layer-1
output carriers also match; the earliest difference moves to layer 2
(969/1005 F32 values). Continuation still first differs at indices 6 and 38.
The first q8191 attempt stopped at an obsolete rocBLAS cache dependency;
completed reruns disable that cache. The code now skips this MoE-only
dependency for the native HIP replacement.

The [all-linear Q1 product tests](../benchmarks/correctness/q1-all-linear-product-20260911.json)
at whole 441f0f2 apply the qualified K16 projection to all 30 Q1 linear
layers, replacing the middle/later AOT QKVZ+A/B bypass. The Windows build
passes, and q8191 out32 now passes the frozen prompt, all 32 output tokens,
first-token logit and streaming boundary. q7169 out512 still first differs
at output index 38, with 467 mismatches; its original first 32 match. Both
requests finish with successful host checks. At first decode, both cases'
layer-2 input/postnorm and complete carriers for layers 0, 1 and 2 match
GB10 exactly. The first remaining carrier difference is full-attention
layer 3 (1823/1788 F32 elements). Diagnostic callback TTFT is 80.858/69.697
seconds and TPOT is 96.846/86.302 ms. No new performance result is retained.

The [expanded Q1 full-attention boundary](../benchmarks/correctness/q1-full-boundary-20260911.json)
passes all eight original GB10 cases at observer ca9e925. Failed r7/r8
captures remain excluded. Scoped hooks handle shared rotary modules and
capture the first accepted decode row's actual logical KV history, excluding
future draft rows. These observations confirm every captured layer-2
projection, convolution, recurrent state and MoE endpoint matches at whole
441f0f2. Full-attention layer 3 receives the exact input norm, but its QKV
projection differs in 3/10 BF16 cells; ordinary rotary arithmetic then differs
in 250/290 Q and 35/35 K cells after BF16 conversion.

Original-input CPU replay matches Q/K/V/output projections with K16
arithmetic and reproduces the BF16 rotary multiply/FMA and sigmoid product.
All tested normalization trees match these two rows, so those rows alone
do not identify a unique reduction tree. Native original-input attention
replay at 4f1701c matches every one of the 4096 context values in both cases,
using the complete actual KV history and original 32-token online reduction.
Both operator runs pass host checks; their 1.316/1.166 ms GPU intervals are
operator diagnostics. Whole 3bf42dc builds, but an obsolete output-consumer
guard stops its first decode before any new full-attention kernel executes.
Whole 623c8ff fixes that guard and integrates the attention core over the
owned prefix/tail caches. Its diagnostic norm writes alias live Q/gate
scratch, and a later terminal guard still stops completion. Whole 3aa6402
separates the observations and scopes the terminal guard to the exact route.
Its q8191 request completes, with all first-decode full-attention endpoints
and all 40 layer carriers bit-exact, but later continuation remains incorrect.

The next q7169 run exposes a distinct lifetime error: pooled attention-output
scratch is released through raw `hipFree`, leaving stale pool entries that
can later release a reused persistent allocation. Whole 341c88d returns those
buffers through their owner and keeps persistent suffix snapshots outside
the transient pool. The actual-function ownership regression rejects the old
implementation and passes the corrected success and failure paths. Local
C/ABI, Rust, Clippy and 288 Python tests (two skips), plus the Windows build,
pass. The fixed pool now completes both requests; q7169's first layer-24
state, convolution, projection and MoE endpoints are exact. The pool-disabled
q7169 out32 control passes the complete frozen boundary. Longer continuation
still differs, independently of the lifetime repair.

Original layer-24 capture r10 and continuation-KV capture r11 each pass all
eight frozen GB10 cases. At q8191 position 8196, native context differs in
114 BF16 values while the current Q/K/V are exact. Full prefix/tail cache
comparison identifies only the earlier row at position 8194. Extending the
bounded capture replay to 16,384 input tokens allows the actual 8,197-token
history to run: all context values match using original inputs, with intact
input buffers and transpose redzones. Its 1.3119 ms GPU interval is an
operator diagnostic, not product timing.

The next original captures r12/r13 pass all eight frozen cases and locate
q8191's earlier KV difference in layer-2 gated normalization. Its input,
convolution, recurrent states and core are exact. One gated BF16 value changes
nine output-projection values, one residual/postnorm value and 494 completed
carrier values. Original TTGIR/PTX distinguishes short decode's 32 lanes of
four adjacent values from prefill's 16 lanes of eight. Both use the original
product-1/FMA-0,2,... order. Replaying the short layout matches all 4096 gated
values. The regression fixture preserves the captured head and compiled-code
fingerprints; it exposes the one-ULP sum difference between the layouts.

Whole df05a32 connects this layout to exact Q1 decode and passes the Windows
build. The original failing position now has all 40 carriers and every
captured layer-2/full-attention endpoint exact. Its full q8191 request still
differs at output index 6, requiring the next continuation comparison. q7169
at position 7206 instead has differing incoming layer-2 recurrent state despite
exact current projections and convolution. Its complete KV first differs at
position 7179. Second-row state comparisons use the original preceding-row
updated state and the actual accepted convolution-history offset.

Capture r14 lacks the requested first-pair transaction and is excluded. The
corrected-position r15 fails the original q7169 out32 control itself (220
instead of 82), so it is also excluded; the frozen oracle is unchanged.
Local C/ABI, Rust, Clippy, 291 Python tests (two skips) and public hygiene pass.
No new product performance or release is qualified.

The [bounded attention-output product tests](../benchmarks/correctness/attention-output-tiles-product-20260911.json)
remove the q8193 layer-3 K4096 tile guard, but the request emits 64 instead
of 220 and then encounters an unspecified launch failure during layer-0
decode copy synchronization. Owned cleanup reaches its 300-second deadline;
all post-run host checks pass. Subsequent q7169/q8192 32-token controls pass.
The new whole db6f3ca q8192 callback TTFT is 4,211.237900 ms, above the
unchanged 4,187.415605 ms retained target. The earlier 4,135.594501 ms result
remains retained; this change does not qualify a replacement package.

The preceding [final-norm reference matrix](../benchmarks/correctness/final-norm-reference-matrix-20260911.json)
passes both original 32-token requests after matching the scalar check to
the selected BF16 numerator. All 80 complete q7169 norms remain GB10-exact;
actual q8192 TTFT, load and TPOT meet the unchanged retained targets.
Both 512-token requests emit all callbacks but fail token equality: q8192
first differs at output index 109 (110th token), q7169 at index 120 (121st).
q8193 emits 64 / 9.75 instead of 220 / 9.75. None of these failed numerical
runs qualifies a performance result or release.

The preceding [transposed-QK product records](../benchmarks/correctness/attention-transpose-product-20260911.json)
qualify an owned 8 MiB key workspace alongside the 4 MiB score slab. All
80 complete q7169 GB10 norms remain exact. The same-profile actual TTFT
improves by 11,059.441299 ms. Native build, allocation/cleanup controls and
full local checks pass, as do same-run actual q8192 retained TTFT, load and
TPOT. Full-prefix exact attention is disabled in the fast q8192 profile;
its overall timing change is not isolated to the new QK mapping. A unified
profile/package and broader product acceptance remain open.

The [complete expanded GB10 capture](../benchmarks/correctness/gb10-token-matrix-20260911.json)
passes all eight requests: the two original 32-token controls, their four
adjacent lengths, and both 512-token continuations. The separate
[cold token matrix oracle](../contracts/gb10_cold_token_matrix_20260911_oracle.json)
freezes every prompt, output token and sampled first raw logit. Original
contracts remain byte-identical. The observer follows the pinned runner’s
actual discard mask and skips one intermediate chunk for q8193. Its sampled
first token is 220 / 9.75. All five completed cases from the
[earlier incomplete capture](../benchmarks/correctness/gb10-token-matrix-incomplete-20260911.json)
are reproduced exactly, including their full first-logits tensors.

The [Windows adjacent-length records](../benchmarks/correctness/neighbor-tokens-20260911.json)
pass all 32 tokens, first logit and streaming for q7168 (220 / 9.5625) and
q7170 (94 / 18.25), using the same strict arithmetic profile. Their actual
TTFT is 69,223.110000 and 69,411.140199 ms. The earlier q8191 run stops before token emission:
the scalar final-norm guard compares an unrounded numerator against the
selected GPU formula’s BF16 carrier. Repairing that mismatch still does not
complete q8191. Follow-up tracing identifies 22,528 nonfinite values in the
layer-0 combined output: all channels of the last 11 tokens, positions
8180 through 8190. The next stage capture has finite QKV, convolution and
gate inputs but 169 nonfinite GDN outputs and 317 nonfinite state values.
An inactive padded gate can move outside the negative-only exponent table;
the actual gate-kernel test reproduces contamination of zero padded dots.
Logical-tail masking removes the model failure as described above, while
q8191 continuation, q8193 and both 512-token Windows gates remain open.

The [MoE phase profile](../benchmarks/correctness/moe-subphases-20260911.json)
observes 40 physical-q8192 MoE calls for the real q7169 request. Total MoE
time is 16,283.295993 ms; gate/up accounts for 9,984.341538 ms and down
for 4,295.882592 ms. Shared work (3,498.791668 ms) overlaps the routed
branch and must not be added to the total. All 32 tokens and 80 complete
GB10 norms pass. Profiling synchronizes each observed call; its 71,368.956700
ms TTFT is diagnostic and does not replace the unprofiled result above.

The [FLA segment records](../benchmarks/correctness/fla-sequence-20260911.json)
qualify ordered batches of at most 32 kernels and failure cleanup. All 80
complete q7169 GB10 norm files remain exact; each measured sequence is below
100 ms (maximum 25.895800 ms). The one-run TTFT difference is 732.659201 ms,
so per-operation CPU synchronization was not the dominant remaining wall.
The four logged stage intervals total 11,334.771059 ms and exclude other work.
Full local and Windows checks pass, as do same-run q8192 retained targets.

The [three-stage attention replay](../benchmarks/correctness/attention-probability-20260911.json)
separates exact QK scores, online probabilities and PV accumulation. All
29,364,224 BF16 cells match GB10, and the native FP32 control is unchanged.
Eight-query batches take 2,917.23 ms versus same-run CK 2,375.87 ms;
32-query batches take 2,670.21 ms versus CK 2,384.68 ms. These slower
component intervals do not qualify a product improvement. Layout 3 remains
replay-only; the selected CK provider and product results above are unchanged.

[Optional stage events](../benchmarks/correctness/attention-stages-20260911.json)
measure the exact QK/PV pair at 2,264.82 ms: QK scores 1,554.21 ms and
online softmax/PV 710.612 ms. The three-stage layout instead takes
2,838.96 ms (QK 1,548.54, probabilities 194.963, PV 1,095.46 ms). Both
complete BF16 outputs match GB10; native FP32 equality remains diagnostic.
QK is the dominant component in this captured layer. Optional events are
absent from provider defaults, and these component intervals are not TTFT.

The [transposed-key replay](../benchmarks/correctness/attention-transpose-20260911.json)
retains the exact K16 sequence while assigning each QK dot to one lane.
All 29,364,224 BF16 endpoints match GB10, with unchanged native FP32
diagnostics, correct transposed inputs and intact redzones. Eight-query
batches take 1,288.17 ms including 0.979980 ms of key preparation; same-run
CK takes 2,409.15 ms. QK falls to 582.884 ms and online softmax/PV takes
704.307 ms. The 32-query control takes 1,293.53 ms. These component
checks precede the full-model qualification recorded above.

The [MoE submission records](../benchmarks/correctness/moe-dispatch-batch-20260911.json)
qualify 1024-CTA batches with the original selector and dot arithmetic.
The same q7169 profile improves by 18,252.573599 ms; all 80 complete GB10
norm files remain exact. The actual-kernel dense control checks 262,144
candidate cells in 6.963310 ms; dense tail and sparse cases also match the
scalar reference with intact redzones and inputs. These synthetic intervals
are safety checks. Both full-model runs supply their own frozen GB10 gates;
q8192 again meets actual retained TTFT, load and TPOT targets.

The [replicated-carry records](../benchmarks/correctness/replicated-carry-20260911.json)
bind all four successful native builds and both product runs to uniform K16
carry ownership. q7169 improves by 6,861.723700 ms with all 80 complete GB10
norm files unchanged. The full attention replay matches all 29,364,224 GB10
BF16 cells and the qualified native FP32 control in 2,306.52 ms; this component
interval is not TTFT. The same-run actual q8192 retained TTFT, load and TPOT
pass. Full local checks also pass; broader product qualification remains open.

The [split-attention product records](../benchmarks/correctness/attention-split-product-20260911.json)
qualify the reusable 4 MiB score arena and eight-query QK/PV dispatch pairs.
q7169 improves by 11,850.483200 ms against the same 1000 ppb profile; all
80 complete GB10 norm files remain exact. The native CK build and full local
check pass. q8192 meets the actual callback TTFT, load and TPOT retained
targets on that same run. The two cold cases still do not qualify a unified
release package or suitable arbitrary-length performance.

The preceding [candidate-batch product records](../benchmarks/correctness/collection-batch-20260911.json)
retain the same precision and all 80 complete GB10 norm files. Batching
candidate collection independently of exact computation reduces q7169 TTFT
by 8,601.340400 ms. The captured real QKV replay matches all 58,728,448
BF16 endpoints with immutable inputs and intact redzones: four collection
windows, 60 exact dispatches, maximum dispatch 2.591 ms. Scratch is bounded
at 64 MiB plus two counters; the exact-dot quantum and 100 ms/10 second
limits are unchanged. Windows build, native safety checks and the complete
local check pass. q8192 again passes the actual retained callback target;
these two profiles still do not qualify a unified release package.

The [independent-PV product records](../benchmarks/correctness/attention-pv-product-20260911.json)
qualify all 32 tokens and logit for both cases. That run’s actual q8192 callback
TTFT passes the immutable retained target, as do load and TPOT. Full-prefix
exact attention is disabled in that fast profile, while exact terminal exports
use the PV mapping. The total q8192 timing change is not isolated to that mapping. q7169 improves by 9,498.169100 ms; all 80 complete
norm files still match GB10. The separate prior shared-stack q8192 target miss
remains recorded below. Package and broader product qualification remain open.

A same-provider q7169 ablation reduces only the four projection selection
bounds from 1000 to 100 ppb. It emits 220 / 9.375, with only token zero
incorrect; actual TTFT is 114,808.991700 ms. This failed numerical gate rejects
the result for performance, despite healthy host checks. MoE remains at
1000 ppb and exact FLA/CK arithmetic is unchanged. The record above preserves
this control separately from the qualified 126,453.136300 ms run.

The [shared-arithmetic records](../benchmarks/correctness/shared-wave16-20260911.json)
bind four successful Windows builds and both full-model runs to the common
K16 implementation, including its final negative-zero canonicalization.
q7169 improves by 11,857.096400 ms from the 147,808.401800 ms control; all
80 complete norm files still match qualified GB10 capture f17592ae. That preceding
q8192 run misses the immutable retained TTFT target by 7.842595 ms. Neither
provider-only time nor the earlier mixed-stack result qualifies this newer
combination's retained performance. Native CPU wide-sum/endpoint controls,
UBSan, 270 Python tests, 45 Rust tests and the complete local check suite pass.

The [q8192 preload records](../benchmarks/correctness/q8192-preload-buffer-20260911.json)
bind actual callback TTFT **4,163.038700 ms**, below the unchanged
4,187.415605 ms retained target; TPOT is also below 35.502151 ms. Command
`run-preload-buffer-q8192-r1.ps1` has SHA
`9aa0c45efb1b1f8eaf5c89aa8d448c9a4b48571e7cd47c876550c72f6931d481`;
run SHA `a5846b46c84939ce3101eb83112bd8b14cae432e2c96aa67dccd81f0e9767e5d`.
The whole DLL is `73060dddc816d0e531a70b8ea21efcae546b909fe3507d2d4acac6aa2db49279`.
Model/resource validation and GPU synchronization remain active. Buffering
three diagnostic lines before writing removes per-field stderr flushes:
reused preload falls from 18.4262 to 1.7355 ms, and its model-store log phase
from 13.8763 to 0.1145 ms. The remaining provider-time variation is separate
from that measured saving. The earlier 4162131 q8192 run missed the target
at 4,202.652900 ms; its shorter provider time was never substituted for TTFT.

The [resident-decode and ablation records](../benchmarks/correctness/decode-bf16-argmax-20260911.json)
bind commands, source/artifact identities, profile changes, prompt hashes and
complete output to each run.

The preceding [q7169 whole-model record](../benchmarks/correctness/q7169-attention-canonical-20260911.json)
passes all 32 tokens and all 80 complete layer norms with the optimized CK
provider. Command `run-q7169-attention-canonical-r1.ps1` has SHA
`358bcf1cd4da622e9939d133f56c9cf596a2b44681915a3efc8f29022ff71a2b`;
run SHA `eec6ef02af3cd47fa2d470c20fb9425d60a491227c6c44dbeb2fbd71b9dd7488`.
Actual TTFT improves by 24,283.513800 ms from the preceding 1000 ppb control,
while remaining too slow for release. MoE, FLA and CLI identities remain
023e5dc, 2f346df and f544cbe. All 80 norm-file hashes remain identical to
qualified GB10 capture f17592ae.

The preceding q7169 command `run-decode-argmax-q7169-selective-r1.ps1`
(SHA `3fb9cb34e0b050928854eeb32145d669054e4bb547d74e913adf3832d84f4ebb`)
has run SHA `a245863ec254794a991aed79bd6995001f621a6a87ab8a67f7e0f59575f042a4`.
It retains MoE 023e5dc, FLA 2f346df, CK b609453 and CLI f544cbe. All 80
complete layer norms still match the independently qualified GB10 capture
f17592ae. Its four whole-provider projection bounds decrease from 10000 to
1000 ppb and ordered FLA removes per-stage synchronization. This reduces
TTFT by 76,555.146600 ms from the
[preceding 378b0c3 control](../benchmarks/correctness/q7169-output-boundary-20260911.json),
which also matched all 40 terminal BF16 residuals and the final norm. Those
additional terminal/final observers were not repeated in the subsequent runs.

Faster q7169 ablations fail the frozen token gate. They retain the
final-norm/BF16-argmax repairs and MoE 023e5dc; fast CK/AITER components are
f544cbe. The first three use whole 378b0c3 and the last uses whole 4162131. Commands and full identities are in the same structured
record. All complete with healthy host checks; none qualifies performance.

| Ablation | First token / logit | Actual TTFT ms | Oracle mismatch |
|---|---|---:|---|
| Fast matrix, attention and AITER recurrence | 220 / 9.3125 | 8,311.036100 | Token 0 and tokens 26–31 |
| Restore qualified FLA; retain fast matrix/attention | 220 / 9.375 | 25,531.042500 | Token 0 |
| FLA plus 100 ppb selective matrix correction; fast CK | 220 / 9.4375 | 64,761.139700 | Token 0 |
| Qualified 1000 ppb control with only CK attention replaced | 220 / 9.375 | 98,466.847700 | Token 0 and tokens 29–31 |

The 100 ppb ablation first differs at layer-zero post-attention normalization:
only 1 of 80 complete norm files matches. The isolated fast-CK ablation
matches the first 7 norms and diverges at layer-three post-attention norm,
the first full-attention layer. Its command is
`run-decode-argmax-q7169-fast-ck-r1.ps1` (SHA
`9737ad2bac59c18de7b52c6913e2fa89a7f9a8b91a6fa1ec181425ed9cf8a9cc`), run
SHA `949d2a234dfdecc37c773ec9854ffc3a430316d999ec1bf87ea37a247fbe2237`.
These norms localize the failed
product boundary; they are not independent acceptance gates. Cold correctness
at two profiles does not establish a unified release configuration, paired
decode, prefix reuse or the required context/API matrix.

The exact-attention optimization at source f9f227e uses bounded unsigned
32-bit K16 reduction with explicit sign recovery, preserving sums that exceed
INT32_MAX. One million native CPU controls, signed-overflow edges and UBSan
pass. In the [complete real-input replay](../benchmarks/correctness/attention-modulo-20260911.json),
all 29,364,224 BF16 and raw FP32 outputs match the independently qualified
reference boundary. Original CK takes 6,802.52 ms per provider call; the new
8-query kernel intervals total 5,585.03 ms with a 12.6929 ms maximum.
The 32-query control remains bit-exact but gives only a small incremental
improvement. The subsequent 5345630 component additionally removes redundant one-value
normalization: kernel intervals total 4,707.43 ms, maximum 10.8648 ms, with
all BF16 and raw FP32 values unchanged. The full-model result above qualifies
that change on the frozen q7169 gate. These component intervals are not TTFT.

A subsequent [shared-memory layout replay](../benchmarks/correctness/attention-layout-20260911.json)
at source 6399c3f keeps all 29,364,224 GB10 BF16 endpoints and the prior
qualified native FP32 boundary exact. Direct-load kernel intervals total
4,670.41 ms; staging Q/V takes 8,077.02 ms and staging Q/K/V takes 9,726.89 ms.
Both staged layouts are rejected for performance and remain absent from the
production default. The replay now returns nonzero for numerical mismatch,
while preserving output artifacts for diagnosis. These component intervals
are not product TTFT.

The replacement at source d116e07 assigns each PV output dimension to an
independent lane and packs each K16 product in one register. All 29,364,224
GB10 BF16 outputs and the same native FP32 control remain identical. Kernel
intervals total **3,822.13 ms**, maximum **9.3963 ms**; the same-run original
provider takes 4,561.25 ms. One million independent packed-product alignment
controls pass under UBSan, as does the complete local check. This mapping is
qualified by the complete frozen q7169 product run above; broader release
qualification remains open.

The next [split-QK replay](../benchmarks/correctness/attention-split-20260911.json)
at source d30356f computes a bounded slab of exact scores before ordered
online softmax/PV. All 29,364,224 GB10 BF16 outputs and the qualified native
FP32 control remain identical. Eight-query dispatch pairs total 2,417.09 ms,
maximum 5.664 ms; the same-run serial-PV provider takes 3,886.01 ms. The
32-query control totals 2,474.38 ms, maximum 23.5475 ms. The next provider
uses eight-query pairs and a reusable 4 MiB score slab protected through
completion and release. The complete local check passes 271 Python tests
(including actual scratch-span/submission-failure guards) and 45 Rust tests.
These component intervals do not qualify a product result; full-model
qualification of the integrated provider remains pending.

The preceding edcbe6f run (SHA
`ec42c758962a2c67c24de8895180a8fa1da3567a049009d7a7ebf2dbaafcbd2f`)
already matched all 80 layer norms and all 40 terminal residuals, but emitted
220 / 9.25. Its final normalization differed in 375 of 2,048 BF16 values.
The repaired final norm rounds the numerator to BF16 while retaining the
unrounded FP32 variance and original reduction order. Its native real-input
control (run SHA
`c16177c36ce7546a2361751f8a1d9dab9d17853c2be2d25e30620e9c21951760`)
changes 375 mismatches to zero with immutable inputs and intact redzones.
The complete model now produces final-norm SHA
`be3354ef1cd706c10a5ee58ae4da7492389d1679cff89e02fa5a6e1229355483`,
identical to GB10. The explicit `QRT_QWEN36_FINAL_LAYER_FULL_PREFIX=1` computes
layer 39 with the same full-prefix providers for 2–8192 tokens and selects
requested rows for the output head.

The independent full-vocabulary CUDA LM-head replay at source
`33e74b24d97d3301276fda6486cd53db31dc49f7`, host `aitopatom-66c4`, original
model `/mnt/data/models/Qwen3.6-35B-A3B`, command
`run-qrt-gb10-lm-head-20260911-r1.py`, capture SHA
`f5f92840ad4e473ff5bf1fda0d02a249c930e0cf5c8d2c3f02baf07435458385`,
confirms both 82 and 220 have BF16 score 9.25 and original argmax chooses 82.
An unrounded dot gives 220 a larger score and incorrectly changes that BF16
model's decision. The native observer independently shows BF16 policies choose
82 while FP32 policies choose 220. `QRT_QWEN36_LM_HEAD_BF16_ARGMAX=1` now keeps
the full-vocabulary BF16 ordering and its minimum-ID exact-tie rule, disabling
unrounded rescoring and empirical permutations. It passes the native q7169 gate with the final-norm repair and remains opt-in
until broader context and performance qualification. The option also applies to
resident single-token and paired output-head execution: BF16 score ordering
owns every decode position, with legacy FP32 rescoring and near-tie policies
inactive. Both frozen batch-one 32-token continuations now pass with the extension.
The paired output-head route has not been exercised by those runs.

The complete-window correction control at source 1ac1ce1 passes all
58,728,448 captured real layer-zero QKV BF16 outputs against GB10, immutable
inputs and intact redzones. Its 897 exact dispatches peak at 2.451 ms.
Host baiying, captured real model tensors, command
`run-native-final-prefix-r1.ps1 -Action test-real-qkv -ErrorBoundPpb 10000`, run
SHA `b6fd8a399b3c28888cd4f0b9afb08a057734e302039417ebcf3673b34ecdd405`.
Component timing does not qualify product speed; all mission gates are unchanged.

The qualified GB10 layer-20 capture confirms that every actual GDN input is
exact, including raw convolution output, FP32 decay and BF16 beta. Native
q64 replay of real positions 6272–6335 reproduces a parallel prefix scan
rounding difference of positive 2^-41 in head 24. The nonpositive exponential
lookup rejects this tiny positive value, producing one NaN that the triangular
inverse spreads to 1,382 entries from local row 18 (global position 6290).
The lookup now returns one for either sign below magnitude 2^-32; larger
positive arguments remain rejected and the existing artifact is unchanged.
Local regression and the complete check suite pass. The original SM121
instruction exhaustively returns exactly one for all 796,917,760 FP32 inputs
in [0, 2^-32). The rebuilt Windows component preserves the scan and upstream
inputs while eliminating all NaNs. Its complete real q7169 layer-20 replay
matches every one of 29,364,224 GB10 BF16 core values. The full-model run confirms this
repair and exposes the later layer-26/27 boundary described above.

The root-cause replay runs on baiying against captured real inputs from
`D:\models\Qwen3.6-35B-A3B`, FLA source
`831c1699c3c956ace00365bb42c2deb2045a7ea7`, command
`run-native-linear20-chunk98-old-r2.ps1`. Run SHA
`99fb6530d56aa99efa6cb6157e79b0eface7165de75c98ed53cd145e2a41942e`;
430.324 ms, all host checks pass. Its finite output and internal sync/async
agreement are insufficient for numerical acceptance. The same-run qualified
GB10 full-model capture, source `ca24ee7c56f176495ccbc48454963b4b9d3e7273`,
command `run-qrt-gb10-linear20-20260911-r2.py`, host `aitopatom-66c4`, model
`/mnt/data/models/Qwen3.6-35B-A3B`, retains 82 / 9.25 and all 32 oracle tokens;
capture SHA `0182e854e9689331751bb7dafe0816203d5f12cf7ce5658ca9b9e2f233aa11f4`.

The shared Blackwell attention arithmetic, original SM121 exponential,
1/4/2/16/8 reduction and reciprocal coefficients match all 29,364,224 BF16
and raw FP32 attention values through the actual CK DLL ABI. The full-prefix
route is opt-in through `QRT_CK_FMHA_SM121_FULL_PREFIX=1` and validates its
tables by SHA. It remains too slow for product acceptance. The q7169 continuation passes in the configuration above; retained speed and
release qualification remain open.

The following records preserve how those boundaries were established.

The corrected-FLA full model uses whole provider e9a7012, FLA 2f346df,
MoE f164f0b, CK b609453 and CLI f544cbe, retaining each component's recorded
hash. Host baiying, model `D:\models\Qwen3.6-35B-A3B`, command
`prepare-fla-model-q7169-exp2-positive-r1.ps1` (SHA
`e54bcaf601699fcc1091d11283e904bfd25225c78faeda539a76a8b536afcf54`), run SHA
`eb749897bb8e6b193ed8f11f4a713284902895ddad7cda7fdd2a285dce761610`.
All host checks pass; 92 observation files total 2,967,621,888 bytes. Full
layer-20 core, gated norm, output projection and post-attention norm are
GB10-exact. Complete norms improve from 41 to 54 exact files. The layer-27
input relative L2 difference is 0.000025863643 and maximum error 0.03125.
No 32-token native continuation or performance acceptance follows this failed
first-token gate. The optional `--moe-layer` reference observer captures the
live router, shared/routed output tuple, MoE sum and next residual inputs;
its total three-GiB ceiling also covers an optional selected linear layer.


The exhaustive tiny-positive capture uses source
`2f346dfea97f07e89fee56d96009d2d539249a5b`, host `aitopatom-66c4`, original
pinned SM121 image, command `run-qrt-gb10-exp2-positive-20260911-r1.py`; no model
is loaded. Capture SHA
`73e7d856a46a8dcaf4b6788074ad9321100f8bc89a9c61e765eadaef36811b43`,
zero mismatches, 0.055475 seconds enumeration, 0.061536 ms maximum GPU dispatch.

FLA rebuilt on baiying at that same source has DLL SHA
`4f2cf28e282aa4e7edf2b34b66ce7044837212ae0965cedfdee66eb3ae7a3134`, 477,184 bytes.
Command `build-native-gdn-exp2-positive-r1.ps1`, build run SHA
`5bb885e061467c02025585f6ce9b1654ba1158207327bcaebc37df17031f6cb5`,
33,200.163 ms, all host checks pass. Corrected q64 real-chunk command
`run-native-linear20-chunk98-fixed-r1.ps1`, run SHA
`853e2372569a208d3dc32606c81ffba54d7051966015008e7f9a8df145264365`,
653.416 ms, eliminates the KKT/inverse NaNs with upstream files unchanged.
Complete q7169 real layer-20 command `run-native-linear20-full-fixed-r1.ps1`,
run SHA `78a19f9a81a504c0780545b1c4dc8b0d9d7e402234fefae6349b06a056118a40`,
1,776.562 ms, output SHA
`b17c9c28a59dbf818d0dbf658915f0ab0774bb7ed3924d0ff144a62369acf837`
is identical to the same-run qualified GB10 full-model capture. These component
runs use saved real `D:\models\Qwen3.6-35B-A3B` inputs from a zero initial state;
the replay does not load the model or claim product TTFT. All host checks pass.


The optional FLA route at `33e0492ee17013aa897eb19aa092c15cf56df2bf` now
matches the saved real q7169 layer-0 recurrence exactly on native Windows:
29,364,224 V-new BF16 values, 59,244,544 checkpoint BF16 values, and all
524,288 raw FP32 terminal-state values. Shorter q64/q384 views also pass.
These replays use fingerprinted GB10 intermediate inputs and load no model.
They establish a component repair, not an accepted inference route.

A new GB10 q384 capture preserves saved BF16 parity and same-run raw terminal
parity before recording 12,480 independent exponent instruction samples.
The CPU probe covers all 4096 head/value rows without feeding reference
checkpoints into its carried trajectory. Exact exponents, continuous
Blackwell K128/K64 accumulation and fused state update match every raw bit;
host exp2, split projection and unfused-update controls do not.

The optional `QRT_FLA_GDN_SM121_EXP2_TABLE` artifact exhaustively enumerates
all 2,139,095,041 nonpositive FP32 inputs through negative infinity. Its
builder accepts no model or prompt input. Lossless packing produces
183,174,448 bytes, SHA256
`f490940df2bd80421159a96424c3e922330b7ca120d5ae7b629a973b9183730b`.
All earlier independent samples also match the actual native lookup.
Windows validates the fixed SHA through system CNG before uploading it;
[dependency policy](dependency-policy.md) records the memory/packaging cost.
The published default profile is unchanged.

The r15 Windows build reuses unchanged, source-validated r14 AOT artifacts;
its FLA DLL SHA256 is
`f8339eadfca6a25379e9455eaa4b091948cccdb3887fac80108a0f4680038b0a`.
The compiler-owned state kernels retain explicit dispatch/shape bounds,
disjoint state ownership and scratch accounting. Full q7169 isolated replay
uses 514,663,728 bytes and 226 dispatches, maximum 2.060 ms. All native guards
complete with healthy cleanup. The launcher still validates compiled ABI,
including Triton's auxiliary scratch slots. These times are diagnostics.

The subsequent real-model q7169 test on `baiying`, using
`D:\models\Qwen3.6-35B-A3B` and command file
`prepare-fla-model-q7169-exact-exp2-r2.ps1`, **still fails**: output token 220,
logit 9.25, versus GB10 token 82 / logit 9.25. Prompt FNV matches
`c900c18703532d6a`. Engine load is 20122.1765 ms and TTFT 13267.5151 ms.
The FLA source is `33e0492`; the unchanged whole-provider source is
`c36e2674a5ade5544ef49a3756dbe99454b00991` and CLI source is `f544cbe`.
Run-record SHA256 is
`e8ca72baaf49064c47caee2c329442d3d5c02c2793d1b559cbaa8684bff13da1`.
This mixed-component diagnostic is not an all-component package build, a
passing continuation result, or accepted performance.

The extension at `0b10c4571c5403f972a4238d2a43720a80466c42` applies the
general exponent table to KKT gating and adds optional native Blackwell W/U
and output kernels. On the full captured q7169 inputs, W and U each match
all 29,364,224 BF16 values exactly, with the production U=V alias. Each CTA
reads its complete V column tile into shared memory before writing U. The
standalone output comparison improves to 177 differences / 29,364,224
values, relative L2 8.8967834e-8; it is still not bit-exact. The q64 component
controls pass, including the output kernel. Dispatches remain bounded to
64 tokens with per-call timing admission. Output reuses dead state residual
scratch; the standalone harness reads back one chunk at a time.

The integrated real q7169 replay still has 1,904,743 output differences
(relative L2 0.0009123102), which are much larger than the isolated output
kernel's residual. The following real-model test also **fails** with token
220 / logit 9.3125; the diagnostic top five are
`220:9.3125, 82:9.125, 64:9.0625, 144:8.9375, 83:8.8125`.
Load is 20004.9421 ms and TTFT 21870.2059 ms. The full route is not promoted.
The r16 FLA DLL is
`78aa91c9f64a611606ef1a3cee5f8fdc7013e24ad393f940e25113a256beeec0`;
the whole-provider and CLI retain their preceding source identities. Command
file `prepare-fla-model-q7169-blackwell-aux-r1.ps1` runs on the same Windows
host/model and oracle; run-record SHA256 is
`df4b3ee3bcaeda11b63ff090ab71e63615aa8b5ed681b868f769adde9d1238e0`.
All native build/replay/model guards complete with passing host checks.

The remaining investigation follows actual input propagation through
normalization, gating and inverse arithmetic; isolated kernel success does
not erase an incorrect predecessor. Full local checks pass 243 Python tests
(two platform/dependency skips), 45 Rust tests,
clippy, C ABI, seven q16 contracts and public hygiene. The skipped NumPy
packing tests pass in the existing reference container; these source checks
cannot replace real-token, prefix, HTTP/archive and retained-performance
qualification. The retained q8192 target and numerical tolerance remain fixed.

The later normalization correction at `3d82d6950b082f36b866cd248845e3973684a1ab`
reproduces every Q/K BF16 output in the complete frozen q7169 component case.
It uses the independently characterized reduction order and the general
reciprocal-root table described in `docs/dependency-policy.md`. Integrated
q64 output differences fall from 31 to one, but the complete model still
emits 220 / 9.3125 instead of the authority's 82 / 9.25. Native load is
20,023.1357 ms and prefill 21,866.8483 ms. This remains an unqualified opt-in
combination; the triangular inverse and output arithmetic remain under
investigation. Source checks pass 254 Python tests (two known skips), 45
Rust tests, clippy, C ABI, q16 contracts and hygiene. Native component and
model runs completed with passing host health/cleanup checks. No release
or retained-performance qualification follows from the component correction.

The next inverse implementation is controlled by
`QRT_FLA_GDN_INVERSE_BLACKWELL=1`. A fresh capture of the original reference
kernel reproduced the saved real q64 inverse exactly with two warps; four
and eight warps differed at eight and nine BF16 elements respectively. Its
emitted PTX identifies the diagonal FMA reduction order and the continued
FP32 accumulator across off-diagonal matrix products. Replaying the actual
new arithmetic header on the host matches all 14,682,112 BF16 inverse values
of the full q7169 capture (SHA-256
`7c6a9c8445ab3496c7468282aa7cd2d6cc47ed6849aa57762b84d56e57cecc62`).
The native kernel uses 33 KiB of shared memory per 64-token/head block and
adds no table or runtime dependency. Native Windows r18 at
`fdace5e1dfd7b5e40a78a59b8e2098b8fb4ea091` also matches every inverse element;
its maximum dispatch is 1.070 ms. The integrated real q64 prefix is exact
at every recorded stage and terminal state. Full q7169 terminal FP32 state
is now exact, while output retains precisely the standalone kernel's 177
differences. All host checks pass; full local checks pass 259 Python tests
(two known skips), 45 Rust tests, clippy, C ABI, q16 and hygiene.

The corresponding real-model command
`prepare-fla-model-q7169-blackwell-inverse-r1.ps1` on baiying, using the same
real model, prompt, oracle, CLI and whole provider, still **fails**:
220 / 9.3125 instead of 82 / 9.25. Load is 20,133.0916 ms and TTFT
21,836.5686 ms. Model run-record SHA-256 is
`f1c5de3a870f0673fc398e1f785e25364b1ee001e78a621e8ffcb1b898c8ef1e`;
the r18 FLA DLL is
`5e91a6450cd1a641cd0edcf65aa9ec4b2515711e9e798923e30918ea83080a21`.
This component correction does not qualify the combined model or performance.

The remaining output correction rounds the scaled prior before fusing the
local term, following the reference's evaluation order. A CPU attribution
with the general exponent table covers all 177 residual cells and 3,585
fixed-stride controls: all 3,762 match with this order. Reversing the FMA
retains all 177 differences; separate scaling retains 74. Windows r19 at
`992b04becc8057c2fc27477c4b40f6e748308184` now matches all 29,364,224 output
BF16 cells and 524,288 terminal FP32 state cells in the integrated q7169
capture. The q64 full-stage control remains exact. Its DLL SHA-256 is
`42c9e97fac635bd2caf773341d1d8f104cebb39dc73bd8b0df46d28b542d8640`.

The matching real-model run `prepare-fla-model-q7169-blackwell-output-r1.ps1`
still **fails** with 220 / 9.3125; token 82's logit is 9.125. Native load is
20,034.3477 ms and TTFT 21,868.1872 ms, with passing cleanup/host checks.
Run-record SHA-256 is
`a3cb14d4d190367bf6469e65de803b92404cc13e32a7b89bbefc33bf0285c2ac`.
The original layer-zero traces were suppressed by the required-marker and
deferred/compact log filters. Disabling those filters makes the existing
traces observable. The exact saved-input GDN case is component evidence; the
model remains unqualified.

`QRT_FLA_GDN_CAPTURE_FIRST_DIR` captures the first complete native GDN call's
actual raw inputs, gates, outputs and final state for 1..8192 tokens. It uses
one 1 MiB host buffer, completed stream reads, an exclusively created output
directory and a completion record after every operation succeeds. It never
supplies values to inference. Failed capture stops the call and cannot be
retried in the same prepared provider. The optional hook avoids reliance on
whole-provider trace branches. Host fault/tail tests and full local checks
pass 262 Python tests (two known skips), Rust, clippy, C ABI, q16 and hygiene.
`QRT_FLA_GDN_CAPTURE_CALL_INDEX` optionally selects the zero-based eligible
call (0..63, default zero). Earlier calls execute without capture reads or
filesystem writes; the completion record identifies the actual call index.
The selected-call tests verify skipped reads, original execution order,
unchanged inputs, one capture, and rejection of malformed indices.


The native r20 first-call control matches the existing q64 inputs and outputs.
The real model on baiying, `D:\models\Qwen3.6-35B-A3B`, at FLA commit
`831c1699c3c956ace00365bb42c2deb2045a7ea7` and whole-provider commit
`c36e2674a5ade5544ef49a3756dbe99454b00991` exposes differences before GDN:
7,014 Q, 8,852 K and 16,121 V BF16 cells differ from the frozen layer-zero
GB10 inputs. Raw gates differ as well. Command
`prepare-fla-model-q7169-first-call-r1.ps1` still emits **220 / 9.3125**
against **82 / 9.25**, with load 20,033.6432 ms and TTFT 22,046.4739 ms.
Run-record SHA-256 is
`2f5c27c099f630f0f7c0d20dbb2da54762f5df1143f643cabf0268850d6735fb`.

A BF16 projection-boundary control, using the same safe base profile and
`prepare-fla-model-q7169-bf16-endpoints-capture-r1.ps1`, reduces G relative
L2 to 1.20422e-7 (all BF16 endpoints match) and beta mismatches from 19,057
to 90. Q/K/V remain unchanged; GDN output mismatches fall from 7,223,511 to
4,308,499. The terminal input RMSNorm, Z, A and B projections match their
GB10 BF16 endpoints exactly, while QKV differs in 49 / 8,192 cells. The
model still emits **220 / 9.3125**; load is 20,076.6755 ms and diagnostic
TTFT 23,104.1936 ms. Run SHA-256 is
`ff049cd32bd2ec3f42808f3ccdd8d00b40d913dbb8ad52a78d70638a636ba752`.

CPU recomputation uses the verified terminal input and actual model QKV
weights (SHA-256 `b06edcc9973be74f862d072f89a195e2eecea067ee615144be8510656122401f`).
The production characterized accumulator at width 26 with continuous K2048
matches all 8,192 reference endpoints; split K1024 retains five differences.
This is a projection diagnostic, not whole-model acceptance. Its record SHA
is `274b785da42728dc7ac0f589e3a6fa82e0ff264305cf8a4ec971dca78189d1e3`.

The initial real-model WMMA/correction probe stops before correction because
915,149 candidates exceed the former whole-projection quota of 131,072,
although its densest original block has only 12 candidates. It emits no
model token. Run-record SHA-256 is
`3e46fa77950447f3438021dd8f258aa35306522f8fca6aaaff5a811a5549a931`.
All three model commands pass host and cleanup checks. No result qualifies
continuation, prefix reuse, product performance, or a release.

The correction launcher now streams 65,536-element collection windows into
constant 512 KiB index scratch, then executes only compacted candidates.
Indices remain absolute across token/row boundaries. Collection uses at most
256 blocks; exact-dot dispatches retain eight blocks, completed-stream
synchronization, the 100 ms dispatch deadline and 10 s aggregate deadline.
The 131,072 candidate and 64-per-source-block limits apply to each window.
Count-only mode leaves outputs unchanged. Host tests execute the production
launcher with mocked HIP to check cross-window tails, no repeated indices,
admission failure, asynchronous error cleanup and count-only behavior.
Native `qrt-projection-safety.exe --correction` adds an irregular K2048
case and a synthetic full q7169 geometry with more candidates than the former
global quota. At `5fc0c7a2e6e7c3c78f4d797e3cd4873ab83e2471`, native
Windows validation on baiying passes all 132,999 endpoints for an irregular
K2048 case and all 58,728,448 endpoints for the synthetic q7169/K16 geometry.
The latter streams 917,633 candidates through 897 windows and 7,170 exact
launches in 301.540 ms (max completed dispatch 0.296 ms); all input/output
redzones and host/cleanup checks pass. The first synthetic test incorrectly
expected some zero outputs to retain a negative final-product sign; its
corrected expectation is independently checked against the production scalar
accumulator. Full-model validation of this launcher is pending.

The isolated original-source gating capture at public commit
`3c0bff19659dde390d2a711dcbe5a555f2a91ff1` uses the pinned SM121 reference
image and actual layer-zero A_log/dt_bias parameters. Its direct real-token
control and independent full-domain table lookup both match all 32 FP32 G
and BF16 beta values. It enumerates every BF16 encoding (2,097,152 G entries),
with 17 GPU calls, maximum 0.094848 ms and 1,323,008 peak allocated bytes.
Capture SHA-256: `aabe5e05361f9f064e6765c6df06c7f1c9b9ed860a41ec43bdf3992408895e08`.
The initial attempt included lazy module loading in a 121.550 ms CUDA timing
interval and stopped before enumeration. Explicit launcher initialization
now completes before timing, preserving the 100 ms dispatch threshold.
Both owned containers have exited; the original reference service remains
stopped. These are arithmetic controls, not model/release acceptance.

The new whole-provider build at `5fc0c7a` has DLL SHA-256
`07a40b5f9feafd40919b014bf157e593a76081307c803737d2c40ee8f52b4add`.
The real q7169 command `prepare-fla-model-q7169-streamed-qkv-capture-r1.ps1`
finishes both early-layer corrections in 858.675 / 873.671 ms. Actual GDN
input Q/K/V mismatches fall to 2,994 / 2,691 / 5,450; gates are unchanged.
GDN output differences fall to 3,892,295. The terminal QKV row retains 47
BF16 differences, 45 near 1e-36 and two cancellation endpoints at rows
1,887 and 7,835;
these are diagnostic coordinates, not a separate rejection boundary.
The model still **fails** with 220 / 9.3125, while token 82 is at 9.1875
against the authority's 82 / 9.25. Load is 20,083.1322 ms, diagnostic TTFT
24,725.1796 ms and wall 45,194.445 ms, with passing host/cleanup checks.
Run SHA-256: `96685eec2bf56137d73e5a6a383fd918a8fb26703aa0d9efc5036e7fa596b309`.

The first combined layer-zero gate-table run stops before gate execution:
its initial collection timing includes an asynchronously queued upstream
QKV projection, reporting 124.455 ms despite no exact-dot launch. The launcher
now completes and separately reports that producer wait before starting its
collection/correction clock. Product TTFT still includes the entire wait;
100 ms per-dispatch and 10 s aggregate correction bounds remain unchanged.
Host tests cover a failed producer synchronization before allocation or any
new kernel submission. Real-model gate-table integration remains pending.

The corrected launcher at `b0847999057132ea027dfdb722cabb8bbcdbce29` builds
DLL SHA-256 `f440e3e8e6eb7e9d9615ba6e158425b582e7e69b50f93ca39ee7b602c47864c0`.
The completed layer-zero gate-table model run separates 80.576 / 75.822 ms
producer waits from 772.244 / 800.431 ms correction work, whose maximum
completed dispatches are 0.662 / 0.587 ms. G differences fall to 84 FP32
cells, but beta remains different at 90 positions. It still emits **220 /
9.375**, with load 20,294.459001 ms and TTFT 24,773.7647 ms. Run SHA-256:
`fa36207aba40f3aafe7ad1371682a5ef07a4b3df7aae750619fa97a427e29ae6`.

The follow-up `prepare-fla-model-q7169-streamed-qkv-gate-dot2-capture-r1.ps1`
uses the existing BF16 dot2 A/B projections in layers zero and one. G has
20 FP32 differences (relative L2 1.49274e-11); beta remains at the same 90,
and Q/K/V retain 2,994 / 2,691 / 5,450. It still **fails** with 220 / 9.375,
while 82 now has the reference 9.25 logit. Load is 20,060.325099 ms, TTFT
24,740.179199 ms and wall 45,181.228 ms. Run SHA-256:
`65b620e957ac5daea3e79d40b6f5ce7b90cf5ff8f40c4d7ccb4697d57e4d0a8d`.
All native host/cleanup checks pass. CPU recomputation from captured input
rows reproduces the 90 expected beta endpoints with four declared projection
orders and the sigmoid table, so actual native A/B values must be observed
before attributing the remaining discrepancy. No product result is promoted.

`QRT_QWEN36_GATE_INPUT_CAPTURE_DIR` optionally saves the actual host A/B
projection vectors and A_log/dt_bias at the gating handoff. The layer selector
is `QRT_QWEN36_GATE_INPUT_CAPTURE_LAYER` (default zero). It accepts 1..8192
tokens, at most 1 MiB per projection, requires a new directory and writes its
completion record last. It only reads existing host vectors. Tests cover an
irregular token count, exact float bits, immutable inputs, invalid shapes,
filesystem failure and no overwrite. Native capture validation is pending.

The mixed-type gate capture passes native Windows compilation at
`5bbddb4161dd3ec5ec359e88c406caa9d6f19a65`; DLL SHA-256 is
`7d724b758734c8b664e512dbd9768942e57f5f77dbe268fa0ab75a618cf04836`.
`prepare-fla-model-q7169-native-gate-input-r1.ps1` still emits 220 / 9.375
with passing host/cleanup checks; its run SHA is
`59374d94e7819e0e5afd91d06a37f716bbc2d1569568c2ea1fbaf88654220e95`.
The actual A/B vectors have 41 / 217 BF16 differences against the complete
CPU projections. Model parameter bits match the table's source exactly, and
both lookup results match every actual GDN gate input. The remaining 20 G
and 90 beta differences therefore arise before the table handoff. A complete
CPU projection-plus-table control reproduces all 229,408 reference G and
beta values; its record SHA is
`71b64b5b179223c15df0ee2478597e62780bd2d4dbf2c0245d7c3ad2546c1ae8`.

`QRT_QWEN36_EXACT_ARBITRARY_EARLY_AB_HAWKEYE_LAYERS` adds an opt-in exact
A/B route using the characterized continuous-K2048, group-16, 26-bit
accumulator. Every output is computed from the current inputs and weights;
it takes precedence over fused A/B projection when configured. The existing
compacted launcher handles full selection with fixed scratch and eight CTAs
per exact dispatch. Its 64-candidates-per-CTA bound now applies to the actual
16-subgroup compacted geometry; source-block density is reported separately.
The 100 ms dispatch and 10 s aggregate deadlines remain. Host tests cover
dense windows as well as sparse/tail/error cases, and the native correction
test adds full `[7169,32]`/K2048 geometry. Native evidence is pending; this
route is not enabled in a default or release profile.

The native `4b73ada31e9918632357de16ba2cd261ab811a4a` correction control
passes all three cases, including all 229,408 dense A/B endpoints at K2048
in 143.212 ms, with intact redzones and passing host/cleanup checks.
`prepare-fla-model-q7169-exact-ab-capture-r1.ps1` then captures actual A/B
with **zero** differences to the complete CPU projections, and actual G/beta
with **zero** differences to all 229,408 GB10 reference values per surface.
Both early layers log complete A/B recomputation. This resolves the observed
layer-zero gate projection/handoff boundary.

The same real-model run still emits 220 / 9.3125 instead of 82 / 9.25.
Load is 20,032.834999 ms, diagnostic TTFT 25,290.212600 ms, wall 45,706.544 ms;
host and cleanup checks pass. Run SHA:
`efa291750a89c4d8f7358965db1169d6dcafba73f824b1dac29dbf129677cec2`.
No first-token, continuation or performance result is accepted. The next
component replay uses actual native input-RMSNorm and model QKV/Z/convolution
weights on SM121, comparing fused and separate projection geometry followed
by the fingerprinted original convolution kernel. Frozen outputs are only
comparison targets. This replay is optional, bounded and model-free.

The SM121 replay at `0c937b6a3aef355b3455b2e692e360870f424863` completes
both fused QKVZ and separate QKV cases. Their complete 58,728,448-element
BF16 projections are identical (SHA
`c15e4bf72433bcf534d82c0d97e34026b01849fa935b68d751dbc0a423a446f4`),
and both terminal projections match all 8,192 GB10 cells. Original convolution
still differs from the saved GB10 sequence in 2,962 Q, 2,630 K and 5,356 V
values. Native Q/K/V differences are 2,994 / 2,691 / 5,450. Thus changing
projection geometry on the reference device does not remove the main residual.
The original PTX confirms BF16 products and FP32 convolution accumulation.
CPU continuous-K26 projection matches all 1,143 sampled SM121 coordinates.

The native mismatch distribution clusters at token 97 and the subsequent
three convolution positions. The replay can now start from fingerprinted
actual model embedding rows and the original GemmaRMSNorm implementation,
comparing eager/compiled normalization and feeding the compiled result into
projection/convolution. This tests the preceding normalization boundary.
The first replay's 512 MiB allocation bound stopped before convolution;
the revised bound includes transient CUDA BLAS workspace, records allocation
stages and completes at a measured 549,770,752-byte peak under 1 GiB. Both
owned containers have exited, and the original reference service remains
stopped. These are component diagnostics, not inference acceptance.

The original compiled GemmaRMSNorm replay at `6862ee7` changes exactly 54
native input cells: one feature at each occurrence of token 97 or 99. Using
that result, both SM121 projection geometries and the original convolution
match **every** saved Q/K/V BF16 value. Correct full input-RMSNorm SHA is
`6c67321f81040780088a742070c48ab3023043b8ef9de71d1ed5a970a8fe9161`;
the complete replay record SHA is
`5522ec3e4725adf459e505dff61aa3981a08b416b7b2bfe40e929ccf250b4d8b`.
The generated reduction has XBLOCK=2, R0_BLOCK=2048 and 16 warps. Eager
PyTorch only changes token 97 and is not substituted for the compiled
authority. The separate native FP32 sequential-reduction route changes 88
cells and worsens Q/K/V differences to 7,043 / 6,181 / 12,708, so that
profile is not retained.

The next optional builder observes the original compiled inverse scale,
requires both its full normalization output and separate inverse application
to reproduce the complete real-token control, then enumerates every model
embedding. Its output is a 248,320-entry FP32 table for the existing native
layer-zero inverse-scale path. Model weights, source, launch geometry and the
comparison capture are fingerprinted; expected outputs do not generate table
entries.

The full-vocabulary inverse table completes at `ca45343`: 123 launches,
0.334720 ms maximum dispatch, 58,761,728 peak device bytes, with both complete
controls exact. Table SHA:
`f4e37f759c586bfc8fcc4d74cefdd89235f0f0c0c90cd286147e331e87509e67`;
capture SHA:
`13de7395d4bc89dbbd39d1292f8d780d9d827ece9e6808efca0abdc1e4a9e1d7`.
`prepare-fla-model-q7169-embeddingnorm-capture-r1.ps1` verifies the complete
native embedding/norm tensor hashes and uses the existing `4b73ada` DLL.
All 14,682,112 native normalized BF16 cells now match the compiled reference.
The model still emits 220 / 9.375 versus 82 / 9.25 (82 is at 9.1875).
Load is 20,045.338200 ms, diagnostic TTFT 25,517.049100 ms, wall 45,940.778 ms;
native host/cleanup checks pass. Run SHA:
`6a4fc47b153e645d0ba1a68903b8fb1e343bea0959e3977f5f48f05d0c79ff8b`.
Terminal QKV retains 47 differences (45 tiny values and two ordinary
cancellation endpoints); core/gated/out have 439 / 453 / 656 BF16 differences.
Complete post-convolution Q / K / V captures retain only 32 / 61 / 94 BF16
differences; all G / beta cells match. The subsequent 10,000 ppb L2 selector
probe reaches its unchanged 10-second correction deadline before any token:
14,686,604 candidates, 115,055 exact dispatches, maximum dispatch 1.125 ms.
The guarded process exits normally with a rejected result and healthy host.
No product result has been accepted.

The compacted correction permits an explicit 64-CTA batch, keeping the
eight-CTA default and the 100 ms / 10 second deadlines. Each subgroup still
owns exactly one dot. The native regression adds a real QKV replay that reads
the complete current normalized inputs and actual weights, evaluates every
BF16 output against the independent SM121 capture, reports selector coverage
and required observed error scale, and checks redzones and input immutability.
Reference values never participate in GPU computation. Native `569846a`
passes the dense K2048 control in 21.581 ms (225 dispatches, maximum 0.199 ms),
with control run SHA
`cabb2fea8f31157a50b36d4e0d0f9f450c5231724630c7b6b22a42792f03f2a6`.
The real QKV replay matches all 58,728,448 BF16 cells, with intact redzones
and unchanged inputs. At 10,000 ppb it corrects 16,842,592 candidates in
1,889.600 ms, using 16,903 dispatches with maximum 2.152 ms. The uncorrected
WMMA has 346,303 BF16 differences; the selected set covers all of them.
The maximum observed required selector scale is 87 ppb, a diagnostic rather
than a general error guarantee. Run SHA:
`2281bf6161bb7e7c4d0730db333a1726fc651e2f81def5267c65911606741589`.
This component check does not substitute for token-loop acceptance.

The corresponding live model run keeps all four terminal QKV/Z/A/B projections
exact, but terminal core/gated/out retain 408 / 427 / 657 BF16 differences.
The first token remains 220 / 9.375 (reference 82 / 9.25; actual 82 is now
9.25). Load is 20,005.306700 ms, diagnostic TTFT 28,181.956999 ms and wall
48,567.896 ms. Both full QKV corrections complete within their bounds; host
and cleanup checks pass. Whole DLL SHA:
`b521728b3bb2a3b9642031562f93af8d13a9473724149315a75bde92333b8f05`;
run SHA: `9ba7dce5b190ed8ef3921271165845639422cc696c523a4c08de8944d5661b43`.
`tools/compare_gdn_capture.cpp` compares complete live GDN inputs/output/state
on either CPU host and emits only counts, errors and at most 64 differing
coordinates per surface. Its local replay reproduces every prior comparison
count and maximum error. This avoids transporting each large live capture
while retaining reference and capture fingerprints.

The native CPU comparison completes in 625.040 ms. Remaining Q/K/V differences
are 28 / 43 / 45; G and beta remain exact. GDN output/state differ in
2,032,879 BF16 / 255,346 FP32 cells. Comparison run SHA:
`765abbc9d58ea51ebe89383bce11e49dc38131a05aa6d5e7806872879ba01a59`.
The 116 convolution endpoint differences map to 23 distinct activation inputs
computed from the already exact QKV and actual convolution weights. A new
model-independent SiLU builder requires all these cases plus 192 distributed
controls to match the original expression before exhaustive finite-FP32
enumeration. Its compact transition representation is subsequently rechecked
over the same full domain; native integration is pending.

The `f99f97e` builder passes every one of 4,278,190,080 finite FP32 inputs
both during enumeration and packed-lookup verification. The 308 held-out
convolution controls pass before and after construction. There are 64,304
transitions; the complete table is 648,036 bytes, SHA
`673f8dd1280700578c1e8743afd2e3b4da134b1fbd463c890527e1c4d9f796b8`.
The capture SHA is
`6a90bb3e61d2b6c18ab69c1bd84e46da90fc09cc3e48a12868f5be77030617eb`.
All 2,040 dispatches complete, maximum 0.262144 ms and peak device allocation
1,111,040 bytes. The first staging attempt failed on a missing Python helper
before GPU execution; the corrected bundle validates its import closure in
isolation. Both owned containers have exited.

`QRT_QWEN36_SM121_SILU_TABLE` opts into this model-independent endpoint for
mode-3 convolution. The native loader rejects incompatible modes, a second
correction table, malformed layout or mismatched SHA. The default remains off.
CPU lookup matches every held-out sample and all transition boundaries, and
rejects malformed schema/directory inputs. The native regression compares
complete real QKV/weight convolution before and after table application;
native Windows validation follows below.

The native `a09b89688ddd7f84dc067cde51be9f75c86b6ac3` convolution control
matches all 58,728,448 real Q/K/V BF16 cells with the table; the same kernel
without it retains 28 / 43 / 45 differences. Redzones and immutable inputs
pass. Command `run-native-silu-table-r2.ps1 -Action test-real-conv` on baiying
uses captured tensors from `D:\models\Qwen3.6-35B-A3B`; control run SHA:
`bb28bcceb6c1e40851341ea4ab8303484db4ecbaf1e71d531ff97d0ac37bac93`.
Whole DLL SHA:
`f50d039872f672b56ed298f672e0d1862bbf8f1a51de6c6e6481207d80b088cf`.
The initial build rejected three legacy launch sites missing the newly added
argument; the completed build passes with their optional table explicitly null.
Full local checks pass 264 Python tests (two known skips), 45 Rust tests,
Clippy, C ABI, seven q16 contracts and hygiene; the launch-site fix also passes
all 32 native-route contract tests.

`prepare-fla-model-q7169-embeddingnorm-sm121silu-capture-r1.ps1` runs the real
model on baiying with that whole provider, unchanged `831c1699` FLA and
`f544cbe` CLI. The complete first live GDN call now matches every Q/K/V/G/beta
cell, all 29,364,224 output BF16 cells and all 524,288 terminal FP32 state
cells. There are no nonfinite values or raw-bit differences. Native CPU
comparison run SHA:
`c922ca663d266411a55e7bba67f370a0745043a9c830816bf35c0ab5c1184297`.
All terminal QKV/Z/A/B, core and gated-RMSNorm BF16 cells also match. The
terminal output projection retains five differences at rows 414, 564, 858,
1682 and 1803.

The first generated token still **fails**: 220 / 9.375 instead of 82 / 9.25.
Load is 20,031.357800 ms, diagnostic TTFT 28,165.328200 ms and wall
48,580.559 ms; host and cleanup checks pass. Model run SHA:
`1c2143295a888155906780956b17b2a20fd0521b32c32027a04249257a976843`.
This is live upstream correctness evidence, not accepted full-model inference,
continuation, performance or package qualification. CPU output-projection
recomputation from the now exact terminal gated input and actual model weights
matches all 2,048 reference BF16 endpoints using continuous K4096 / width 26;
split K2048 retains four differences. CPU run SHA:
`6ac5318e953ab9fd3056faf0ce65f7df47961fb7228d90c09748f13f336554eb`.
The follow-up `prepare-fla-model-q7169-sm121silu-out-capture-r1.ps1` uses the
same source and DLL to apply full-sequence output correction. All terminal
output-projection BF16 cells now match. The layer-zero residual/add BF16
endpoints also match; post-attention RMSNorm retains 420 differences. The
first token remains 220 / 9.375, load 19,992.455600 ms, diagnostic TTFT
30,718.662300 ms and wall 51,099.908 ms. Host and cleanup checks pass.
Run SHA: `f2266de9becaa2305833749fb3b4ae60a40e9683acb6e056362f35d808605984`.
The selector processes 12,911,801 of 14,682,112 output cells with compacted
64-CTA batches; the original source-window cap covers the explicit full shape,
while fixed scratch and dispatch deadlines still apply. CPU normalization
replay reproduces all 420 native differences and eliminates them when the
numerator uses the BF16-rounded residual sum while variance uses the unrounded
sum. Computing variance from the rounded sum retains 237 differences. This
supports testing the existing vLLM residual/norm path with the corrected
upstream; it does not establish a full-sequence or token-loop result.

The existing residual/norm path with the repaired upstream is verified by
`prepare-fla-model-q7169-sm121silu-out-postnorm-r1.ps1` on baiying using the
same real model, `a09b896` whole provider, `831c1699` FLA and `f544cbe` CLI.
All terminal BF16 surfaces through post-attention RMSNorm now match GB10;
the eight MoE routing IDs and FP32 weights also match. The first token still
fails at 220 / 9.375, with load 20,008.115900 ms and diagnostic TTFT
31,746.907800 ms. Host and cleanup checks pass. Run SHA:
`2108ab0cc5665f7fe3aac2e0eafea7bbad61feecb86864abffe64cab7bcfc0b5`.
The follow-up MoE trace requires both the all-layer selector and the existing
layer-one stage switch even when capturing layer zero. Its next-layer terminal
seed retains 334 BF16 differences; the token remains 220 / 9.375, native load
20,030.214900 ms and TTFT 31,476.069000 ms, with passing host/cleanup checks.
Run SHA: `71fe30daf1a3021179f785a341d625c62c4c301cb4f63549f557abcd11f1f4dc`.

`scripts/capture_sm121_post_gdn.py` adds a full q7169 reference replay for Z,
gated normalization, output projection and residual normalization. It starts
from fingerprinted real normalized inputs, actual weights and the explicitly
GB10-validated GDN component output. Original kernel sources and every input
are pinned. Full fused QKV must reproduce the existing 58,728,448-cell control
before its Z result proceeds; each new reference surface must pass its frozen
terminal control before full native comparisons are retained. Additional
replays hold native predecessors fixed to distinguish propagation from local
arithmetic. Native captures and expected terminal values only compare results.
This supervised, opt-in component probe is not an inference or timing gate.

That replay completed on GB10 with source `1cdb433`: the full QKV and all four
terminal controls have zero differences. Holding native predecessors fixed
leaves 342 gated-normalization differences, zero output-projection differences
across 14,682,112 cells, and 46 residual-normalization differences. Capture SHA:
`67e01e6dc0cf26580ad4b8c5a9d1e49c7fca01e612f247c4c78070734f6a5a49`.
The original gated kernel runs in 0.792064 ms after module initialization is
excluded from its unchanged 100 ms dispatch limit. This is component timing.

On baiying, `prepare-fla-model-q7169-sm121silu-z-moe-endpoints-r1.ps1`
uses the same real model and component commits while correcting Z and selecting
the GB10-anchored BF16 weighted-contribution/FP32 route-sum endpoints. All
29,364,224 Z cells now match. Gated normalization retains 342 differences,
propagating to 10,432 output and 2,809 post-normalization differences. The
second layer's terminal seed improves from 334 differences to zero; its input
normalization still has 444 differences. Token 220 / 9.375 remains incorrect;
load is 20,001.767300 ms and diagnostic TTFT 33,294.909000 ms. Run SHA:
`a31589eb53eb36eabda81cd10a6508976c6f52e69562c8463f380ddb85759155`.

Selecting residual normalization for all 40 layers exposes a missing variance
handoff in the padded MoE route. The bounded run exits normally with code 5
before emitting a token; host and cleanup checks pass. Run SHA:
`ed26e2bf872fedd1d0811a94187bf1e7176e798218e8bbc5fb3aa4443fd7843a`.
The provider now publishes the unrounded variance when the next layer requests
it, including padded tiles. It rejects providers without the required residual
endpoint before attempting to derive that variance. The full local check suite
and the native Windows build pass at `f0d916e`. The next real-model run,
`prepare-fla-model-q7169-sm121silu-allnorm-gated-r1.ps1`, also enables the
independently replayed gated mode 3 and fingerprinted SM121 SiLU/rsqrt tables.
All four complete layer-zero surfaces (Z, gated norm, output projection and
postnorm) now match GB10, including all 229,408 FP32 gated inverse scales.
All 39 variance publications are consumed successfully. Layer one's terminal
seed, input norm, QKV/Z/A/B projections match; raw convolution still has
6/11/15 Q/K/V differences and 14 FP32 log-gate differences. Its core output
retains 967 BF16 differences. The final token remains 220 / 9.375, load
20,231.831800 ms and diagnostic TTFT 33,933.016600 ms. Host and cleanup checks
pass; this is not inference acceptance. Native run SHA:
`bc70fdf56fe5c27c1d5b523087f2c5dda415e303f4200d3419b7bdb37d7109b7`.
Full/terminal comparison SHA:
`fe07898caa59d64c1b60bbc5fb9fd84d80e936dcf85cf7220caa23de2b9230eb`.

The gating-table probe can now enumerate all 30 linear-attention layers in one
supervised run from fingerprinted model parameters. The primary real-token
control remains mandatory; additional captured layers verify the same original
kernel and independent table lookup. Every table enumerates all BF16 inputs,
and the shared sigmoid table must agree across every head and layer. Local
validation covers the complete parameter set, malformed spans/shapes, duplicate
layers and a mismatched primary model binding. The 30-layer data would occupy
240 MiB plus a 128 KiB shared sigmoid table. The bounded SM121 capture at
`9d612717` has now enumerated all 62,914,560 layer/head inputs, with both
independent real-token layer controls matching. Its 483 launches take at most
0.299808 ms each, and layer zero's tables reproduce the prior fingerprints.
Capture SHA:
`a59388434d3a42a81de464a519f432a67f8d9c61fa14c626469917feb282ba46`.
No runtime default is changed.

The following native q7169 run binds all 60 live model parameter tensors and
loads all 30 gate tables. Layer one's terminal G and beta now match, while its
core retains 960 BF16 differences. The final token remains 220 / 9.375;
load is 20,021.511200 ms and diagnostic TTFT is 33,732.892200 ms. Host and
cleanup checks pass. The CLI, FLA and whole-provider sources remain the
explicitly mixed diagnostic stack described above; this is not acceptance.
Command file: `prepare-fla-model-q7169-allgate-r1.ps1`. Native run SHA:
`0687e013ce30631c525607e2107d956c721326ed538ce124ddd1d2c16a17474e`.

A complete layer-one convolution replay from the native QKV capture reproduces
its terminal native output exactly, but comparison with the independently
captured GB10 prefix finds 638,008 Q, 633,891 K and 1,371,267 V differences.
The first mismatches occur at token zero. Layer-one terminal projection parity
therefore does not establish prefix parity. The next probe,
`scripts/capture_sm121_moe_full.py`, replays the complete layer-zero MoE from
the already validated post-attention input, using actual GB10 model weights
fingerprinted against native safetensor spans. It requires 15 independent
terminal controls before emitting full reference tensors and compares the
unrounded residual and next input normalization. Local preflight and malformed
input rejection pass. The full replay at `f87c1b02` passes all 15 controls and
verifies all seven MoE tensors plus the next normalization weight against the
actual GB10 model. It finds 115,035 unrounded-residual FP32 differences and
55,851 next-normalization BF16 differences against the native prefix capture.
Capture SHA:
`94ecfdd428aa3e1fadc43000e6530264fea4f087e42909c4b0944379ad56431a`.
The owned component container exits normally, with peak device allocation
2,097,953,280 bytes; the original reference service remains stopped.

The opt-in `QRT_QWEN36_FULL_MOE_STAGE_DUMP_PREFIX` diagnostic now copies the
full native router, shared-expert intermediates, routed activations and raw
per-route FP32 outputs through the existing debug ABI. It requires the matching
full top-k layer/token capture and bounds its copy loop to 30 seconds. Reference
replay also retains full expert stages. These additions preserve runtime math
and allow the remaining full-prefix error to be separated into routing,
activation, projection and combination boundaries.

The subsequent native capture at `db049515` completes with passing host and
cleanup checks. MoE input, unrounded residual and next normalization retain
their preceding full fingerprints. All 57,352 expert IDs match, but 19,464
router weights differ at FP32 precision. Shared gate/up projections differ in
3,025 / 4,144 cells; routed activation differs in 88,488. A full CPU combination
replay reproduces all native residual cells exactly. Both reference and native
shared activation obey the same BF16 SiLU/multiply endpoints, directing the
next correction to projection accumulation and router exponent arithmetic.
Native run SHA:
`da87185ec273631a46e70c27db64ffe0d1452e0c39d530e08c3e9181575825d3`.
Token remains 220 / 9.375; load 20,142.950900 ms, diagnostic TTFT including
full capture 36,534.516300 ms. This does not pass the model oracle.


CPU replay from actual small MoE weight tensors validates the existing
26-bit/group-16 accumulator: zero differences across 1,856 router, 6,702
shared-gate, 7,822 shared-up and 22,913 shared-down selected cells. The
selection includes every gate/up mismatch plus a fixed stride control;
shared-down uses reference activations to isolate its projection. Width-25
and FP64 controls retain differences. The next bounded model profile will
exercise the existing sparse projection corrections and CUDA router
exponent compatibility, with independently captured primitive tables.


The MoE primitive capture at `221e1a0c` enumerates all 8,388,608 CUDA
expf fraction-table entries and all 65,536 BF16 SiLU inputs. All 65,280
finite non-unit SiLU product controls match. Capture SHA
`af1c7ad5d3efa69b9adb07e35f07e19d512a1298acddbd13aea08330ced9f2cb`.
CPU replay with the reference router logits reproduces every one of 57,352
real top-k weights. Using native logits leaves 35 differences.

The following native run enables these primitives and midpoint radius 512
for the existing MoE corrections. Top-k weight differences fall from
19,464 to 30; shared output differences fall from 195,199 to 3,047. Routed
activation instead worsens from 88,488 to 100,218 and routed output from
102,007 to 263,798. The conditional routed AOT kernels use `tl.sum` FP32
trees, unlike the characterized group-16 MMA used by shared correction.
Overall MoE differences rise to 164,073; this combination is not retained
as an inference or performance improvement. Native run SHA
`d940e95549756fa3d65e9322d953cbb0e21895c8128c785ab502adfda23a5fe7`;
command `prepare-fla-model-q7169-moe-correction-r1.ps1`, whole `db049515`,
MoE/CLI `f544cbe`, FLA `831c1699`, same real model and q7169 oracle.
Output remains 220 / 9.375, load 20,069.185300 ms and diagnostic TTFT
42,160.582800 ms. Host/cleanup checks pass and the full MoE input is unchanged.

The opt-in `QRT_QWEN36_SM121_ROUTED_HAWKEYE` now selects the existing
wave16, 26-bit/group-16 native gate/up/down correction kernels in builds
that also contain conditional AOT. It requires native batched gate/down
support and the BF16 SiLU table. The default remains the existing route;
this new selection requires full native product-shape qualification.


The native SM121 routed selection at `dc84f3a7` passes Windows compilation,
default smoke, asynchronous parity, and all 32 dynamic logical component
cases. DLL SHA `67f820752b1a8c1cc381f3d83cd99987c4a2221da72a86d4e72f13abc5e5453b`.
Real q7169 run `prepare-fla-model-q7169-moe-sm121-r1.ps1` on `baiying`,
model `D:\models\Qwen3.6-35B-A3B`, uses whole `db049515`, MoE `dc84f3a7`,
FLA `831c1699` and CLI `f544cbe`. Run SHA
`ceaca018e3310fc3c5f251d0d8114e9a25da5774649a41fad73e05d0f01a626f`.
It completes normally with host/cleanup checks passing. The first token
is still wrong: 220 / 9.3125 versus GB10 82 / 9.25. Load is 20,087.255900 ms;
diagnostic TTFT is 44,552.757100 ms, including stage copies and corrections.
No inference or performance acceptance is claimed.

The full MoE input is unchanged and GB10-exact. CPU replay reproduces all
14,682,112 native unrounded combination cells. Compared with the same
GB10 full-stage capture, routed BF16 output differences fall from 263,798
to 79 and total MoE differences from 164,073 to 231. The complete next-layer
normalization differs in 122 cells, versus 55,851 before the primitive and
MoE changes. Shared output still differs in 3,047 cells; router logits in
12, top-k weights in 30, expert IDs in zero. This retains the corrected
accumulation as a component improvement, not a qualified model route.
Full comparison SHA
`7178bf459fa1c640420cdfcc2b0fe21eaa3f9136998250d7a6c7553e6bfb006b`;
next-normalization comparison SHA
`822d2098d6920b2841a08e749328917228605a4cac241ee9ce7173f1acbffea2`.

The next optional selector adds live input/weight L2 metadata across router,
shared gate/up/down and routed gate/up/down. It supplements the midpoint
radius with `QRT_QWEN36_SM121_MOE_HAWKEYE_ABSOLUTE_ERROR_PPB`, default zero.
The configured coefficient is an error estimate requiring numerical
validation; Cauchy bounds the absolute-product sum, not the cross-device
accumulation error by itself. Shared metadata has separate stream ownership.
Each correction launch admits at most 64 CTAs of 256 candidates, and norm
launches at most 4,096 rows. Both neighboring BF16 rounding boundaries are
considered, including the asymmetric spacing at exponent changes. The new
selector requires native product-shape qualification before any acceptance.


The optional L2 selector at public `f164f0b0` passes 265 Python tests (two
known skips), Rust/Clippy, C ABI, q16 and public hygiene. Windows build,
default async parity and all 32 logical-shape cases pass. Native build run
SHA `a15734ff6282115048a2286893532fda55eee8af7e8712a1611c0ff891cd4a23`;
MoE DLL SHA `2249e6e3c7b282d09e08c614d0caf10ef4eba4a5c31a1a5e505ca8446c9f4bdc`.
The first staging attempt failed before build because its source worktree
path was incorrect; the corrected command is `run-native-moe-l2-build-r2.ps1`.

Real `prepare-fla-model-q7169-moe-l2-r1.ps1` on `baiying` uses MoE `f164f0b0`,
whole `db049515`, FLA `831c1699`, CLI `f544cbe` and
`D:\models\Qwen3.6-35B-A3B`. Run SHA
`43b115fd56c36bdf29e93daf247e6c30aabc57f13f85f077f43e5069e0879daa`.
At coefficient 10,000 ppb, every captured first-layer MoE boundary matches
the same GB10 full replay: router, expert IDs/weights, shared gate/up,
activation/down/product, routed activation, all 117,456,896 weighted route
contributions, routed output, total MoE and the next normalization. Native
unrounded combination is also independently reproduced in full. The next
normalization SHA is the reference
`48f9e2ae4c55f3842c81e3294da506416abbbe50958ce7e41dd4b77d5f9474f2`.
Second-layer terminal QKV/Z/A/B and gated normalization now match; its output
projection retains one BF16 difference. Full MoE comparison SHA
`551cc73c73cbcdaaebd8a2c2fca75f5ab783125bf8f1e370fd6886235dfa8eb0`;
next-norm comparison SHA
`7e4de04cf6ed375e705e38e3fbd4c2553e300090e11a26ac40d23ea84ec05f11`.

The complete model still emits 220 / 9.375 versus GB10 82 / 9.25. Load is
20,093.562900 ms, diagnostic TTFT 166,892.948800 ms, and run wall time
187,385.478 ms. It completes inside its 240-second bound and passes all
host/cleanup checks. This establishes component correctness, with a large
cost that prevents performance retention. The next bounded model case
extends the existing characterized QKV/Z/A/B/output corrections to all 30
linear-attention layers and evaluates 1,000 ppb MoE selection against the
same full first-layer boundary. The reference token/logit and q8192 retained
performance target are unchanged.


The all-linear q7169 case retains zero differences at every full first-layer
MoE output/weighted-contribution endpoint and at the next input norm with
1,000 ppb MoE selection. Routed activation contains 2,313 differences that
round away at the weighted outputs; these diagnostic differences alone do
not reject the downstream GB10-valid component. All 30 linear-attention
layers now use the existing characterized QKV/Z/A/B/output route. Native
`prepare-fla-model-q7169-alllinear-l2-r1.ps1` uses the same component commits
and real model as the preceding case. Run SHA
`458d7c3ebf9ee1b14edd69161067a842546ef23a5a7ee5374cd7a6b6a6bcd28a`;
output remains 220 / 9.3125, load 20,104.182800 ms and diagnostic TTFT
221,746.981000 ms. Host/cleanup checks pass within the 300-second bound.
Full MoE comparison SHA `fa7ee3efe39f7f482de76a199022cf93fce7eba7fd7981bbd201c1bf7fc9afcc`.
This is not a retained model or performance result.

The existing `QRT_QWEN36_EXACT_ARBITRARY_LAYER_OUTPUT_TRACE` already reads
each tiled MoE residual and the compact final-layer boundaries. The next
bounded model run enables this trace to locate the first divergent layer
against the captured GB10 rows. No runtime change or new build is needed.
The second-layer terminal QKV/Z/A/B, gated normalization and output
projection all match in this case.


Existing all-layer output tracing completes on the same q7169 profile,
run `9e7122e1fd9496d811319cf7869caac93a37645830cb038203bea1877d035a2d`,
command `prepare-fla-model-q7169-alllinear-trace-r1.ps1`. Host/cleanup checks
pass, output is 220 / 9.3125, load 20,154.908000 ms, diagnostic TTFT
218,689.481000 ms. The complete layer-1 input normalization keeps its GB10
fingerprint. All 40 terminal residual rows are compared with the existing
GB10 capture: layers 0, 1 and 2 match; layer 3 differs in 589 / 2,048 BF16
values. Later layers diverge further. A terminal mismatch can originate in
earlier prefix rows, so this does not identify the failing operator alone.
Comparison SHA `d02ee27de3fab49495d27e506ec3c3f3b4bbadb53aeef391312b789d832656b0`.

Enabling the existing fused full-attention QKV WMMA/correction route at
radius 512 and 10,000 ppb does not close that boundary: layer 3 differs in
593 cells and the output remains 220 / 9.375. Command
`prepare-fla-model-q7169-fullattn-qkv-r1.ps1`, run SHA
`7071eb53abc1167e128d8db6c3384fff93639cf5024131b5a5f667612fb6a6d6`,
load 20,135.284000 ms and diagnostic TTFT 237,543.623600 ms. Both runs use
whole `db049515`, MoE `f164f0b0`, FLA `831c1699`, CLI `f544cbe`, host `baiying`
and model `D:\models\Qwen3.6-35B-A3B`. Neither qualifies for acceptance.

The latter run finishes normally but its old `same_boot` check fails.
The only changed host check is exact CIM boot-time text: it moves backward
1.398022 seconds. The latest kernel boot event is still record 122570 from
00:41 local time, and kernel time-change record 122931 records a -1,398 ms
clock adjustment. GPU status, free-memory checks and process cleanup pass.
The updated guard compares kernel boot-event identity and records CIM time
separately; it retains strict CIM comparison if event lookup is unavailable.
The original failed record remains unchanged.

The next rotary-cache probe binds the original stopped reference's 19 rotary
source files and the actual model config SHA
`93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99`.
It uses the original SM121 MRoPE constructor and checks the complete native
262,144-position BF16 prefix against a second cache extent. Only model
configuration and mathematical positions enter the builder; no prompt IDs,
model weights or inference outputs are inputs. Runtime table use remains
pending native numerical qualification.

## MMLU-Pro full evaluation

| Measure | Windows engine | BF16 authority |
|---|---:|---:|
| Dataset rows | 12,032 | 12,032 |
| Exact correct | 7,486 | 7,486 |
| Accuracy | 62.2174% | 62.2174% |
| Parsed predictions | 12,026 | 12,030 |
| Projection mismatches | 0 | 0 |
| Candidate/reference prediction agreement | 11,180 | — |

Identical aggregate accuracy does not mean every free-form prediction string
is identical. The publication includes each sanitized row so the 11,180
agreement count, parsing differences, correctness flags, and zero projection
mismatch can be independently recomputed.

The row file contains only stable question IDs, input hashes, answer key,
candidate/reference prediction and correctness, usage, and timing fields. It
does not contain dataset question text, answer choices, private endpoints,
machine paths, credentials, or raw service identifiers.

## Reproduction

Obtain MMLU-Pro separately, then run the included OpenAI evaluator against the
resident engine:

```powershell
python .\scripts\eval_mmlu_pro_openai.py `
  --base-url http://127.0.0.1:8000/v1 `
  --model qwen3.6-35b-a3b `
  --output .\candidate.jsonl
```

The harness fetches the pinned dataset revision from the Hugging Face dataset
server and caches it under the output directory. Use `--dataset-cache PATH` for
an existing cache. Candidate-only evaluation is the default; authority parity
requires `--side both --reference-url URL` and should never embed a private
reference endpoint in committed output.

Use the exact v1.0.0 command options recorded in
`benchmarks/eval/mmlu-pro-summary-v1.0.0.json`. Dataset/harness revisions,
prompt templates, few-shot policy, and answer extraction can materially alter
results, so comparisons must retain those fields.

For parity reproduction, pass every assignment from
`mmlu-pro-runtime-overrides-v1.0.0.env` as a `qrt start --set-env KEY=VALUE`
override after loading the production `runtime.env`; `--set-env` values take
precedence over the profile.

## Evidence files

- `benchmarks/eval/mmlu-pro-summary-v1.0.0.json`: formal aggregate acceptance.
- `benchmarks/eval/mmlu-pro-full-parity-v1.0.0.jsonl`: sanitized 12,032 rows.
- `benchmarks/correctness/`: GB10-bound first-token and continuation summaries.
- `benchmarks/openai/`: service/API/queue/prefix acceptance summaries.

All JSON/JSONL evidence is newline-terminated and accompanied by SHA256 in the
release manifest.


The original SM121 rotary constructor is now captured with an explicit vLLM
configuration. The first attempt stops at CustomOp configuration lookup before
emitting a table; source `8d35a1d07b673c5804fa3fb6c0386fef7e46b2a2` fixes that
construction context. Capture `ae093b416a31cebba3b060e8371bedd25dcae35d166eb61592865990e2f7ff6d`
uses the pinned original image on `aitopatom-66c4`, command file
`run-qrt-rope-cache-20260911-r2.sh`, and the real model config. The complete
262,144-position prefix has 16,777,216 BF16 values and zero differences between
the original four-context MRoPE cache and a separate one-context construction.
Peak device allocation is 675,283,456 bytes. The 33,554,432-byte table has SHA
`ba12ce218327d4cf23aac7dfacd8e9efbc99fd207611a8466227089838ef0e80`.
No weights, prompt IDs, hidden states, logits or generated tokens enter this
primitive builder. The owned container exits 0; the original service remains
stopped.

Command `prepare-fla-model-q7169-sm121-rope-r1.ps1` binds the table to the actual
Windows model config and reuses whole `db049515`, MoE `f164f0b0`, FLA `831c1699`
and CLI `f544cbe`. Run `047ecf5cefb5e0874660b0ad6d6114a9774fb0f54da7bd81004b94228e553434`
on `baiying`, model `D:\models\Qwen3.6-35B-A3B`, completes with output 220 / 9.375,
load 20,090.013500 ms and diagnostic TTFT 219,695.110600 ms. All host checks
pass with the new boot-event guard. The full layer-1 input normalization
retains the GB10 fingerprint `48f9e2ae4c55f3842c81e3294da506416abbbe50958ce7e41dd4b77d5f9474f2`.
The log confirms the authoritative cache was used. Terminal residual layers
0–2 remain exact; layer 3 improves from 589 to 491 BF16 differences against the
same GB10 row. The post-attention terminal residual has 118 differences.
The fused-QKV experiment's corresponding residual had 202 differences, but
that run also changed QKV projection, so these counts do not isolate a RoPE
operator error by themselves. Comparison SHA `26ca014e1df82cc42551d67376e642cb94e299e0ba5ef2813b86ba8603c41060`.
The first-token boundary still fails; no continuation or performance
acceptance follows from this improvement.

The next bounded reference capture uses an owned full-model GB10 process,
read-only hooks after startup, a 768 MiB artifact ceiling, and the unchanged
q7169 32-token/raw-logit oracle. It will expose complete prefill inputs and
outputs around the first full-attention layer to separate upstream history
errors from QKV, normalization/rotation, attention and output projection.

The native observer now optionally writes every selected stage and layer
boundary through QRT_QWEN36_FULL_STAGE_DUMP_PREFIX/LAYER/TOKENS. It copies
read-only device data in bounded chunks, caps the aggregate at 768 MiB and
24 files with a 30-second observation deadline, and refuses overwrites.
It requires an explicitly matching prefill shape up to q8192. The default
route adds no copies. All 268 Python cases (two existing skips), Rust/Clippy,
C ABI, q16 transaction checks and public hygiene pass before native build.


The owned full-model GB10 capture now qualifies against the unchanged q7169
oracle: all 32 output tokens match, raw argmax is 82 / 9.25, and the final
normalized row keeps SHA `be3354ef1cd706c10a5ee58ae4da7492389d1679cff89e02fa5a6e1229355483`.
Source `e286e39bbcc9466777c08004d9dea5495a5912e9`, host `aitopatom-66c4`, real model
`/mnt/data/models/Qwen3.6-35B-A3B`, command file `run-qrt-gb10-fullattn-20260911-r1.py`,
pinned original image. Capture SHA
`087343091048d50a12977548a24a6f2a0fd3346352bceb42fb160c3cf6a6bd37` binds 136 tensor
files / 595,121,152 bytes. All 40 captured terminal residual rows also match
the older independently captured GB10 rows. The owned container exits 0 after
329.688 seconds, with minimum available host memory 16,546,750,464 bytes.
GB10 engine startup is not a Windows load-time measurement.

Native whole `2f20ac284f697976cc6e2746073d6345290f4f72` builds on `baiying` with all
host checks passing; DLL SHA `7a8676df3a6818e7411bc3e507e26db091b2a0f514a132fa05e3034787b50d5c`.
Command `prepare-fla-model-q7169-full-stage-r1.ps1`, run
`cf8fc4d935ff3d820753572ee73c26672bc5ed18dbbaef4904f4dfa503af88db`, real model
`D:\models\Qwen3.6-35B-A3B`, completes with 220 / 9.375, load 20,273.837400 ms
and diagnostic TTFT 218,314.385400 ms. The 15 full-stage files total
528,556,032 bytes; every terminal residual remains identical to the preceding
rotary-only run. The complete layer-3 normalized input (14,682,112 BF16 values)
is GB10-exact, ruling out upstream history differences at this materialized
boundary. Q/K/V projection differences are 53,896 / 4,989 / 5,948; full attention
context differs in 1,079,247 of 29,364,224 values.

Combining the existing fused QKV correction (radius 512, 10,000 ppb) with the
original SM121 rotary cache closes the *entire* Q/K/V and normalized/rotated
Q/K boundaries. Command `prepare-fla-model-q7169-fa-qkv-rope-r1.ps1`, run
`95ec62e1f74712e79335972a0825aa19b2ccbf5b890f4e9311a9d16fc0030c1a`, same native
components/model, completes with 220 / 9.3125, load 20,101.129500 ms and diagnostic
TTFT 237,951.357200 ms, all host checks passing. The terminal attention context
and gated context are now exact. The full context still differs in 65,059
values and the full gated context in 58,453. Terminal output projection has
10 differences; its residual and normalization each have five, and the
post-MoE terminal residual has 310. This is a component improvement, not
first-token, continuation or performance acceptance. Full comparison SHA
`08abb008f1006df34d9ffa3e3b5cbd92cb8c20aeaefc44a6d8886ff28294afe0`.

The next output-projection route reuses the characterized continuous K4096
correction, with FP32 matrix output, live L2 bounds and bounded 64-CTA windows,
then materializes BF16 once. A CPU control using the actual layer-3 weights
matches all 2,048 reference terminal output values with the existing group-16 /
26-bit accumulator; FP64 accumulation differs in six cells. It also detects
nine projection-only differences when replaying the rotary-only native input.
The opt-in wrapper covers normal and tiled full-attention output projections;
default settings preserve the original matrix route. The remaining full-prefix
attention differences require independent attention-core work.


Full-attention output correction now closes the terminal layer-3 boundary.

Native whole source `28c4f9df501e8e95ac357acc45d30daf48633bf8` builds cleanly
on baiying; DLL SHA
`0e7c9e2c6f0c64e45b32012e755735a415e0a62f63e4845c47a29df210ddcee5`.
`make check PYTHON=python3.12` passes all 268 Python cases (two existing skips),
45 Rust tests, Clippy, C ABI, seven q16 contracts and public hygiene.
Command `D:\projects\prepare-fla-model-q7169-fa-out-r1.ps1` uses the real
`D:\models\Qwen3.6-35B-A3B`, whole `28c4f9df`, FLA `831c1699`, MoE `f164f0b0`,
CLI/CK `f544cbe`, with the same qualified GB10 full-model capture
`087343091048d50a12977548a24a6f2a0fd3346352bceb42fb160c3cf6a6bd37`.
Run SHA `4a4334d65056371866a10626bec395f9778a2cb6a182ead0560c35ea5cb20146`.

The terminal output projection's ten differences disappear, as do the five
post-attention residual and five normalized-output differences. All 2048
post-MoE terminal residual values at layer 3 now match, so terminal layers
0–3 are exact. Complete layer-3 normalized input, Q/K/V and normalized/rotated
Q/K remain exact. Full context still has 65,059 differences and gated context
58,453; their captures are unchanged from the preceding run. Output projection
now has 784,949 full-prefix differences, post-attention residual 195,153 and
normalization 191,996. These remaining downstream counts are affected by the
known differing context. Layer 4 has 1,033 terminal residual differences.

The final first token remains wrong: 220 / 9.375 instead of 82 / 9.25.
Load is 20,373.593500 ms and diagnostic TTFT 253,150.381300 ms; all host checks
pass. This is neither continuation nor performance acceptance. Evidence is
under `build/recovery-20260910/fla-model-q7169-fa-out-r1/`. Next work isolates
the full-prefix attention core using the now-exact Q/K/V input capture,
without loading the whole model for each numerical decision.


Full attention component closure uses source
`73a15846f7aceb47f982a03cc4a6258d7c9ebf19`, command
`D:\projects\run-native-attention-replay-r3.ps1 -Action replay -Name full-sm121-rcp-r1
-QueryCount 7169 -Batch 8 -UseTable -UseRcp`, host baiying. The captured Q/K/V
belongs to the qualified layer-3 real-model q7169 boundary described above.
Run SHA `6eece9c9c7af25d0e0cb3e2163400197d8492c5057e17abd25fbecef24f33004`.
Both complete BF16 output SHA
`f3d9b30a1e70f97a5510a0242006e91e0754207bb8e437cdf46d777bfb6fed33` and raw FP32 SHA
`71b486111bd9d21a3f3b09e9aa8966321aaa73df9873395bdb6ebf220ee141f4` equal the qualified
GB10 outputs. The unnormalized accumulator is unchanged from the preceding
replay, confirming that the final differences were in the denominator sum
and reciprocal endpoint. All host checks pass; process wall 7,792.119 ms.
The provider integration retains the default CK route, caps correction at
q8192 with eight queries per dispatch and a 20-second aggregate deadline,
prepares both immutable tables during provider initialization and releases
them with the provider. It loads no expected output tensor.


CK provider integration is independently qualified on baiying. Source
`b6094534d6eb440f4817404acc28c0ed351715ae` builds the 427,520-byte DLL SHA
`8b360e44584c7c0c595b2466a8b3324bff1aa645eb0b8f3c655aa862c3a049b0` in 33,759.702 ms,
all host checks passing. The external CK source manifest binds 5,350 source
files with SHA `412419e0747ae26380362f90a161070611d37a0666f4b7727edb33e71aa5e7d3`.
The complete local check suite passes before the build.

Command `D:\projects\test-native-ck-sm121-r2.ps1`, executor source
`6c959cddf1002a3cfa57ec2bd929d31db4bdd1f3`, calls the actual DLL ABI with the same
qualified q7169 layer-3 Q/K/V from `D:\models\Qwen3.6-35B-A3B`. With correction
enabled, all 29,364,224 BF16 and raw FP32 values equal GB10; the separate
terminal call also matches all 4,096 BF16 values. Run SHA
`dfa26091fdf02ad6e79f10b76eee18a957b507993b6b2f45244888cae482ccef`; provider call
6,847.94 ms, process 7,920.576 ms, all host checks passing. With correction
disabled, full BF16 SHA remains
`d8aac33b2097b01d47647b8fcb2d223193a07956bca36b1634ae4e6865a2914f`, identical to the
prior DLL, including its 65,059 known mismatches. Default-control run SHA
`a6b0f2aff89da915417fedc5f2fb5cc65ae6afd022ff8b3314f64f1e2c22eef5`.

The first enabled-ABI probe completed the provider call but the diagnostic
harness then rejected its aggregate time as if it were one GPU dispatch.
That failed record remains. Executor `6c959cd` distinguishes a provider call
(20-second bound) from an individual diagnostic dispatch (3-second bound).
The provider itself still submits at most eight queries per synchronized
kernel and enforces its aggregate deadline. These records establish component
integration; they do not establish full-model tokens, continuation or retained
performance. The exact arithmetic remains too slow for product acceptance.


The complete-model CK integration still fails the unchanged q7169 gate.
Command `D:\projects\prepare-fla-model-q7169-ck-sm121-r2.ps1`, SHA
`fe820a76560cbf4c13d6902ec30c4cfdf5b59b3a18d64813112b8a0f7a80ae14`, runs on baiying
with `D:\models\Qwen3.6-35B-A3B`; whole `28c4f9df`, FLA `831c1699`, MoE
`f164f0b0`, CLI `f544cbe` and CK `b6094534`. Run SHA
`7248f6263449c3af289052f70a78d7a2fad45c6558e0757a368bfc68e2d68421` binds the mixed
components and oracle. It emits 220 / 9.3125, rather than 82 / 9.25, with load
20,056.130600 ms, diagnostic TTFT 318,233.503800 ms and wall 338,728.882 ms.
All host checks pass. The first dispatch stopped before inference because its
expected-output filename collided with the preceding run; the retained file
was unchanged `[82]`, and the second dispatch uses independent paths.

Complete layer-3 normalized input, Q/K/V, normalized/rotated Q/K, attention
context, gated context, output projection and BF16 residual now match GB10
exactly. The next normalization has 50 differing BF16 cells across 47 tokens,
with none at the terminal position. Terminal residual layers 0–3 remain exact;
layer 4 has 608 differences. These are diagnostics, not acceptance by internal
hashes. Evidence: `fla-model-q7169-ck-sm121-r2/`.

CPU replay with the actual native residual/update and model normalization
weights matches all 14,682,112 GB10 postnorm values when using the characterized
FMA/reduction order and general SM121 reciprocal-root table. An all-FMA
scalar fold retains nine differences; host reciprocal root retains 89 even
with the correct fold. The exact CPU variant differs from the actual native
output in the same 50 cells. The next bounded native replay extracts the
current provider functions verbatim, checks its baseline against the model
capture, and separates reciprocal-root evaluation from endpoint contraction.
Reference tensors stay on the host. Whole-model continuation, retained speed
and release qualification remain open.


The next native control identifies a missing argument, not a new numerical
primitive. The full-attention caller loaded the configured SM121 reciprocal-root
correction but passed `nullptr` to `output_bf16_residual_postnorm_vllm_kernel`.
The two linear-attention call sites already forwarded the pointer. The first
standalone extraction used the pointer and unexpectedly matched GB10 while
differing from the actual model's 50 cells; its control rejection is preserved
as run `9a748af39f6a5a143475844d88e93dd0dff0b4bceeff58505369fe8e900e1ce0`.

Source `080d517d210280491536402248b9b8ff7ea9e42c` fixes the full-attention argument
and reports whether the correction is active. The functions themselves are
unchanged: extracted-header SHA remains
`7b44988bf981363f66534f1305bea86729af925806e01181bea73c82647d22ee`.
Command `D:\projects\run-native-postnorm-replay-r2.ps1 -Action replay`, host baiying,
uses captured real q7169 layer-3 inputs from `D:\models\Qwen3.6-35B-A3B`, the
actual weights and the same qualified GB10 postnorm reference. Run SHA
`b274ad53742eca3b3991ae81084dbe572a44335e85948e9f1cfb77407e9ead41`, process wall
669.280 ms, all host checks passing.

Its null-pointer control reproduces every prior native value, including all
50 GB10 differences. Forwarding the loaded four-MiB correction matches every
one of 14,682,112 reference BF16 values. Direct use of the independently
validated SM121 reciprocal-root table and explicit endpoint multiply controls
also match. All 7,169 raw sums, variances and both reciprocal-root variants
match the CPU replay bit-for-bit. The fix adds no table or runtime dependency.
The complete local check suite passes. This closes a component boundary;
full-model tokens and continuation still require the corrected provider build.


The corrected whole provider is built from `080d517d210280491536402248b9b8ff7ea9e42c`
on baiying in 78,808.702 ms, all host checks passing. DLL SHA
`e164182d895c7e19cad0aab586abd74a12b7caa99cf3dcfeb02442397d39c93f`, 11,462,656 bytes.
Command `D:\projects\prepare-fla-model-q7169-postnorm-wire-r1.ps1`, SHA
`93983689888996a5a5813adff4a5b20edd0aa330c21e25f7d863ab696f25a0bd`, uses
`D:\models\Qwen3.6-35B-A3B` with the unchanged FLA, MoE, CK and CLI components
from the preceding model run. Run SHA
`725a06551e14bc7c98c717b9f334cdf39c7c223d51423b550071b92f4fb56572`.

All eleven complete layer-3 attention boundaries, including post-attention
normalization, are now GB10-exact. Only the postnorm and subsequent MoE output
capture files changed; all earlier full-stage files retain their hashes.
Terminal residual layers 0–19 are exact, whereas layer 20 has 1,999 differences
and final layer 39 has 1,839. The final first token still fails at 220 / 9.3125.
Load is 20,062.162500 ms, diagnostic TTFT 319,214.462000 ms and process wall
339,693.604 ms. Host and cleanup checks pass. No continuation or performance
acceptance follows; the prepared 32-token command is not dispatched.

The next observation captures all 80 complete input/post-attention
normalizations, rather than inferring the earliest full-prefix difference from
a terminal row alone. GB10 additionally observes full attention at layer 19,
with the unchanged same-run 32-token and raw-logit qualification. Both observers
are opt-in, capped at three GiB; native copying uses at most one MiB per read
and 60 seconds of aggregate observer work, separate from bounded inference
time. The default single-layer observer keeps its original limits. The broad
native record will be compared with the preceding model's output and terminal
rows to check that observation did not change arithmetic.


The expanded GB10 capture is independently qualified. Source
`743bd42155d912353024024e14d1dab4d329a917`, command
`run-qrt-gb10-allnorm-20260911-r1.py`, host `aitopatom-66c4`, model
`/mnt/data/models/Qwen3.6-35B-A3B`, pinned original image. It records all 80
complete normalization boundaries and full attention at layer 19: 212 files,
2,826,802,176 bytes. Capture SHA
`f17592ae9d7d332386eb4b2a3f001f73e497aab94463588b4252e2f23eebdcc0`.
All 32 output tokens equal the unchanged oracle, the raw first token is
82 / 9.25, and final norm SHA remains
`be3354ef1cd706c10a5ee58ae4da7492389d1679cff89e02fa5a6e1229355483`.
Load 298.795621 seconds, observed request 15.839573 seconds, owned container
346.500576 seconds / exit 0, minimum host available memory 17,963,675,648 bytes.
These are GB10 reference timings, not Windows performance evidence.

Review of the native final layer shows `target_tokens=1` with 7,169 history
positions. The first all-layer observer run was explicitly stopped after
120,601.910 ms, before reaching that sparse final target. All host and cleanup
checks pass; the partial files remain diagnostic only. The revised observer
requires materialized rows to equal the requested full-prefix extent before
reading a full surface. This q7169 route therefore supplies 78 full
normalization boundaries (layers 0–38); the existing terminal trace covers
layer 39 and a structured marker explains its omitted full-prefix capture.
No model operator is replaced to create nonexistent last-layer rows.

The interrupted native run is bound to source `743bd42155d912353024024e14d1dab4d329a917`,
host `baiying`, model `D:\models\Qwen3.6-35B-A3B`, command
`prepare-fla-model-q7169-all-norm-r1.ps1`, and run SHA
`45fa9db93fee1b2028d224f00486ba3b92cd9fc13ecb7db622e08178075d8a0a`.
All 24 complete normalization files from layers 0–11 match the newly qualified
GB10 reference by SHA256. No output token was produced before the controlled
stop. The materialized-row guard passes `make check PYTHON=python3.12`:
268 Python tests (two existing skips), 45 Rust tests, Clippy, C ABI, seven q16
checks and public hygiene. Native rebuild and full-model observation follow.

The corrected observer completes on `baiying` with source
`e9a70123494b02a94efaf008556e42b20e0df448`, model
`D:\models\Qwen3.6-35B-A3B`, command
`prepare-fla-model-q7169-all-norm-r2.ps1` (SHA
`51fab79a33bd934f6783f717fb0e4ec626d382f2be043ff0c6169e419cac1871`).
Run SHA `0126d82befc7eade81db8b242d178d2834e8f94298fb2315d1fbf98ee157f991`.
The rebuilt whole DLL is `abc3508ec0defb917a06880fdf798411eac6b460c0730fb567038935a56555d9`;
FLA, MoE and CK remain the qualified component revisions documented above.
All host checks pass. The 78 requested complete files are present, and the
last-layer marker reports one materialized row instead of reading history-sized
storage. All 40 terminal FP32 rows are unchanged from the preceding model run.
The first token remains 220 / 9.3125, so inference acceptance still fails.
Load is 20,295.979900 ms, observed TTFT 318,605.831000 ms.

Against GB10 capture `f17592ae9d7d332386eb4b2a3f001f73e497aab94463588b4252e2f23eebdcc0`,
all normalization boundaries in layers 0–19 and the layer-20 input are exact.
The first changed surface is layer-20 post-attention normalization: positions
0–6289 remain exact, while all 879 later rows differ (1,696,051 BF16 elements,
relative L2 0.021853610). The final 37 normalization files differ downstream.
This localizes the cause before layer-20 MoE. The next paired observation
captures the original layer-20 projections, convolution inputs, gates, GDN
output, gated normalization and output projection. Its GB10 hooks are installed
after startup, preserve the existing operators, and require the unchanged
32-token/raw-logit oracle. The separate linear scope has a 1.5-GiB total bound.

The first MoE observer attempt at source 102b468 on GB10 loads the model but
stops before requesting tokens: its observer incorrectly required an external
router, while the pinned original runner invokes the same gate module inside
FusedMoE. No tensor files or qualified output are produced. Source inspection
of the original runner confirms the module alias and selector call. The
observer now supports that unchanged internal call and copies the selector's
actual output, shared intermediate tensors and expert output tuple, restoring
the selector afterward. The failed capture is retained as unqualified evidence.


The expanded original GB10 request at source
`11bedbec5b6c89fdf2598aaac83b70fc4a5a10e9` qualifies all 32 tokens and raw
82 / 9.25. Host `aitopatom-66c4`, model `/mnt/data/models/Qwen3.6-35B-A3B`,
command `run-qrt-gb10-moe26-20260911-r2.py` (SHA
`e72ab7c31b5a591426b5caf79ec0e960b36a12078b18b146a12b1cfefaa8eccd`).
Capture SHA `fb2e1a4c20b852b966c83c4853b0f75b0476fc0fd64650a0841ba2e9b8883c2c`:
168 files, 1,563,237,250 bytes; all 130 common terminal/normalization boundaries
match the previous qualified all-norm capture. Load 313.838127 seconds, observed
request 16.149033 seconds, owned container 361.343875 seconds / exit 0.
These reference timings do not qualify Windows performance.

Compared with native run `6352bd60fff19a92115fd7fd6a7a9dd9723d760399b90ca928655dd8c5be1d64`,
all full residual bytes agree (SHA
`0d247e004598abe4bc8d0140682ff0ea224085cb423f1cb99136697c188ca91b`). At position
946 all 256 router values, eight expert IDs and FP32 weights, 512 shared gate/up
and activation values, 2,048 shared-down values and 2,048 routed output values
are exact. Only the shared scalar gate differs: BF16 bfb4 versus bfb5. Its
rounded sigmoid is 0.197265625 instead of 0.1953125, changing all 2,048 shared
and 1,485 combined MoE values. Offline substitution reproduces the original
shared and total MoE outputs exactly. This does not substitute for a native
model rerun. The existing adjacent FP32 reduction lands on -1.41015625, exactly
the BF16 midpoint; the FP64 dot is -1.4101563433468982, beyond it.
`scripts/capture_sm121_shared_gate.py` now replays all 40 original CUDA gates
from qualified full-model inputs and requires the full layer-26 output control.
Its alternative CPU reductions are diagnostics, not correctness authority.


The shared-gate reference replay at source
`ae2efa56b0860a856b494d503e544f5b57f0d953`, host `aitopatom-66c4`, model
`/mnt/data/models/Qwen3.6-35B-A3B`, command
`run-qrt-gb10-shared-gate-20260911-r1.py` (SHA
`d1572a01c0464b2725a481056418ee73e918fad8403c9e204dd042ce05cbe7f6`)
qualifies the full layer-26 gate against the original model capture and saves
all 40 reference gate vectors and actual model weights. Capture SHA
`af3f4e9939b3330101177d03ae43dc6eec82438b0e03e87e1e5784009bf8861a`.
Owned container 5.538974 seconds / exit 0; peak device allocation 37,917,696
bytes. These are component timings, not TTFT. The native adjacent tree differs
at five positions in layers 26/39; FP64 differs at three in layers 23/39.

The actual profiler records a 16-by-4-thread cuBLAS `internal::gemvx` kernel.
Its installed SM120 SASS confirms sequential FP32 multiply-adds and halving
lane reduction. Library SHA
`f6297579dfc8ada7869639aaa0b5e1de43adc749dd0dea1b747b043e127382c6`.
An independent CPU replay of sixteen strided folds and offsets 8/4/2/1
produces all 286,760 original BF16 gates exactly. Native shared gating now uses
that order with four independent rows per block. The six observed midpoint
and cancellation cases are retained as source-bound numerical regression
fixtures. A rebuilt native full-model run is required before inference or
performance acceptance; the first-token gate is still open.


MoE source `023e5dc6baa74ee2b0e57a58826adeb363cee938` passes the full local
check (269 Python tests, two existing skips; Rust/Clippy, C ABI, q16 and hygiene)
and the bounded native build. Host baiying, command
`run-native-moe-gate16-build-r1.ps1`, build run SHA
`50de2c8b1746a668acb420679ce1ddd5c44c41b202347e33bdbbb8807656b798`,
16,875.896 ms, all host checks pass. DLL SHA
`013a8ad9085d4a139db47a80bff817e3ae19834a63dc384d43cd5ce0750da1be`,
513,536 bytes. Async parity and all 32 dynamic logical cases pass. The old
synthetic self-hash changes and remains diagnostic. The real q7169 run uses
`prepare-fla-model-q7169-gate16-r1.ps1` (SHA
`9bd34000b184691a8343e2c0866d8ac12bf2fc435ea37056963b180fb8f1285f`),
retaining whole e9a7012 / FLA 2f346df / CK b609453 / CLI f544cbe.

The preceding complete model trace measures 1,420,778 synchronized exact-dot
dispatches, with about 1,000 candidates each. The compacted scheduler now allows
an explicit cap up to one full 65,536-candidate window (4,096 CTAs). The default
remains eight CTAs, each still owns at most sixteen dots, scratch remains
512 KiB plus counters, and the 100-ms dispatch / 10-second correction deadlines
remain active. Host execution of the actual launcher verifies sparse/dense
windows, partial tails, unique index coverage and cleanup across caps
8/64/999/4096/UINT32_MAX. Native numerical and product timing checks must follow
before this scheduling option is retained as a performance improvement.


The shared-gate full model completes with all host checks passing: run SHA
`c2bcd3936abb3479c74c1667408a1fc9902911b9fdb8c01c332c24d56db0be1b`,
105 files / 2,997,564,232 bytes, process wall 339,481.310 ms. All 78 full norms
match capture f17592ae; all 39 preceding terminal residuals are exact. Its
remaining layer-39 difference does not qualify inference. The expanded final
layer path retains the requested terminal output selection and its complete
KV history while selecting full-prefix computation internally. The component
provider and normalization observers can therefore cover the final layer too.
