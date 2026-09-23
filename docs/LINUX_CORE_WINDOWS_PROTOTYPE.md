# Linux native core Windows experiment

The optional native GDN path now uses unsigned 32-bit accumulation throughout
the complete original-order integer pipeline. Both this version and its 64-bit
control match every core output and final FP32 state across all30 GDN layers of
the original q8192 model request. The comparison returns original outputs to
the model and preserves all576 control/continuation tokens. The complete q7169
chain and host launch/state checks also pass on CUDA. Actual AMD execution
now rejects the unsigned candidate: repeated identical inputs give different
incorrect W results, and both barrier/two-warp controls fail. The 64-bit control
passes the native complete-state comparison and real q8192/out512 model gate,
but TTFT regresses to 40680.6294 ms. The option remains off, with no retained
performance improvement. CUDA equality did not establish AMD correctness.
[Pipeline scope and evidence](NATIVE_GDN_ORDERED_COMPONENTS.md).

This standalone experiment ports the compute core declared by Linux release
`v1.5.1-native-vl.10`, source `ec9934446911fdf376da8eebcd83e7b137efbb7c`.
It is not enabled in the Windows product. The fastest qualified observation,
`21e3234` passes the complete cold q8192/out512 correctness boundary on baiying
with the real model: all512 tokens and callbacks match GB10, with
first144/logit10.375. Loading is27593.4027ms, TTFT20086.8502ms and
TPOT226.471535ms with detailed projection events and timeline disabled.
The earlier `9447947` run also has all72 second-decode
comparisons matching in BF16 and a bitwise-exact final normalization.
Performance, other product shapes, prefix continuation and release remain open.

The preceding [ordered GDN components](NATIVE_GDN_ORDERED_COMPONENTS.md) explicitly
preserve the original inverse reduction and BF16 U accumulator. Their CUDA checks
match47,425,536 original BF16 values across the current first chunk, historical
complete q7169 and tail boundaries, plus262,144 characterized FP32 U values.
Both compile for gfx1151 and now form part of the complete optional pipeline.
The complete 64-bit AMD path passes numerically; the current unsigned replacement fails. The qualified prototype configuration remains unchanged.

The profile-off control preserves the qualified compute settings and the
original full GB10 checker. Its single TTFT observation is368.2925ms below the
instrumented `0c80a89` result; no paired repetitions establish that difference
as a stable speedup. Both terminal paths and tuned input GEMM choices remain
active. All60 Windows build units,72 imported images and host guards pass.
[Profile-off model result](../benchmarks/correctness/linux-core-profile-off-native-20260923.json).

An optional `AIMA_PORT_PREFILL_BATCH_REPLAY=1` route submits independent dense
projection windows in one two-dimensional selector grid followed by one replay
grid. Each window retains its own counter and full worst-case queue extent;
the admission predicate, original ascending-K16 carry and BF16 endpoint are
unchanged. The extra400556416 device bytes trade memory for fewer submissions:
the preceding q8192 profile contains4284 windows across189 projections. Actual
post-READY submission counts are emitted once on owner teardown. ASan/UBSan
checks empty/full/partial windows, queue addressing,97-window dispatch and221
invalid bindings. All17 generated overlays and72 imported images match the
control.
[Batch replay preparation](../benchmarks/correctness/linux-core-batch-replay-preparation-20260923.json).

The real `21e3234` trial verifies189 selector/replay pairs covering4284 windows,
with all512 GB10 tokens/callbacks and first144/logit10.375 matching. Native build
and cleanup pass. Its TTFT is202.2684ms below the unbatched profile-off control;
that small unpaired difference does not establish a stable retained speedup
for the extra memory. The option provides a qualified basis for broader replay
layout work. Performance and release targets remain unmet.
[Native batch replay result](../benchmarks/correctness/linux-core-batch-replay-native-20260923.json).

The optional `AIMA_PORT_PREFILL_GROUP_MAJOR_WEIGHTS=1` layout stores the
same prepared weight rows with the K16 group outermost. Nearby candidates can
then read neighboring output rows within a smaller region, while each dot
still consumes the original groups in ascending order. The input layout,
scaled-half values, carry arithmetic and allocation size remain unchanged.
Host checks compare both source views, all four operand lanes, tail guards,
and both sequential/batched dispatch.
[Group-major weight preparation](../benchmarks/correctness/linux-core-group-major-preparation-20260923.json).

Source `747bb00` completes the real q8192/out512 test with the new layout and
all189 batch pairs verified. All512 tokens and callbacks, first144/logit10.375,
Windows build and cleanup pass. Load is27473.2794ms, but TTFT regresses to
24138.978ms,4052.1278ms above the batched row-major control. TPOT is229.758258ms.
The layout remains disabled and is rejected for retained performance. Fewer
submissions and this alternate layout have not removed the multi-second
arithmetic cost; the 10-second and retained targets remain open.
[Group-major native result](../benchmarks/correctness/linux-core-group-major-native-20260923.json).

Native GDN observation `fce7fdd` reproduces all 512 outputs from the failed
chunk32 experiment, including 486 differences from GB10 and the first at
index 1 (244 versus 255). Layer0 input normalization and QKV/Z/A/B projections
are bitwise exact. The prefill core differs at 11467 of32768 sampled BF16
values, and its complete final state differs at443872 of524288 FP32 values
(maximum absolute error0.09027356). Transposing each state matrix increases
the error; a layout transpose does not explain the failure. The first decode
receives this same incorrect state while its input projections and convolution
remain exact. Instrumented TTFT20820.2864ms is diagnostic only.

The next opt-in native GDN experiment replaces the fused preparation with
the qualified XOR16 Q/K normalization, original BF16 beta and FP32 decay
table. It reuses six already embedded dynamic-T chunk64 kernels, with live
q8192 tensors and96MiB of owner-managed A/Ai scratch. Persistent invocation
bindings remain unchanged; inverse storage and cold initial state are cleared
at their semantic lifetime boundaries. CPU replay matches66048 original
operand values, including32768 normalized Q/K values checked against the
original vLLM normalization kernel on GB10. The initial raw Q/K capture is
before normalization and is not a comparable normalized boundary. Host
ASan/UBSan checks12480 conversion values and65 invalid preparation bindings.
[GDN diagnosis and chunk64 preparation](../benchmarks/correctness/linux-core-native-gdn-boundary-and-chunk64-20260923.json).

Source `01659ba` builds all60 native units and completes the real-model
q8192/out512 run with all30 chunk64 activations. Actual GPU Q/K normalization
and V match the original eight sampled positions bitwise, but the core still
differs at5039 of32768 BF16 values. Final state differs at408681 of524288
FP32 values, maximum absolute error0.049892806. The first continuation
divergence moves from index1 to115 (196 versus271), with390 output differences.
First144/logit10.375 is exact. Load27579.1347ms, instrumented TTFT21260.3428ms
and TPOT222.559631ms are diagnostic only. The remaining block arithmetic
requires correction; the chunk64 route is not qualified or retained for speed.
[Native chunk64 GDN result](../benchmarks/correctness/linux-core-native-gdn-chunk64-native-20260923.json).

The W/U follow-up preserves the original BF16 rounding of K times beta before
multiplying FP32 decay. The imported FP32 beta interface had removed that
intermediate rounding. One corrected gfx1151 module is embedded separately
from the unchanged 72 imported COFF images; its image is 119768 bytes. Offline
compilation uses the pinned Triton 3.6.0 environment with GPU access disabled.
Windows gains no Python or Triton dependency. The existing 96MiB matrix scratch
is reused, and host ASan/UBSan verifies 49 rejected W/U bindings plus the actual
AOT module ABI, image identity and launch geometry. GPU arithmetic and the
complete GB10 continuation gate are evaluated separately from these host checks.
[W/U repair preparation](../benchmarks/correctness/linux-core-native-gdn-wu-preparation-20260923.json).

Source `867d528` completes the real q8192/out512 trial but fails 395 tokens,
first at index 2 (220 versus 82). The first token and logit remain 144/10.375.
The W/U correction reduces sampled layer0 core differences from 5039 to 3370
and maximum final-state error from 0.049892806 to 0.021411896; 401699 FP32
state values still differ. Original input projections and normalized Q/K/V
remain exact at the eight sampled positions. Load is 27483.8565ms, diagnostic
TTFT 20500.7038ms and TPOT 228.230562ms. All 60 build units, 72 imported images,
the separately embedded W/U module, 30 GDN activations and cleanup complete.
The route remains unqualified; earlier output divergence despite smaller
layer0 errors requires further stage comparisons against original operands.
[Native W/U correction result](../benchmarks/correctness/linux-core-native-gdn-wu-native-20260923.json).

An isolated replay of the original first64 layer0 operands reproduces every
one of the 262144 original model-core BF16 values on GB10 using its frozen
kernel choices. On baiying the same-image native chain's last row also matches
the actual q8192 product's first chunk endpoint. Cumsum is exact, but identical
original operands still expose independent differences in KKT, inverse, W/U,
state and output. Isolated inverse differs at 8 BF16 values, U at 32, output
at 13, and state at 256249 FP32 values (124 BF16 endpoints). Corrected W differs
at 883 values, down from 93279 with the imported uncorrected image. These are
operator diagnostics; the full-model continuation boundary remains unchanged.
[Original first64 stage comparisons](../benchmarks/correctness/linux-core-native-gdn-firstchunk-stages-20260923.json).

Compiling all six original stages with GB10's frozen tile choices does not
remove the arithmetic differences. A direct BF16 beta ABI also exposes RTZ
products on AMD: all4096 first-row U values match truncation, while GB10 matches
RNE and1916 values distinguish the two. Restoring FP32 multiplication and the
explicit BF16 W boundary avoids that failure. The aligned batch still differs
at119 core values, versus133 previously, and leaves independent state/output
errors. These experimental images are not selected for the runtime. The
qualified GDN path remains the basis for further product performance work.
[Aligned block and product-rounding diagnosis](../benchmarks/correctness/linux-core-native-gdn-aligned-blocks-20260923.json).

Native MoE correction `f210ff3` also passes the original cold q8192/out512
boundary: all 512 tokens/callbacks, first 144 and logit 10.375 (zero error).
It replaces the expert gate/up and weighted-down AOT products with FP32 WMMA
and selective original SM121 replay before the BF16 endpoints. The FP32
routing weight is applied before down selection and rounding. All 39 native
MoE layers, 78 routed projections, 345 dense projections and both terminal
paths execute. Loading is 27826.9393 ms, TTFT 24282.2874 ms and TPOT
138.061143 ms. The 394-output continuation failure described below is repaired
for this case; the slower TTFT is not retained as a performance improvement.
The opt-in route adds 1241104128 device workspace bytes and no runtime artifact.
Host guards and cleanup pass. A local ENOSPC interrupted build receipt writing;
the same remote build was verified and resumed without recompilation.
[Qualified native MoE replay result](../benchmarks/correctness/linux-core-native-moe-replay-native-20260923.json).

The preceding output-only observation reproduces all 512 failed outputs.
Seventeen upstream/shared/router surfaces match the original GB10 capture
bitwise. Expert gate/up first differs at 20632 sampled BF16 values, including
seven non-tiny pairs; five numerical activation differences and 877 weighted
down differences remain. The next normalized carrier differs at 41 values.
Both expert products now use the corrected producer/replay pipeline.
[Boundary diagnosis and correction preparation](../benchmarks/correctness/linux-core-native-moe-boundary-and-replay-20260923.json).

The original optional comparison, `AIMA_PORT_NATIVE_MOE_PREFILL=1`, replaced
layers0..38 of the approximately4927.5989ms MoE wall with imported expert
GEMMs. It uses the existing FP32-routing-weight images with q8192 live grids,
four corrected dense projections, the original scalar gate reduction,
BF16 SiLU tables, FP32 routing, and the unrounded residual carrier. Layer39
retains the qualified terminal provider, and native GDN remains disabled.
Direct CPU replay of original GB10 operands matches all eight pointwise and
routing surfaces across three layers/eight positions, including all12288
shared and98304 routed activations. FP32 SiLU before the up product fails this
same reference; both branches require the BF16 table endpoint. ASan/UBSan
checks complete native layer lifetimes, borrowed table owners and rejection
of incomplete or invalid device state. Only the MoE overlay changes; the other
16 overlays and all327 imported files remain identical.
[Native MoE preparation](../benchmarks/correctness/linux-core-native-moe-preparation-20260923.json).

Source `2cbd9a4` builds all60 units and completes the native q8192/out512 run,
but394 tokens differ from GB10, first at index115 (196 versus271).
First144/logit10.3125 is within the unchanged0.125 tolerance. Diagnostic
TTFT is19009.101ms, loading27531.8107ms and TPOT230.662252ms. All39 native MoE
markers,345 projections and both terminal paths execute. MoE wall is3215.4904ms;
linear8696.9268ms and full attention7090.3181ms remain close to the qualified
baseline. Eighty-nine projection calls report invalid HIP elapsed intervals,
so GPU stage sums remain diagnostic. The continuation failure rejects this
route despite the shorter independently measured TTFT.

The follow-up adds only three output observation calls: expert gate/up,
post-activation and weighted down. Sampling now accepts the full16384-channel
weighted row, with a4MiB gather into the existing128MiB scratch. ASan/UBSan
checks2199680 sampled values, row strides, guard bytes and unchanged inputs.
All arithmetic is unchanged. The next diagnostic compares real layer0
operands with the existing original GB10 capture; its timings are not product
performance evidence.
[Native MoE failure and observation preparation](../benchmarks/correctness/linux-core-native-moe-negative-and-observation-20260923.json).

Source `a56baa9` omits `--gb10-prefill-projections` and uses the imported
hipBLASLt BF16 dense producer while retaining every other repair. It builds
all59 units and completes the native request, but385 of512 tokens differ,
starting at index115 (196 versus271). First144/logit10.4375 is within tolerance;
that alone does not qualify the continuation. Loading is27261.5664ms,
diagnostic TTFT14933.2741ms and TPOT232.780459ms. The BF16-only route is not
retained. Its removal of598360324bytes of scratch and the whole preparation,
selection and replay pipeline identifies a substantial cost to investigate.
[Native correctness and dense-prefill preparation](../benchmarks/correctness/linux-core-decode-gated-native-and-bf16-prefill-preparation-20260922.json).

The exact-replay route now has optional completed GPU profiling, enabled by
`AIMA_PORT_PREFILL_PROJECTION_PROFILE=1` and armed after READY. It reports
producer, operand preparation, norm-bound, selection and exact-replay times
for each whole GEMM, plus candidate counts. The owner allocates295 events and
a388-byte count array before READY, bounded to97 windows; counts are read
only after completion and do not control arithmetic. Host ASan/UBSan checks
cover maximum bounds, warmup exclusion, event lifetime and14 invalid bindings.
Preparation preserves327 imports,60 units,72 images and17 overlays. Source
`8ebdba0` completes all190 projection measurements and passes all512 GB10
outputs/callbacks and first144/logit10.375. Loading is27504.329ms,
diagnostic TTFT31209.5764ms and TPOT234.052419ms. Completed projection work
totals16713.466615ms: producer4026.57947, operands189.366, norm bounds161.52025,
selection498.05741 and exact replay11789.119477ms. The40 K4096 OUT calls
alone replay382466673 of671088640 cells and consume8863.161088ms in replay.
These timings include diagnostic overhead and do not meet performance goals.
[BF16-only native failure and profiling preparation](../benchmarks/correctness/linux-core-bf16-prefill-native-and-profile-preparation-20260922.json).

Source `42c7983` enables `AIMA_PORT_PREFILL_WMMA=1` for the
existing M64/N128/K16 matrix producer across all eligible dense projections,
and `AIMA_PORT_PREFILL_LINEAR_BOUND=1` for the original1000-ppb linear OUT
selector. The actual linear call site owns that scope; full-attention K4096
keeps10000ppb. The prior port incorrectly used the full-attention bound for
both kinds. The512-radius selection and complete exact replay remain.
ASan/UBSan checks all30 linear layer scopes, exception unwinding, maximum
profile bounds and107 invalid bindings. Preparation passes with the same
60 units and no new dependency or allocation. The native result above lowers
observed TTFT4693.0508ms. Linear OUT replay drops6384.800812 to1321.111019ms,
with58490373 candidates. Input-projection producers instead increase from
2976.078555 to3995.487975ms. Seven short selection aggregates are negative
HIP elapsed readings; raw values are preserved as diagnostic anomalies.
The independently measured wall time and unchanged GB10 token gate pass.
[Completed projection profile and replacement preparation](../benchmarks/correctness/linux-core-prefill-profile-and-wmma-preparation-20260922.json).

Source `cbe133b` keeps WMMA only for K4096 OUT
(`AIMA_PORT_PREFILL_WMMA_OUTPUT_ONLY=1`), uses hipBLASLt for inputs and enables
`AIMA_PORT_PREFILL_FULL_COARSE=1` at the10 actual full-attention OUT call sites.
The existing vector/domain C64 producer computes a center and interval;
ambiguous BF16 endpoints and ineligible rows retain complete original replay.
It adds67108864bytes of preallocated error scratch, reusing dead norm buffers
for eligibility. The native coefficient remains2^-19; no universal hardware
error proof is claimed. Existing Windows DPP integer transport and compact
canonical normalization are also enabled in the imported SM121 backend.
Host ASan/UBSan checks30 linear and10 full scopes, interval edges, event
lifetime and171 invalid bindings. Existing arithmetic checks pass4194304
normalization cases and the original coarse-bound suite. Native preparation
preserves327 imports,60 units and72 images, binding419 source inputs. Profiling
now flags every negative elapsed interval without clamping it. The native
run passes all512 outputs/callbacks and first144/logit10.375. All10 full OUT
and30 linear OUT scopes execute. TTFT falls2773.2888ms from the preceding
run. Full OUT candidates fall to39589439 and diagnostic replay to826.95183ms.
Seventy-three elapsed intervals across38 calls are flagged; stage sums remain
diagnostic. The native request/callback wall clocks are separate and valid.
[Native WMMA result and full-OUT preparation](../benchmarks/correctness/linux-core-prefill-wmma-native-and-coarse-preparation-20260922.json).

Source `3e091e4` enables `AIMA_PORT_NATIVE_ATTENTION_PREFILL=1` for the
already resident embedded `kernel_unified_attention_2d` plan on ordinary
cold q8192 text. It uses the same contiguous normalized Q and resident K/V,
existing metadata and dead FP32 scratch as its BF16 destination, then the
existing BF16-input sigmoid gate. It applies only to8192 unpadded queries
at cache position zero without M-RoPE; other routing and decode stay as before.
Each actual invocation reports its layer, query/KV geometry and embedded
kernel identity. No artifact or allocation is added. Prepared-source checks
preserve327 imports,60 units and72 images. This arithmetic replacement needs
its own complete GB10 continuation result before any runtime gain is retained.
The native build passes and all10 actual activations are verified, but395 of512
outputs differ from GB10, first at index115 (196 versus271). First144/logit10.375
is exact. Loading27428.1791ms, diagnostic TTFT19712.6649ms and TPOT226.616026ms
do not qualify this route. Source `cbe133b` remains the correct experimental
baseline. Source `9af3371` disables native-text attention and enables the CK
completed-stage observer. All512 outputs/callbacks and first144/logit10.375
pass. Load is27470.2248ms; instrumented TTFT23938.4993ms and TPOT230.002446ms
are diagnostic. Ten completed attention calls total5889.253ms: QK2507.5478,
fused probability/native-PV1556.654 and exact-PV1668.4516ms dominate. Dense
projection profiles total9333.048755ms, including4693.320469ms replay;26
calls report invalid GPU-event intervals, so these stage sums remain diagnostic.
The prior source `cbe133b` remains the performance comparison baseline.
[Native attention rejection and CK profiling](../benchmarks/correctness/linux-core-native-attention-negative-and-ck-profile-20260922.json).
[Coarse native result and embedded-attention preparation](../benchmarks/correctness/linux-core-prefill-coarse-native-and-attention-preparation-20260922.json).

The next experiment, `AIMA_PORT_PREFILL_TERMINAL_ONLY=1`, preserves complete
layer39 input projections, head normalization, RoPE and resident K/V writes,
then computes only row8191 through attention, gate, OUT, residual norm and MoE.
The resident engine's final hidden consumer selects that row; no later layer
reads the other outputs. The attention helper reuses the original CK terminal
prefill arithmetic, including its non-FMA denominator, and borrows existing
exp2/reciprocal tables and scratch. OUT uses the original SM121 vector kernel;
MoE uses the pinned provider's dynamic ABI with one logical token. Its FP32
carrier remains at row8191 for terminal normalization. No dependency, artifact
or device allocation is added. Cold q8192 text and disabled tensor observers
are required. Local ASan/UBSan
checks pass68 attention rejection controls, two ordered40-layer MoE requests,
the one-row pointer bindings, carrier ownership and135 MoE rejection controls.
[CK result and terminal preparation](../benchmarks/correctness/linux-core-ck-profile-and-terminal-preparation-20260922.json).

Source `00d34c5` passes native compilation and the complete512-token/callback
GB10 boundary with first144/logit10.375. Both terminal activation records
execute, and the projection profile contains189 calls with nine full OUTs.
Load is27457.7687ms, TTFT23062.2803ms and TPOT229.610074ms. This observation
is680.9565ms below the preceding uninstrumented CK baseline; no paired-repeat
estimate is claimed. Final-layer attention falls to136.3701ms and MoE to
24.6663ms. Projection diagnostics still total9299.884187ms, including
3807.777714ms producer and4628.0469ms replay, with14 flagged event-timing calls.
The terminal route is retained for subsequent cold-q8192 experiments; it
does not qualify prefix, other contexts, the10-second target or release.
[Complete terminal result](../benchmarks/correctness/linux-core-terminal-prefill-native-20260922.json).

The standalone `linux_core_gemm_algorithms` diagnostic compares up to32
hipBLASLt heuristics at the actual q8192 N8192/K2048, N4096/K2048 and
N2048/K4096 shapes. Descriptors and the128MiB workspace limit match the port.
BF16 inputs use small exact rational values, allowing an independent integer
reference for every FP32 output, plus unchanged input and guard checks before
and after repeated launches. The initial exact-equality diagnostic rejects
the first algorithm:50649664 of67108864 cells differ, with no input or guard
changes; sampled errors are around1e-6. Source2141af2 preserves those values.
The revised diagnostic adds a complete independent CPU check of generated
input bytes and records every nonexact cell plus maximum absolute error.
Producer admission uses the existing input selector's1000-ppb L2 radius;
that component criterion is not a universal arithmetic proof. Injected output
and guard corruptions test the observer. Completed steady host timing includes
stream drains. This is a
component probe; an algorithm change still requires the full real-model GB10
boundary and load/TTFT accounting. The diagnostic adds no runtime dependency
or package artifact. Source25ea284 completes all27 candidates (nine per shape).
Complete CPU input controls pass; no output exceeds the selector radius and
all inputs/guards remain unchanged. Raw FP32 differences are retained, with
maximum absolute error4.08291817e-6 at K2048 and7.86781311e-6 at K4096.
Heuristic4/solution5651 is fastest in all three shapes: median11.7862 versus
49.373ms for N8192/K2048,5.0387 versus25.448ms for N4096/K2048 and5.0589
versus24.5016ms for N2048/K4096. Each timing has three completed host samples;
these controls are not model performance. A metadata-transfer timeout after
the completed build was recovered by reusing the exact same binary.

`AIMA_PORT_PREFILL_GEMM_TUNED=1` enables an opt-in model trial using
that candidate on those three contiguous-weight shapes. It verifies library
version100100, the complete24-byte algorithm identity, heuristic ordinal and
zero workspace requirement before READY. Changed identities fail closed.
With all three shapes enabled, WMMA overrides must be disabled; profiles identify the
selected producer. Existing exact replay, input/linear bounds, nine coarse
full OUTs and terminal pruning remain. No artifact or allocation is added.
ASan/UBSan checks cover the three shapes, excluded shapes and all24 one-byte
algorithm mutations alongside existing compaction/scope contracts.
[Native algorithm sweep and model preparation](../benchmarks/correctness/linux-core-gemm-algorithms-and-product-preparation-20260922.json).

The native trial at `558a3e7` builds all60 units and completes q8192/out512,
with100 actual tuned projections and both terminal activations verified.
It fails the original GB10 continuation observer:475 outputs differ, starting
at index4 (220 versus79). First144/logit10.375 and all512 callbacks match
the actual result. Load28601.6745ms, TTFT20186.3439ms and TPOT224.660759ms
are rejected performance observations. Of189 projection profiles,51 flag
invalid HIP event intervals; stage sums remain diagnostic. The qualified
terminal route remains the correctness baseline.

`AIMA_PORT_PREFILL_GEMM_TUNED_INPUT_ONLY=1` now isolates the two K2048 input
shapes while restoring the qualified WMMA linear OUT producer. It requires
tuned selection; combining it with WMMA requires both `AIMA_PORT_PREFILL_WMMA=1`
and `AIMA_PORT_PREFILL_WMMA_OUTPUT_ONLY=1`, keeping their scopes disjoint.
The trial expects70 tuned input projections,30 WMMA linear OUTs and nine
coarse full OUTs. Selector bounds and the full512-token GB10 gate are unchanged.
All16 producer configurations and180 rejection controls pass ASan/UBSan;
all327 imported files and17 generated overlays remain unchanged. This is
preparation, and neither projection family is yet identified as the cause.
[Failed combined trial and input-only preparation](../benchmarks/correctness/linux-core-tuned-gemm-negative-and-input-preparation-20260923.json).

The input-only run at `0c80a89` now passes the original q8192/out512 observer:
all512 tokens and callbacks match, first144/logit10.375 has zero error. The
actual70 tuned inputs,30 WMMA linear OUTs, nine coarse full OUTs and both
terminal activations pass. Load27608.3814ms, TTFT20657.4111ms and
TPOT228.040989ms are one correctness-attached observation. The TTFT reduction
of2404.8692ms versus `00d34c5` is not a paired estimate and remains above10s.
Projection GPU totals are producer1581.862139ms and replay4692.447539ms;
29 calls flag invalid event intervals. Whole-wall linear attention is
8635.9538ms, full attention7087.1113ms and MoE4927.5989ms. Input-only tuning
is retained for further cold-q8192 trials; other contexts and release stay open.
[Qualified input-only native result](../benchmarks/correctness/linux-core-tuned-input-native-20260923.json).

The isolated trial selects `AIMA_PORT_NATIVE_GDN_PREFILL=1`. It restores
the imported seven-stage post-convolution GDN block, including original
inverse scratch clearing and cold state zeroing. The generated block matches
the pinned source byte-for-byte except indentation. Its final state continues
to bind directly to the same resident vLLM state; no transpose is introduced.
The qualified convolution, input/output projections, gated norm, attention,
MoE and decode repairs remain selected. The existing FLA owner and arithmetic
tables remain loaded, so no dependency, artifact or device allocation is added.
ASan/UBSan checks reject11 invalid scope/configuration/provider bindings.
Only the linear-prefill overlay changes; native qualification must verify
30 actual seven-stage activations and the complete original512-token boundary.
[Native GDN preparation](../benchmarks/correctness/linux-core-native-gdn-preparation-20260923.json).

Source `3bd7969` builds all60 units and executes all30 native GDN layers,
with the70 tuned inputs,30 WMMA OUTs, nine coarse OUTs and terminal activations
verified. It fails the original continuation observer at output1 (244 versus255),
with486 total differences. First144/logit10.375 and512 actual callbacks pass.
Load27499.1025ms, TTFT19730.6036ms and TPOT226.338971ms are diagnostic only;
37 projection calls report invalid event intervals. Native GDN prefill is not
retained. Subsequent trials restore the qualified FLA path and keep the
`0c80a89` input-only baseline. Existing fused GDN state/output controls also
already passed correctness but increased whole-model TTFT by549.8157ms in
their original pair, so no gain is inferred from that flag.
[Native GDN failure and prior fusion review](../benchmarks/correctness/linux-core-native-gdn-negative-20260923.json).

The preceding complete decode-MoE repair (`5585977`) runs the real model on
baiying. All 40 first-decode layer carriers
match the original after BF16 rounding, and final normalization matches exactly.
All 69 comparisons pass, including the complete full-layer3 path and its cached
K/V. First144/logit10.375 passes; continuation still differs at index3 and in
471 of512 tokens. Second-decode observation isolates an unwired gated
normalization replacement at layer12. Correctness and release remain open.

The MoE-repair run builds all60 native units, loads in27332.5239ms, and measures
diagnostic TTFT30929.1631ms and TPOT231.562583ms. All67 captures and host checks
pass. These timings are not accepted performance. The first-step observations
do not qualify subsequent recurrent updates or the full continuation.
[Native complete decode-MoE result](../benchmarks/correctness/linux-core-decode-moe-native-20260922.json).

Two unchanged-source runs observe second decode at layers0 and12. Layer0 is
exact through all captured stages and both recurrent states. Layer12's prefill
state, both decode states and stages through recurrent output also match
bitwise. Its gated normalization first differs at one BF16 value, column3765;
projection19, post normalization4 and layer carrier177 values then differ.
The current decode branch still called the imported AOT gated kernel despite
the existing GB10 short-row helper. Original arithmetic reproduces all245760
values across30 layers/two rows and the native layer12 operands. The overlay
now calls that helper and updates launch accounting, adding no allocation or
artifact. Host binding/artifact checks and generated-source preparation pass;
native execution of this binding change subsequently passes above.
[Decode gated diagnosis and preparation](../benchmarks/correctness/linux-core-decode-gated-diagnosis-and-preparation-20260922.json).

The attention-repair run builds all59 native units, loads in27430.1645ms,
and measures diagnostic TTFT30923.1363ms and TPOT84.6361949ms. All67 observed
files and host checks pass. Original arithmetic applied to native operands
finds4 of8 different FP32 router weights, with identical expert choices;
the checked activation, sum and shared-scale operations agree. This does not
separately qualify native expert projections.
[Native attention result](../benchmarks/correctness/linux-core-decode-attention-native-20260922.json).

The optional `AIMA_PORT_DECODE_MOE=1` replacement uses existing SM121 routing,
shared/expert projections, activations and sums on the engine's real buffers.
It keeps separate gate/up model weights, borrows the GDN sigmoid table, and
loads the existing original SiLU/router-exp artifacts before READY. Ordered
layers0–39 accumulate a device error flag, checked before publishing each token.
The last layer snapshots its two BF16 operands so final RMSNorm uses the
unrounded FP32 residual sum for variance. ASan/UBSan checks80 layer bindings,
two complete terminal lifetimes and85 rejection/failure cases. The native result
above verifies its first decode step; complete continuation still fails.
[MoE preparation](../benchmarks/correctness/linux-core-decode-moe-preparation-20260922.json).

The preceding head-repair run loads in27227.3378ms, with diagnostic TTFT30973.0705ms
and TPOT67.3512951ms. Its output-only observer captures67 verified files;
host checks and cleanup pass. These timings are not accepted performance.
[Native head repair and attention isolation](../benchmarks/correctness/linux-core-full-head-native-and-decode-attention-preparation-20260922.json).

With `--gb10-normalization`, the optional `AIMA_PORT_DECODE_ATTENTION=1`
owner reads the actual resident token-major K/V planes and compact Q row.
It reuses existing SM121 K16 QK and original Q2 online softmax/PV kernels,
including the denominator's explicit FMA. It borrows the GDN owner's exp2
table and SHA-validates `AIMA_PORT_ATTENTION_RCP_TABLE` before READY.
The default-stream scratch is allocated once; no history copy, request-time
allocation or reference activation input is introduced. Five host dispatch
bindings and38 rejection/failure controls pass ASan/UBSan. These checks do
not execute GPU attention arithmetic; the subsequent08e5694 run above verifies
this attention path against the original, with remaining MoE differences.

The latest sampler preserves all 512 outputs and all 71 earlier observations.
Across 128 fixed prefill rows, three of 262,144 input-normalization values
differ; they belong to the same actual token ID 36. Reapplying the existing
full-vocabulary embedding inverse table removes all three differences. With
identical input, SM121 K16 arithmetic also matches all 12,352 first-decode
QKV/Z/A/B reference values, while a CPU replay of Linux wvSplitK reproduces
the native nine differences. Six sampled prefill rows separately reproduce
all reference QKV values with the SM121 accumulator and original input.
[Arithmetic diagnosis](../benchmarks/correctness/linux-core-projection-arithmetic-diagnosis-20260922.json),
[prefill and preparation evidence](../benchmarks/correctness/linux-core-gb10-projection-preparation-20260922.json).

`--gb10-projections` now opts into that singleton projection arithmetic and
first-layer embedding normalization on top of `--gb10-gdn`. Prefill dense
GEMMs retain their current arithmetic. Grouped projections retain one launch;
actual request token IDs select normalization scales. ASan/UBSan host checks
pass for elementwise normalization, binding, bounds and artifact failures.

Source `f321d93` passes all 55 native compilation units and completes the real
q8192/out512 request. All 12,352 layer-zero first-decode QKV/Z/A/B outputs now
match GB10. All 262,144 sampled prefill input-normalization values also match.
Prefill QKV still differs in 6,657 of 1,048,576 sampled values, with downstream
state differences. Continuation fails at output 115 (196 versus 271), with 396
differences in total. First144/10.375 remains correct. Diagnostic loading is
25685.8562 ms, TTFT11766.9144 ms and TPOT66.8326589 ms. These timings are not
retained performance. Host checks and cleanup pass.
[Native result](../benchmarks/correctness/linux-core-gb10-projection-native-failure-20260922.json).
The next structural change targets dense prefill projection arithmetic using
the existing matrix-producer and exact-replay approach.

`--gb10-prefill-projections` implements that optional q8192 path. hipBLASLt
produces FP32 scratch output; the existing radius/L2 selector admits cells for
the existing lossless scaled-half SM121 replay. Both contiguous and fused
transposed weight views preserve ascending K16 order. A bounded device queue
avoids host count reads. Short-context plans select their own BLAS algorithm
when their BF16 destination differs from the q8192 source plan's FP32 type.
Shared scratch is restricted to the default stream and loaded before READY.
Host tests execute preparation and classification with guard regions, a fully
selected 1,048,576-cell window, selector edges and invalid bindings. They do not
execute GPU replay or qualify the model. The next native run measures the
complete operation, including all preparation and correction work.

Source `d126aa4` passes all 56 native compilation units but exits during model
loading: the installed hipBLASLt returns no supported algorithm for one FP32
destination plan. No READY event or output token is emitted. Native wall is
25331.332 ms, exit2; host checks and cleanup pass.
[Preserved setup failure](../benchmarks/correctness/linux-core-gb10-prefill-projection-native-setup-failure-20260922.json).
Eligible plans with no BLAS solution now select the existing Windows provider's
M64/N128/K16 WMMA producer geometry. The two weight layouts and K512/2048/4096
are supported, followed by the same selection and exact replay. Unsupported
short-context plans keep their original behavior, and a borrowed plan cannot
reuse an empty fallback algorithm. The following run validates its native wiring.

Source `c6c7903` now completes the native q8192/out512 request. Only the N1/K2048
shared-gate plan needs WMMA fallback. All 1,048,576 sampled layer-zero QKV values,
other sampled input projections, convolution and final core match GB10. All
524,288 FP32 prefill-state values match exactly; the selected first decode also
matches through gated normalization, attention output and attention residual.
The next observed discrepancies are 1,055 prefill gated-norm values (of the
4,096-value last row) and 107 decode post-attention-norm values. Same-input CPU
replay with existing GB10 gated math matches original decode and prefill; the
prior native prefill normalization differs on its own inputs.

Complete continuation still fails at output115, actual196 versus expected271,
with394 mismatches. First144/10.375 remains correct. Diagnostic loading is
25756.5479 ms, TTFT28805.7198 ms and TPOT67.0215659 ms. This slower path is retained
as a correctness diagnostic, not a product performance result. All host checks
and cleanup pass. [Native result and next numerical boundary](../benchmarks/correctness/linux-core-gb10-prefill-projection-native-failure-20260922.json).
The next correction covers the original prefill gated norm and unrounded
residual variance, while keeping the complete model and performance gates.

Same-input CPU replay now identifies the residual cause: all 2,048 reference
values match when variance uses the unrounded FP32 residual sum. Computing
variance after BF16 rounding reproduces all native values, including its 107
differences. `--gb10-normalization` applies the existing GB10 gated math to
q8192 linear prefill and the existing residual kernel to linear/full prefill
and singleton decode. Its table owner borrows the GDN reciprocal-root table;
the additional FP32 SiLU table is loaded before READY. Host checks verify table
identity, lifetime requirements, output ordering and invalid bindings. They do
not execute the GPU reductions or establish complete-model correctness.

Source `6ea9adf` passes all 57 native compilation units and completes the real
q8192/out512 run. The 1,055 prefill gated-norm and 107 decode residual-norm
differences are eliminated. Every observed layer-zero prefill surface through
output projection and all 524,288 FP32 state values match GB10. First decode
matches through shared MoE. One FP32 routing weight and one BF16 routed output
differ; the resulting layer-zero BF16 carrier still matches, while the next
layer's carrier differs. These observations do not establish the next cause.

Continuation still fails at output 115 (196 versus 271), with 395 differences.
First token 144 and logit 10.3125 pass the 0.125 reference tolerance. Loading is
25804.0586 ms, diagnostic TTFT28864.1257 ms and TPOT67.2769170 ms. Host checks
and cleanup pass. These timings are not retained performance. The next
observation selects layer one in the unchanged executable to separate its
incoming normalization, prefill state and MoE boundaries.
[Native normalization result](../benchmarks/correctness/linux-core-gb10-normalization-native-20260922.json).

The unchanged-source layer-one observation preserves all 512 outputs. Its
input norm has21 BF16 differences. CPU replay of the fused tail's rounded
variance reproduces every native value; the existing unrounded-residual math
matches all2048 GB10 values using either engine's layer-zero operands. Prefill
layer-one input norm still has547 differences in the last row. The qualified
reference contains layer-one norms and layer carriers, but its detailed linear
state/projection capture is limited to layer zero; no missing-state comparison
is inferred.

The normalization option now snapshots the live decode residual before the
MoE tail and applies the existing residual kernel to the next layer's input.
The 4096-byte copy uses the default stream and prevents output aliases from
destroying the variance operand. One additional decode observation exposes
that next norm. Twelve output-only prefill MoE samples expose the remaining
prefill boundary. Host checks verify the snapshot after source overwrite,
launch bindings, newly sampled widths and unchanged earlier overlay modes.
The following native run tests this repair.
[Layer-one diagnosis and preparation](../benchmarks/correctness/linux-core-gb10-cross-layer-preparation-20260922.json).

Source `acced06` completes all57 native compilation units and q8192/out512.
The observed next-layer input norm now matches all2048 GB10 values. However,
complete continuation regresses to index2 (220 versus82), with398 differences;
this remains an unqualified diagnostic path. First144/logit10.3125 is unchanged.
The new prefill captures show exact layer-zero MoE input normalization and
residual, followed by874 differences in the final MoE output and463 in the
rounded layer carrier. The existing reference lacks prefill MoE internals;
a bounded fresh capture uses unchanged reference sources and frozen tuning,
adds selected-row observations and must reproduce all1216 original outputs.

The first expanded reference attempt fails its original q7169 control:
first220/logit9.375 instead of82/9.25; the remaining31 outputs match. All500
surfaces shared with an earlier rejected reference and its full first logits
are bit-identical. The first observed difference from the qualified reference
is after layer3 attention, but earlier agreement covers the selected row, not
all context rows. No new reference is accepted. Host checks and frozen tuning
checks pass. A bounded retry keeps the source archive, math and observation
configuration byte-identical; the cause of this reference variation remains
unresolved.
[Rejected reference control](../benchmarks/correctness/gb10-q8192-moe-control-rejection-20260922.json).

That retry reproduces all8 original cases and1216 outputs in359.5582s, with
host and tuning checks intact. Its completed capture is qualified. A separate
postprocessor corrects an `armed` versus `worker` metadata lookup error; the
original run, logs and capture stay unchanged. All643 downloaded artifacts,
56791625bytes, are SHA-verified.

At all8 selected q8192 positions, native layer-zero input norm, QKV/z/a/b,
recurrent core, gated core, output projection, MoE norm/residual, shared
projections and router logits match exactly. Shared activation first differs
in1132/4096 values. Original-operand replay matches all4096 using BF16 SiLU
before multiplication and reproduces exactly those1132 differences using
FP32 SiLU. All64 observed original routing weights are FP32 values not exactly
representable as BF16, the frozen q8192 closure's ABI. Routed output differs
in7340/16384 values; that count alone does not isolate its projection errors.
The combined MoE output differs in7113 values, and its layer carrier in4292.
Separately, original prefill residual operands reproduce all2048 layer-one
norm values with unrounded variance; rounding the carrier first creates163
differences. The next implementation addresses the complete prefill MoE and
its cross-layer variance boundary.
[Qualified prefill diagnosis](../benchmarks/correctness/linux-core-gb10-prefill-moe-diagnosis-20260922.json).

The optional `--gb10-moe` route replaces the complete prefill MoE with the
existing Windows provider `9235750`, using live normalized inputs and model
weights. It retains the unrounded FP32 layer output for the following input
norm and terminal prefill norm. An ordered, single-consumption carrier check
rejects stale, aliased or differently shaped bindings. All40 weight pairs are
registered before READY and unregistered before model destruction.
ASan/UBSan checks exercise40 ordered calls,133 rejected bindings/artifacts,
18,435 conversion values and768 residual reduction lanes. Original-operand
replay matches all16,384 observed layer-one norm values and all16,384 terminal
norm values. Eight previous preparation modes remain byte-identical; this
mode prepares58 native units,72 images and16 overlays. GPU/model acceptance
is still pending.
[MoE replacement preparation](../benchmarks/correctness/linux-core-gb10-moe-preparation-20260922.json).

The first native build of source266472f passes57 units and fails the new MoE
unit because its math header precedes HIP runtime declarations. The runtime
header now comes first; the unchanged arithmetic passes the existing host
contract again. That failed build exits1 in117053.528ms with clean host checks;
no product invocation or output is recorded. A fresh native build is required.
[Preserved build failure](../benchmarks/correctness/linux-core-gb10-moe-build-failure-20260922.json).

Sourceafa0671 subsequently passes all58 native units and runs the original
q8192/out512 case. At all8 selected prefill positions, layer-zero MoE output
now matches the rounded GB10 carrier and the following input norm matches
exactly. The full layer-zero recurrent state and its first-decode next norm
also match. However, the continuation fails at output1:248 instead of255,
with470 total mismatches. The layer-one first-decode carrier has663 BF16
differences; their cause is not established by the layer-zero samples.
First144/logit10.375 passes. Diagnostic load/TTFT/TPOT are
27318.1778/31214.4606/67.2300900ms. Exit0, all host checks and cleanup pass;
85 verified files total17264322bytes. No retained performance is claimed.
[Native complete-MoE result](../benchmarks/correctness/linux-core-gb10-moe-native-20260922.json).

An unchanged-source layer-one observation reproduces all512 native outputs
and the first logit. Its current decode inputs match through convolution,
but the inherited prefill state has221911 FP32-bit differences and54053 BF16
differences. At eight prefill positions, input norm, QKV/Z/A/B, raw Q/K/V,
and all256 g/beta values match. Recurrence output already has307 BF16
differences at position63. The preceding63 rows were not observed, so the
recurrence implementation is not yet isolated from earlier input differences.
[Layer-one diagnosis](../benchmarks/correctness/linux-core-gb10-layer1-prefill-diagnosis-20260922.json).

For the next original-q8192 comparison, `AIMA_PORT_GDN_PREFILL_FIRST64=1`
adds read-only first64 rows of each selected prefill surface. Existing samples,
tensor byte limits and all model arithmetic remain unchanged. The reference
observer accepts up to96 distinct original positions to include this complete
first chunk plus the existing eight positions. Host ASan/UBSan checks and33
reference-observer tests pass. The expanded original GB10 capture reproduces
all8 cases and1216 outputs, qualifying its71 selected prefill rows.

Native source1c8b444 completes separate layer-zero and layer-one observations
with all512 outputs unchanged. Layer-zero input norm, projections, raw
convolution, gates and recurrent core match throughout the first64 rows.
The first difference is gated RMSNorm at position57/channel3123: native BF16
0xbae5 versus original0xbae6. It propagates into23 attention-output values,
7 MoE inputs and313 next-layer input-norm values. Original-operand MoE carrier
normalization matches all145408 values, so that arithmetic does not explain
the earlier difference.

The current repair gives q8192 gated RMSNorm its original sixteen-lane,
eight-adjacent-value reduction. Short decode keeps its separate thirty-two-lane,
four-value reduction. CPU replay of the old short layout reproduces the one
native error exactly; the prefill layout matches all872448 original BF16
endpoints across71 rows in layers0,1,2. The original decode rows also pass.
ASan/UBSan verifies both dispatches and the original midpoint regression,
including the failing short-layout negative control.
[First64 diagnosis and repair preparation](../benchmarks/correctness/linux-core-gated-prefill-layout-diagnosis-20260922.json).

The native repair completes all58 Windows compilation units and the real
q8192/out512 request. All15 first64 layer-zero surfaces match, as does the
complete layer-zero prefill recurrent state. First-decode layer outputs0–2
match after BF16 rounding; layer3 is the next observed difference, with944
values differing. The first continuation mismatch moves from index1 to115,
but393 output tokens still differ. First token144/logit10.3125 passes the
original10.375 logit boundary within0.125. Diagnostic load/TTFT/TPOT are
27736.6656/31262.7129/67.382747ms. Host checks and cleanup pass.
An initial controller assertion rejected different ordering of385 identical
source records after the successful build. Recovery verifies every path,
length and SHA, preserves the failure and runs the previously unrun product
phase once. The checker now compares source identities independent of array
order and still rejects missing, changed, duplicate or extra inputs.
[Native gated-prefill result](../benchmarks/correctness/linux-core-gated-prefill-layout-native-20260922.json).

The optional `AIMA_PORT_OBSERVE_FULL_LAYER=3` probe selector captures the
existing full-attention observer at the selected decode output index. It
retains all41 layer boundaries and records QKV, RoPE, context, gating and
MoE stages. K/V caches preserve their token-major layout, split into8192
prefill rows and the decode tail. Linear captures are disabled in this mode;
first64 capture is incompatible. The8MiB/file,32MiB/collection and128-file
bounds remain unchanged. ASan/UBSan verifies179 captured files at both decode
index limits and25 rejection controls. This is output-only diagnosis.

Source9fe809d completes all58 native units and records67 verified observations.
All512 outputs and the first logit are unchanged. First-decode layer3 QKV
matches, as do all4194816 values in the complete8193-row V cache. Q/K after
RoPE differ in411/57 BF16 values, entirely within their rotated64 channels.
The prefill K cache differs in244282 values, including19 unrotated values.
Context differs in1806 values and the layer output in944. The existing original
reference is qualified and matches291 common surfaces from the latest capture.
Its first32 outputs match the frozen continuation. Load/TTFT/TPOT are
27247.5223/31075.3178/67.201921ms, with clean host checks. Synchronized prefill
profiling records15961.2196ms linear attention,10032.9056ms full attention and
5073.0936ms MoE. These remain diagnostic timings.

The optional normalization owner now accepts `AIMA_PORT_FULL_ATTENTION_ROPE_TABLE`.
It reuses the existing Windows SM121 head reduction and single-round BF16 RoPE
helpers for ordinary text q8192 prefill and singleton decode. The fixed rotary
table derives from model configuration and covers262144 positions; no observed
tensor is a runtime input. Four original decode rows across q7169/q8192 match
all36864 normalization/rotary endpoints in CPU replay. Replaying the old rotary
formula with original coefficients still differs from native in7 Q and1 K
values, so it does not separately isolate every coefficient/normalization
effect. The subsequent73ce357 native run resolves all19 unrotated prefill errors
and every other observed Q/K/cache difference, as recorded above.
ASan/UBSan passes8 dispatch checks and54 invalid-binding/artifact controls.
That head-repair build retains327 unchanged files,58 compile units and16 overlays.
[Full-attention diagnosis and preparation](../benchmarks/correctness/linux-core-full-head-norm-rope-preparation-20260922.json).

The upstream release refresh at09:56UTC finds the same five releases and
unchanged bodies; latest remains v1.5.1-native-vl.10/tag0522a57, declaring
native sourceec993444. This does not change the Windows acceptance boundary.
[Release identity refresh](../benchmarks/correctness/linux-windows-release-review-refresh-20260922-r2.json).

Diagnostic load/TTFT/TPOT are25810.6137/28908.8576/67.1387918ms. All host checks
and cleanup pass, with93 verified observations totaling19296194bytes. No
performance or release qualification is claimed.
[Native cross-layer result](../benchmarks/correctness/linux-core-gb10-cross-layer-native-20260922.json).

The latest ordinary Windows q8192 control is 23272.0441 ms TTFT. Structural
projection and QK alternatives preserved component bits but increased their
measured execution times. This experiment instead evaluates the Linux release's
complete resident engine, hipBLASLt GEMMs, precompiled HIP/Triton kernels and
native decode. Linux measurements do not qualify Windows or replace GB10.

## Source and adaptation

`third_party/aima_linux/UPSTREAM.json` inventories 327 unchanged upstream files,
29,415,573 bytes, including the Apache license and preserved component notices.
`tools/import_linux_native_core.py --source-repo PATH --verify` compares every
file and the complete inventory against the pinned Git tree. Build preparation
also verifies the fixed inventory SHA and all imported file hashes.

`tools/prepare_linux_core_windows.py --out build/FRESH_DIRECTORY` produces seven
overlays without modifying the imported tree:

- Win32 shard reads retain 64-bit offsets, aligned direct I/O, buffered fallback,
  tensor scatter and GPU payload checksums. The Windows fallback uses a sequential
  access hint; it has no process-local equivalent of `POSIX_FADV_DONTNEED`.
- Windows dynamic-library loading maps to `LoadLibraryW` and `GetProcAddress`.
  The existing q8192 CK ABI is unchanged. Short-owner initialization uses the
  same explicit DLL and its existing dynamic square-attention entry point;
  rectangular support is not fabricated. This first probe admits only q8192.
- An output-only metric records `first.top1_logit`, the existing certified
  greedy LM-head result. It does not change selection or model arithmetic.
- UTF-8 model/report paths cross the loader ABI explicitly. The two upstream
  media enum-name functions are copied intact without media transport code.

The upstream registry generators validate kernel hashes, sizes, metadata and
schedule bindings. LLVM assembles their 72 unique GPU images into read-only
AMD64 COFF data. `tools/linux_core_coff.py` checks every symbol, image extent,
alignment and SHA, and rejects writable/executable or relocated image sections.
Embedded images total 1,319,512 bytes. A separate 51,056-byte qualified vision
image remains a file loaded by the unchanged engine. Its size is validated by
the upstream inventory; its SHA is checked again by the engine.

The prototype preserves the release's language/vision weight topology,
auxiliary prefill owners, visual warmup and resident prefix-cache allocation.
Those costs count toward loading. Its entry point accepts pretokenized text;
there is no image/video, tokenizer, HTTP or media-fetch frontend in this probe.
This is not a claim of Windows visual support.

## Bounded build and product observation

Run `tools/build_linux_core_windows.py --out build/FRESH_DIRECTORY` locally on
baiying through the existing guarded process owner. It requires a clean commit,
uses the installed ROCm 7.1 toolchain and hipBLASLt import library, imposes a
total deadline and per-process timeouts, preserves compiler output, checks the
Win32 host contract, and records source/generated/binary hashes. It never starts
a model. A CPU build or COFF check is not native inference evidence.
`scripts/baiying_linux_core_probe.ps1` binds the completed prior owner, frozen
source manifest and artifacts for separate build/product phases. It keeps the
existing global experiment mutex, memory checks and process cleanup. The build
has a 1500-second internal deadline inside its 1740-second owner; a product run
has a 600-second owner. The executable name matches the existing `qrt*` process
filter. The Win32 host fixture explicitly marks its file sparse before writing
beyond 4 GiB, avoiding a multi-gigabyte zero-filled allocation.

The resulting `qrt-linux-core-q8192-probe.exe` requires explicit model, prompt
u32 file, CK DLL, vision image and fresh load-report paths. It loads the real
model, performs one cold q8192 request, emits all 512 greedy outputs and streaming
callbacks, and records the first raw logit, first-callback TTFT and loading time.
There is no expected-token or oracle-activation input to this executable.

`tools/check_linux_core_q8192.py` binds completed guarded build/run records,
the frozen dispatch plan, source commits and artifacts to the unchanged
`contracts/gb10_cold_token_matrix_20260911_oracle.json` q8192/out512 case. It
checks actual prompt SHA, all 512 outputs and callbacks, and the first logit
within 0.125. It reports the 10000 ms boundary, retained 4187.415605 ms target
and 30000 ms loading bound without promoting a single run to product acceptance.
Broader contexts, prefix continuation, packaging, protocol and soak requirements
remain unchanged.

## Current verification

`python3 tools/test_linux_core_port.py --out build/FRESH_DIRECTORY` passes on
the macOS controller. It checks the pinned import, generated overlays, real COFF
assembly/image bytes, ASan/UBSan host I/O and input parsing, 25 synthetic observer
rejection controls and both exact logit-tolerance edges. The host I/O test reads
a Unicode-named sparse fixture beyond 4 GiB and checks truncation, EOF and invalid
offset handling. Its POSIX branch does not qualify the Win32 branch.

Native Windows compilation now passes all 53 units, the Win32 Unicode/sparse
file contract, and all 72 embedded image checks. The resulting executable is
7,135,744 bytes. Its actual q8192/out512 run completes with all 512 callbacks,
no oracle reads and clean host checks. The first 115 outputs match GB10;
output 115 is 196 instead of 271. The first token/logit is 144/10.375, matching
the original reference. Internal LM-head certification is not a GB10 result.

Observed command-to-ready is 24,186.8901 ms, TTFT 11,467.4034 ms and TPOT
28.6790133 ms. These are diagnostic timings from a failed continuation, not
retained performance. The 10-second boundary and all original targets remain.
[Completed native build and original-model result](../benchmarks/correctness/linux-core-windows-original-q8192-20260922.json).

The [preparation evidence](../benchmarks/correctness/linux-core-windows-preparation-20260922.json)
binds these checks to Windows experiment source
`0a57516e8c7c1dba55a077eba9d38b6155a0620e`, including the actual PowerShell parser
result on baiying. Its queued controller waited for the repaired runtime's
complete 256k owner to exit and pass cleanup before dispatching the build and
model run. The fixed blueprint binds 335 build inputs, the original prompt,
existing CK DLL and arithmetic tables. The immutable continuation observer
rejects the completed model output; no broader product claim follows.

The later [dependency inventory](../benchmarks/correctness/linux-core-windows-dependencies-20260922.json)
also records the installed gfx1151 hipBLASLt data and DLL imports. Its96 selected
data files total18,849,143bytes; the DLL adds6,012,312bytes. Actual relocated
execution is still needed to qualify that file set. A non-compiling driver query
confirms automatic MSVC, Windows SDK and linker discovery. The queued source,
environment and product boundary remain unchanged.

The subsequent [data graph and symbol review](../benchmarks/correctness/linux-core-windows-data-graph-20260922.json)
decodes all48 selected MessagePack files, resolves46 lazy sublibraries and
verifies the defined functions and descriptors for all1061 distinct solution
kernels and11 extended operations. Every selected file matches its recorded SHA.
This strengthens the dependency inventory; it executes no kernel and does not
qualify relocation or the experiment's numerical output.

## Optional continuation ABI preparation

Later source adds `--windows-rectangular-ck` to the preparation and build tools.
The completed `0a57516` experiment does not use it. The default seven overlays are
byte-identical to that frozen source, and its GB10 observer is unchanged.

The option maps Linux's contiguous token-major BF16 KV cache to the existing
Windows `qrt_ck_fmha_sm121_suffix_bf16_v1` export. It splits K and V at
`kv_tokens - query_tokens`, preserves the original compact Q, local F32 output
and stream, and performs no host read, allocation or copy of device data.
The DLL retains ownership of attention computation and staging. Square q8192
still uses the original entry point; other admitted square lengths use the
existing dynamic entry point. A provider exposing Linux's generic rectangular
ABI keeps that path.

This optional suffix mapping admits at most8192 queries and262144 total KV
tokens. It checks non-null, aligned, non-wrapping spans and rejects output/input
overlap before dispatch. Those are the current imported engine bounds; this
does not yet enable the263168-token256k prefix request or expand the q8192 probe.
Missing exports and failed launches remain errors.

The [CPU adapter evidence](../benchmarks/correctness/linux-core-ck-suffix-local-20260922.json)
records ASan/UBSan checks of the actual generated loader methods against four
recording libraries. All34 checks and31 rejection controls pass, including
maximum geometry, repeated KV addresses, generic-ABI precedence, absent symbols,
provider failure and release cleanup. These libraries only record arguments;
they perform no inference. Native Windows compilation and GB10-attached context
and prefix runs remain required before selecting this option.

## Output-only decode diagnosis

The probe now optionally accepts `--observe-directory NEW_DIR`,
`--observe-output-index 1..511` and `--observe-linear-layer L` together, where
L is one of the model's linear-attention layers. Default requests leave the
existing callbacks empty. The option preserves q8192/out512 and reads no
expected token, logit or activation.

The imported engine's callbacks expose the selected layer's prefill state,
all 40 decode layer outputs and final normalization at the chosen step. The
current arithmetic branch additionally exposes decode attention/MoE stages;
the historical text branch returns before those callbacks. The collector copies only
device-to-host, records byte extents/dtypes/SHA values, and rejects existing
directories, duplicate names, invalid paths, more than 128 files or 32 MiB.
Observation times are explicitly diagnostic. This records an execution; it
does not repair or qualify the continuation.

`tools/test_linux_core_observation.py --out build/FRESH_DIRECTORY` extracts
the actual collector into an ASan/UBSan host harness with recording HIP calls.
It verifies 45 output files byte-for-byte, input immutability, 17 rejection
controls, and the host I/O/parser contract.

Source `893a22c` completes the same native request with exactly the original
512 outputs and first logit. Its 43 captured files total 2,314,240 bytes. At
decode output index 1, the original GB10 transaction's qualified row consumes
token 144 at position 8192. The layer-0 prefill state already differs in 496,140
FP32 cells (maximum absolute difference 0.1292424202); its convolution history
differs in 169 BF16 cells. The first decode layer's accumulated carrier has
1,424 BF16 differences. These observations do not isolate the cause of output
115. They establish that divergence precedes it and that the requested detailed
callbacks were absent. [Native observation and comparison](../benchmarks/correctness/linux-core-windows-first-decode-observation-20260922.json).

## Current arithmetic experiment

`--current-text-decode` in the preparation/build tools selects the imported
current-vLLM decode implementation for text. Upstream selects that implementation
only with an M-RoPE plan; its historical text branch retains older projection,
recurrence, normalization and MoE arithmetic. This experiment reuses the complete
current decode path, including in-place linear state, cross-layer normalization
and the BF16 rotary cache. Ordinary text still uses its actual scalar position;
no visual request or M-RoPE plan is fabricated. Prefill is unchanged.

The option changes only the generated resident-engine source. Default overlays
remain byte-identical to `893a22c`, with and without the independent rectangular
CK option. Preparation verifies all 327 imported files, 53 compilation units
and 72 images.

Source `80b2da8` builds and completes q8192/out512 with the current decode
selection. Output 115 still differs (196 versus 271); 392 positions differ in
total. The first token/logit remains 144/10.375. Its 71 output-only files show
that prefill state/history are byte-identical to the preceding run. Decode
layer-0 input normalization and A/B projections match GB10 exactly; QKV differs
in four BF16 values, but convolution differs in 4211. Timings remain diagnostic:
23893.8577 ms load, 11378.9941 ms TTFT and 34.4807977 ms TPOT.

The installed disassembler shows BF16 multiplication through
`v_dot2_bf16_bf16`, which truncates toward zero. CPU replay with the actual
native inputs and that rounding reproduces all 8192 observed convolution
values. Round-to-nearest products with original GB10 inputs instead reproduce
all 8192 reference decode values and the original q8192 prefill last row's
4096 V values. Applying that arithmetic to native decode inputs leaves four
differences. This identifies the selected convolution error, without attributing
the entire output-115 failure to it.
[Native result and arithmetic diagnosis](../benchmarks/correctness/linux-core-current-decode-convolution-diagnosis-20260922.json).

`--gb10-convolution`, together with `--current-text-decode`, replaces canonical
q8192 prefill and singleton decode convolution with explicit BF16 RNE products,
ordered FP32 addition and the existing model-independent SiLU table. The table
is required through `AIMA_PORT_SILU_TABLE`; its size, SHA and complete layout
are checked before upload. Its allocation and load count toward command-to-ready.
The prefill kernel processes 32 rows per tile and commits final history only
after all readers of initial history finish. Other prefill buckets retain their
imported route and have no new qualification claim.

ASan/UBSan execution of the actual kernel bodies checks 1,089,536 outputs across
tile boundaries, a partial tile, cold/seeded history, and three decode updates.
The actual arithmetic header also reproduces the selected GB10 decode values.

Source `eeaecd7` now builds and completes the native request. Its GPU convolution
matches the RNE host calculation for all 8192 actual input channels, leaving
the same four upstream differences against GB10. The selected prefill state
has 443991 FP32 differences (maximum absolute difference 0.0930145979), compared
with 496140 before the repair. However, complete continuation worsens: output
4 is 220 instead of 79, with 475 differing positions. First144/10.375 remains
correct. Diagnostic load/TTFT/TPOT are 24015.5654/11174.6623/34.6410432 ms.
All host checks and cleanup pass; no native process remains.
[Completed convolution repair and remaining failure](../benchmarks/correctness/linux-core-gb10-convolution-native-failure-20260922.json).

The component fix remains experimental and is not a retained product result.
The next structural investigation covers the complete GDN prefill and decode
boundaries, including reuse of the existing Windows reference-compatible FLA
provider. The imported packed singleton recurrence differs from the GB10
reference's observed two-row speculative transaction.

Same-input CPU replay now reproduces all 524288 original FP32 state cells and
4096 BF16 core values using the existing Windows Q2 arithmetic. Rounding only
beta to BF16 changes 333 core values and 492547 FP32 state cells. On actual
native inputs, the original arithmetic reduces core differences from 1148 to
884, while the carried prefill state still differs. These are operator results,
not a complete model qualification.

`--gb10-gdn` requires both preceding options. It replaces the entire q8192
prefill GDN pipeline with the existing Windows FLA provider from source
`1d11bf7`, and decode with the existing Q2 update on the single accepted token.
State remains FP32 in `[value_head][value][key]` order. Input conversion uses
the qualified model-parameter gate tables, BF16 prefill beta and FP32 decode
beta. It reads no expected output or captured state. The adapter loads its
pinned DLL/AOT and tables before model loading, includes that time in READY,
and reuses conversion scratch across layers.

This option covers unpadded q8192 prefill and singleton decode. It rejects the
retired AOT's intermediate fixtures and checkpoint requests, whose scratch
layouts no longer describe the replacement. Existing model-output observers
remain intact. Five prior preparation modes remain byte-identical; the new
mode has 54 compilation units and the same 72 embedded upstream images.

`python3 tools/test_linux_core_gdn.py --out build/FRESH_DIRECTORY` checks the
actual conversion kernels (33027 values and guard regions), cold/seeded provider
selection, FP32 state forwarding, decode arithmetic flags and injected failures
under ASan/UBSan. This test records HIP launches without executing a model.
Windows native compilation and the original q8192/out512 gate remain required.

Source `4a91bbf` now passes native compilation and completes the original
request. Its first 91 outputs match; output91 is248046 instead of196, with419
differences in total. First144/10.375 remains correct. The layer0 prefill state
has360427 FP32 differences, maximum0.0250058621; the first decode core has362
BF16 differences. Crucially, same-input CPU replay matches every one of the
GPU's524288 updated FP32 state cells and4096 BF16 core values. The remaining
selected-step differences therefore enter through its inputs/state.
Diagnostic load/TTFT/TPOT are25709.4953/11793.0976/38.2652184ms. Host checks
and cleanup pass. [Native GDN result](../benchmarks/correctness/linux-core-gb10-gdn-native-failure-20260922.json).

With the GDN option and an existing output-observation request, the probe also
records the final row of each64-token prefill chunk. It samples normalization,
QKV/Z/A/B projection, convolution, core, gated norm and attention output for the
selected layer. The128 fixed row indices are63,127,...,8191. Sampling gathers
into dead adapter scratch and uses the unchanged bounded output-only collector.
No observation inputs are read. Additional host checks cover4096 samples,
source immutability, redzones, layer selection and five observer faults. A native
run must verify both the new data and unchanged model outputs.
