# Completed FLA stages and latency

The repaired whole32a/FLA1d stack now passes the complete original 128k prefix
case on baiying. All 16 owner chunks complete; owner first token/logit 16/25.0
and all 32 combined owner IDs match. Both suffix requests match every original
512-token continuation, including output index 122 (321) that previously
diverged. Suffix first token/logit 248045/6.28125 has zero error. The 512 actual
timed callbacks, 31 cached-owner callbacks, prefix restoration, changed-prefix
rejection before provider invocation, source and final host/cleanup checks all
pass. Native wall is 5773211.967 ms within the original 7200-second deadline.
Load 21204.9022 ms, hit TTFT 62731.6067 ms and TPOT 588.789589 ms are instrumented
functional observations; retained performance, portable package and release
acceptance remain open. See the [complete original 128k boundary](../benchmarks/correctness/rope-single-round-prefix128k-product-20260919.json).

The earlier observation at2026-09-19T02:44:20.5467488Z contains all708 selected
comparisons at twelve original positions, matching their BF16 consumers.
Position91018 matches all40 layer carriers and ten layer15 attention stages;
the decisive Q/RoPE cell is the original -1.9453125. Raw FP32 intermediates
before BF16 consumption may still differ. The completed product proof verifies
that the earlier launch/preflight hashes belong to this same complete run. The
[native boundary observation](../benchmarks/correctness/rope-single-round-native-long-boundary-20260919.json)
attaches actual command/launch, DLL identities and original GB10 references.
Selected intermediate rows alone do not establish every intermediate value.

A separate cold16k/out512 request on the same stack also matches all original
512 IDs, first logit25.625 with zero error, and512 actual callbacks. A new
process completes both8192-input chunks without a cached prefix substitution;
all source, host and cleanup checks pass. Load21269.2616 ms, TTFT73757.2789 ms
and TPOT155.777343 ms are single functional observations. See the
[original cold16k boundary](../benchmarks/correctness/rope-single-round-cold16k-out512-20260919.json).

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
The repaired original128k owner/suffix run completes all sixteen owner chunks
but fails the initial suffix continuation. Owner first token16 and logit24.875
pass the original25.0 reference at the0.125 tolerance boundary. Suffix first
token248045 and logit6.34375 also pass, with0.0625 logit error. The first122
suffix IDs match; output index122 is466 instead of321, and382 of512 positions
differ overall. The CLI exits6 before the cached owner32 and timed suffix
requests. Full128k correctness is rejected, with all original references and
tolerances retained.

The completed guard policy is exercised once by an output-stage interval of
279.346100 ms GPU/279.481000 ms host, and three invalid GPU intervals use valid
host clocks. All owner chunks complete in5006058.3825 ms; the initial suffix
takes362333.5653 ms. Native process wall is5390442.832 ms, inside the unchanged
7200-second deadline, with all host and cleanup checks passing. These clocks
are diagnostic; the failed seed provides no engine-load or timed-hit summary.
No evidence attributes the numerical divergence to the completed latency
outlier. See the [full failed continuation](../benchmarks/correctness/fla-completed-latency-prefix128k-divergence-20260919.json).

An independent GB10 observation reproduces all1088 output IDs across the two
long512-token cases and two short controls, including both unchanged complete
first-logit tensors. At the first divergent suffix output, the matching input
is token11256 at position132217. GB10 assigns both321 and466 logit24.0 and
selects321. The failed native log instead reports466 at24.25 and321 at24.0.
The observed difference is numerical, not a choice between equal native
logits. The reference observer returns every original compute result unchanged;
no native value is supplied to the service. Selected64k/96k/later prefill rows
and40 layer carriers at the divergent decode input are available for diagnosis.
See the [qualified intermediate reference](../benchmarks/correctness/gb10-prefix128-divergence-boundaries-20260919.json).

The subsequent native diagnostic at whole50a/FLA1d localizes an earlier
prefill difference. All40 BF16-rounded layer carriers and the actual final
normalization vector match GB10 at each of the first11 chunk-terminal rows,
through position90111. At position98303, layers0–15 still match; layer16 has
286 differing BF16 values. The final normalization then differs in2025 of2048
values, with maximum absolute error1.10546875. Position106495 also first
differs at layer16. These observations cover selected rows, not every prior
input or recurrent state, and do not identify the faulty operator.

Codex deliberately stopped this diagnostic after13 complete chunks and seven
layer observations in the next chunk to run a focused first12-chunk observer.
Native wall3214709.929 ms, exit-1, the explicit owned-process cancellation and
all passing host/cleanup checks are preserved. The128k prompt did not finish;
no product output or target decode capture was produced. This is neither a
new runtime-error finding nor full128k qualification. See the
[external prefill comparison and recorded stop](../benchmarks/correctness/prefix128-prefill-layer16-divergence-20260919.json).

An independent [layer16 reference window](../benchmarks/correctness/gb10-prefix128-layer16-window-20260919.json)
now contains the original8192 inputs at positions90112–98303, gates, initial
and final FP32 states, and BF16 core output. Its eight tensors total207093760
bytes. All576 original owner/control IDs and the full owner first-logit
tensor reproduce; all2047 downloaded files and host/cleanup checks pass.
The original observer and compute methods are unchanged. This window enables
component diagnosis without supplying native inputs to the reference.

The [selected layer16 MoE reference](../benchmarks/correctness/gb10-prefix128-layer16-moe-20260919.json)
also reproduces all576 owner/control IDs and the original complete first-logit
tensors. Observer8cf7575 retains14 original MoE stages at positions8191,
90111,98303,131071 and131072, including actual input, routing, shared expert
and output boundaries. All2144 downloads and original-history checks pass.
It supports the focused Windows diagnostic; it does not identify the faulty
operator or qualify the unfinished Windows continuation.

The focused first98304-input diagnostic now completes all12 chunks with
whole50a/FLA1d. At98303, the layer16 input norm is bitwise exact, but the
attention update differs in201 BF16 cells, the actual residual in55, and
the post-attention norm supplied to MoE in51. Routing IDs and weights are
exact. The first observable discrepancy therefore precedes MoE. At8191 and
90111, two and five attention-update cells differ, respectively, while the
actual residual and normalized consumer remain exact; these earlier dead
differences alone do not identify an error in the selective projection repair.
See the [completed diagnostic](../benchmarks/correctness/prefix96-layer16-attention-divergence-20260919.json).

Native exit0, one emitted token248046/logit31.875 and all host/cleanup checks
are preserved. The outer wrapper incorrectly checked `output_token_count`
instead of the actual `output_tokens`; its failed validation is preserved
alongside the successful native record. No expected output sequence was
supplied to this diagnostic. It does not qualify the128k product case.

An independent [native layer16 GDN component](../benchmarks/correctness/prefix128-layer16-native-seeded-gdn-20260919.json)
using the original GB10 window at90112–98303 matches every33554432 BF16 output
and524288 FP32 state cell bitwise. The seeded repeat and unchanged inputs
pass; the zero-state control differs. This narrows diagnosis but does not
establish that Windows full-model execution supplied the same operands.

The optional `QRT_QWEN36_PREFIX_LINEAR_CAPTURE_DIR`, `_LAYER`, `_POSITION`
and `_TOKENS` select one original seeded linear transaction. The directory
must be new, the layer linear, the position aligned to8192 and the input
extent1024 or8192 within the supported capacity. The observer copies original
raw/gate inputs, initial/final resident FP32 state, complete core output and
existing terminal projection surfaces, without modifying values or rerunning
kernels. It retains the actual state layout and allows the unused legacy
postconv surface to be absent. It does not enable the older stage-trace option
that disables FLA device preparation. Copies use at most1 MiB host scratch
and a90-second capture deadline; failures prevent a complete core record.
Four sanitizer-backed capture tests and two existing suffix tests pass.
Source41cbbb9 builds on baiying in95271.403 ms; all91 pinned build inputs
verify, and all494 embedded GPU kernels and AMDHSA metadata match the
qualified control. With capture disabled, the original q8192 prompt, all512
output IDs, first logit10.375 and512 callbacks pass. Load21313.516099 ms,
TTFT23264.054501 ms and TPOT101.079970 ms are a single functional regression,
not a new performance baseline. See the
[native build and regression](../benchmarks/correctness/prefix-linear-capture-native-q8192-20260919.json).
The [completed original-input capture](../benchmarks/correctness/prefix96-layer16-original-input-divergence-20260919.json)
runs all98304 original owner inputs on baiying and exits0 in2656581.994 ms;
all host and cleanup checks pass. Output1 remains a diagnostic with no
expected-token gate. Its524288-cell initial FP32 state is bitwise exact.
Raw Q/K/V differ only at rows906–909, original positions91018–91021
(788/893/1825 BF16 cells respectively); one FP32 gate differs at906 and beta
is exact. Core first differs at906 and ends with75 different terminal BF16
cells. The original terminal QKV/Z/A/B are exact; gated norm reproduces
bitwise from actual core/Z. Canonical OUT replay has6 producer differences
but gives the same consumed residual/norm, leaving the observed51 norm
differences unchanged. The upstream cause at91018 remains unresolved.

All16 captured files,409092860 bytes, are unchanged between the completed
layer and completed run. The original PS5.1 wrapper fails after successful
native completion because Measure-Object cannot sum ordered-dictionary
properties. Its script and failed dispatch are preserved; a separate bounded
archive command verifies the completed run and unchanged files. No native
rerun or acceptance claim is made.

Optional QRT_QWEN36_EXACT_ARBITRARY_ALL_LAYER_BOUNDARY_TRACE=1 extends the
existing boundary-trace selector to all materialized layers. It still requires
the boundary-trace flag and uses the existing selected row and extent checks.
It reads input norm, attention residual/norm and output residual through the
existing copy path. It neither dumps whole tensors nor changes arithmetic.
Sourcef726539 builds on baiying in94223.601 ms with all91 inputs verified;
its494 GPU executable kernels and metadata remain identical to the control.
All512 original q8192 outputs, callbacks and first logit10.375 pass. Load is
21392.9991 ms and TTFT23404.0045 ms, a functional observation with no new
performance claim. See the [native regression](../benchmarks/correctness/all-layer-boundary-native-q8192-20260919.json).
The first96k row906 observation reaches the diagnostic64MiB log ceiling and
exits124 after1964375.798 ms. Ten chunks and416 layer rows complete; no token
is produced. All2595 available comparisons through selected position82826
retain exact consumed BF16 boundaries. Host and cleanup checks pass; target91018
is not reached. The [preserved incomplete run](../benchmarks/correctness/prefix96-row906-log-limit-20260919.json)
attaches its original command and the next named diagnostic. That repeat
keeps the same DLLs, inputs and5400-second deadline, bounds logs at128MiB and
observes upstream MoE15 plus linear16 and all-layer norms/residuals.

The [completed repeat](../benchmarks/correctness/prefix96-row906-attention15-divergence-20260919.json)
now finishes all12 chunks, exit0, wall2660594.22 ms, with passing host and
cleanup checks. It emits diagnostic token248046/logit31.875. All2988 selected
row comparisons are preserved. The first11 chunk rows retain exact consumed
BF16 boundaries. At91018, layers0–14, layer15 seed carrier and layer15 input
norm match; the first observed consumed difference is layer15 post-attention
residual,50 cells/max0.000244140625, followed by47 post-attention norm cells.
Actual MoE15 residual and the next norm replay bitwise from their saved inputs.
Canonical routed-down recomputation leaves the same246 next-norm differences.
The earliest observed divergence precedes linear16/GDN. This trace does not
contain attention15 context/update and does not yet identify its operator.

[Original CPU controls](../benchmarks/correctness/row906-upstream15-cpu-controls-20260919.json)
reproduce40 selected expert-down rows/81920 weighted BF16 cells and10 layer15
output projections plus10 post-attention norms bitwise against GB10. A new
default-off observer reads the entire original Q, prefix/tail KV and context
around one unchanged suffix-attention invocation. Its4 capture tests,2 prefix
ABI tests and22 GB10 observation tests pass. Whole model/context qualification
remains separate; saved reference tensors never feed the actual model run.
The [native regression](../benchmarks/correctness/prefix-attention-capture-native-q8192-20260919.json)
builds source02e2a13 on baiying in94977.644 ms with all92 inputs checked.
All494 GPU executable kernels and metadata are unchanged. Every original
q8192 output/callback and first logit10.375 passes. Load21331.5202 ms,
TTFT23241.1652 ms and TPOT100.587183 ms are one functional sample. The new
96k attention15 capture completes in2661453.722 ms with all process/host
guards passing. Its diagnostic output is248046/logit31.875; the truncated
owner has no supplied expected output and grants no product acceptance.
The [original full attention15 window](../benchmarks/correctness/gb10-prefix128-full15-window-20260919.json)
passes all576 original IDs and complete owner first logits. Its10 tensors
total671088640 bytes and retain the8192 actual queries plus98304 logical KV
positions. Selected rows equal the full window, and the cache tail equals
the original K RoPE/V projection. All4380 downloaded files verify.
[CPU controls](../benchmarks/correctness/full15-projection-context-cpu-controls-20260919.json)
reproduce18 Q/K/V projections and10 complete selected attention contexts,
including91017–91021 and98303, bitwise in BF16. Both declared separate/FMA
denominator variants match these consumed outputs; no intermediate FP32
equivalence is inferred.

The [complete native operand comparison and single-round repair](../benchmarks/correctness/prefix96-rope-single-round-root-cause-20260919.json)
identify the earlier defect. All98304 original K/V rows match. Among33554432
Q cells, only position91018/head10/channel16 differs: native-1.9375 instead
of original-1.9453125. Q/K/V projections match at this position. The rotary
product2.21875*(-0.875) is exactly a BF16 midpoint; the rounded cross term
selects its lower neighbour. FP32 FMA discards that tiny term before BF16
rounding, whereas the original BF16 FMA rounds the exact sum once.

Replaying actual Q reproduces all20 context BF16 differences. Original Q
removes all20 with the same complete KV. Actual O projection and residual/norm
replay match native output, locating the defect before those consumers.
The candidate uses a FP32 fast path away from BF16 midpoints and an integer
single-round endpoint at midpoints and edge cases in both prefill and decode.
Full-FP32 legacy coefficients retain their arithmetic. All33554432 original
Q and4194304 K RoPE cells match with the new helper; the old helper reproduces
the same single Q mismatch. Six local FMA tests cover244568 cases, signed
zeros, overflow, subnormals, cancellation and the captured real endpoint.
After correcting device bitcasts, Windows source32a1b96 builds in95260.867 ms
with93 inputs checked. The actual gfx1151 probe matches all489122 endpoints
from244561 independent integer-oracle cases. Original q8192/out512 matches
every output and callback, with first token144/logit10.375/error0. Load is
21252.8118 ms, TTFT23311.4135 ms and TPOT101.181524 ms; these are functional
sample timings, not a replacement median. See the
[native arithmetic and q8192 regression](../benchmarks/correctness/rope-single-round-native-q8192-20260919.json).
The complete original128k owner/1024 suffix/512-output regression is running
with read-only row906 carriers and attention15 stages. Its7200-second bound,
owner32 check, both512 suffix continuations, original logits and restoration
requirements are unchanged. Full128k continuation and performance remain open.

The new independent row906 GB10 capture fails its q7169 control before any
long request: token220/logit9.375 instead of82/9.25. Its first observed
prefill difference is again3 cells at layer3 post-attention norm. All470
common tensor files,32 outputs and the complete first-logit tensor match the
previous rejected routed16 run exactly. These values are not a qualified
reference; the original expectations remain unchanged. Host, cleanup and
frozen-Triton-cache checks pass. The identical-configuration row906 repeat
passes all576 original control/owner output IDs and complete owner first logits;
all4205 downloaded files verify. Its21 selected prefill/decode rows include
row906 in the first12 chunks and positions91017–91021. This independently
qualifies the reference for the native row trace, without qualifying Windows
long-context inference. See the [qualified row reference](../benchmarks/correctness/gb10-prefix128-row906-boundaries-20260919.json)
and [preserved rejected control](../benchmarks/correctness/gb10-row906-observer-control-rejection-20260919.json).
The [additional layer3 attention observation](../benchmarks/correctness/gb10-prefix128-row906-full3-boundaries-20260919.json)
passes all576 original control/owner IDs and complete owner first logits.
All4325 downloaded files verify; eight original prefill attention endpoints
are checked at selected rows. The [same-configuration repeat](../benchmarks/correctness/gb10-prefix128-row906-full3-repeat-20260919.json)
also passes all576 IDs and full owner first logits. All4281 tensor files match
the first qualified run bitwise. The cause of the earlier alternate-control
failures remains unresolved; neither oracle nor model arithmetic changes.

The [upstream15 reference](../benchmarks/correctness/gb10-prefix128-row906-upstream15-boundaries-20260919.json)
also passes all576 original IDs and full owner first logits. Its4370 verified
downloads include eight selected attention stages,17 routed/shared MoE stages
per prefill row and linear16 stages at the same original positions. These
independent references support the active native diagnosis; Windows128k and
performance acceptance remain open.

[CPU arithmetic controls](../benchmarks/correctness/row906-cpu-arithmetic-controls-20260919.json)
now qualify positions91017–91021 directly:15 QKV/Z/OUT projections,10 gated/
post-attention norms,5 residual rows and5 input norms all reproduce GB10
bitwise. The input-norm replay uses the original unrounded FP32 sum for
variance and its BF16 value as numerator. Ten selected layer3 rows, including
both immutable short controls, also reproduce output projection and following
norm exactly. These saved-input CPU controls support diagnosis of actual
native operands; they are not Windows model or performance acceptance.

Independent CPU controls now reproduce all9 original QKV/Z/OUT projection
rows and all6 gated/residual-normalization cases at positions8191,90111 and
98303. They use the pinned original GB10 model weights and complete-domain
math tables. The native residual/norm outputs are reproduced bitwise from
the saved actual native attention update. In an offline counterfactual only,
substituting the original GB10 update removes all55 residual and51 norm
differences at98303. This locates the discrepancy before that consumer; it
does not yet distinguish core, gated normalization or output projection.
See the [qualified CPU boundary controls](../benchmarks/correctness/prefix96-layer16-cpu-boundary-controls-20260919.json).

The actual diagnostic profile leaves `QRT_QWEN36_FLA_DEVICE_PREPARATION`
absent/default-off and emits no device-preparation markers. The new observer
preserves that existing decision. Its optional-null postconv handling also
supports profiles where the existing preparation option is enabled.

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

The actual failed layer8 output also matches an independent GB10 capture.
The five completed 1024-token segments following the layer8 handoff locate
the failed sixth segment at original prompt positions119808–120831. All
4194304 BF16 output cells match the corresponding original GB10 core slice
bitwise, with zero maximum error. The GB10 run independently reproduces all
512 owner IDs and the full first-logit tensor; both immutable controls pass.
No native tensor was supplied to GB10. This directly checks the output that
the old 100 ms guard rejected, without identifying the delay's cause or
qualifying the unfinished native token loop. See the
[failed-output comparison](../benchmarks/correctness/fla-failed-layer8-gb10-output-comparison-20260919.json).

Historical short-context qualification remains attached to its exact source.
Provider `2b33665` passes captured q7169, full q8192/out512 and the original
16k prefix-owner/suffix transactions. The16k run demonstrates a negative
GPU interval with a valid0.3358 ms enclosing host interval. See the
[dual-clock product evidence](../benchmarks/correctness/fla-completion-guard-native-product-20260918.json).
Capture provider `7a33fc9` also passes captured q7169 and every q8192/out512
ID, callback and first logit10.375/error0, with capture disabled. Its load is
21345.493299 ms, TTFT23192.6479 ms and TPOT101.249491 ms. That single sample
is not a replacement performance median; see the [diagnostic provider regression](../benchmarks/correctness/fla-output-failure-capture-native-q8192-20260919.json).

The compiled repair also preserves all GPU executable sections and complete
AMDHSA metadata from the qualified `7b20c90` control: six embedded code objects
containing 32 kernels compare identically. Both DLL identities are checked
against their native build records. This read-only comparison covers neither
all host code nor device non-executable storage; it complements the actual
q7169/q8192 numerical checks and does not qualify another context. See the
[compiled GPU comparison](../benchmarks/correctness/fla-completed-latency-device-objects-20260919.json).

The unpublished R6 portable candidate retains FLA `2b33665`. The current
policy repair has not been put in that archive. Full128k recovery, product
performance and release acceptance remain open.
