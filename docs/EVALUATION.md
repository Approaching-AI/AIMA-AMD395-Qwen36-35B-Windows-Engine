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

## Unreleased FLA diagnostics (updated September 10)

The latest live-model layer-zero GDN capture is exact across the complete
q7169 Q/K/V/G/beta inputs, output and terminal state. The model still emits
220 / 9.375 against GB10 82 / 9.25, so the route remains unqualified. The
terminal gated RMSNorm, output projection and post-attention RMSNorm are
exact. The terminal MoE top-eight IDs and FP32 weights also match.
The following records preserve how those boundaries were established.

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
240 MiB plus a 128 KiB shared sigmoid table; generation and model qualification
are still pending and no runtime default is changed.

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
