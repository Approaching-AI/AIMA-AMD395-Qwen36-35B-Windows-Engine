# Execution state

Updated 2026-09-24 Asia/Shanghai. The mission remains open; no new release
or tag is qualified. Immutable requirements remain in [AGENTS.md](../AGENTS.md),
[project goals](project-goals.md) and [target contract](../contracts/target-contract.json).
Earlier experiments remain evidence, not route restrictions. The previous
active page is preserved in [the September23 history](active-gate-history-20260923-before-native-components.md)
(SHA256 `fddd4b5df9b06b2c82a9d884b2efd962f353bf62c1f539daf9b0f0750f4a3b5d`).

## Current real-model correctness

Controller `JiaweiMnideMini.lan` was identified before dispatch. Native evidence
runs on `baiying` with `D:/models/Qwen3.6-35B-A3B`. GB10 prompt/output token IDs
and first-logit tolerance0.125 remain unchanged; engine self-hashes are diagnostic.

Runtime `c90feccdfc01661b582c5a1db101f2660dc8642c` retains the32-chain packed
K2048/N32 gate repair. Five cold controls pass all1120 original output IDs,
first logits and callbacks. Native MTP remains opt-in; the actual8192+1 bridge
passes. Ordinary q8192 measures21386.521ms load and23392.4967ms TTFT; native
q8192 measures21466.83ms load and24500.6993ms TTFT. These individual observations
do not establish retained performance. Existing product report SHA256 is
`75fc16730e01e422ac3841edc0f792d3f0b1f4deff15f35940b8869169032505`.

The complete256k run has **finished and failed**. All32 cold chunks complete
262144 owner tokens, with first16/logit24.375. Restoring the owner and adding
1024 suffix tokens produces first248045/logit5.78125 and512 outputs. The first
wrong output is zero-based157:5486 instead of2233;322 positions differ.
Native wall22558263.671ms, exit6, host/process/memory guards and cleanup pass.
The subsequent owner32 continuation, second suffix512, timed callbacks and
negative branches were not reached. The unchanged protocol remains required.
Command `D:/projects/run-packed-gate32-prefix256k-r1.ps1 -LongFinalBound 1 -Revision 1`.
[Complete failure evidence](../benchmarks/correctness/packed-gate32-prefix256k-failure-20260923.json)
SHA256 `0a414a708f810beba3e5418c1e620b8a4b01a3c60416d82e0fe3e8f63db168c2`.

At position263291,628 captured surfaces first localize the difference to
layer19 attention context; layers0–18 and the current Q/K/V agree. The completed
logical-KV transfer now compares all556373760 bytes against the original gb10
run. Of134805504 historical K BF16 values, exactly **one** differs:
position263282, KV head1, channel255 is native `0xbf75` and gb10 `0xbf74`.
All134805504 historical V values, current Q/K/V, and the remaining K values
are bit exact. That unrotated K channel changes eight QK scores (query heads8–15),
256 segment-output values and2044 merged-context FP32 values. The separate
real-model run first disagrees at output index157 (`5486` versus `2233`) and
remains failed; this operand diagnosis does not itself prove which upstream
stage produced the one-bit K difference.

The **actual observed Windows DLL** QK and segmented-PV images have now been
replayed on the original gb10 KV and query. All16 QK cases and48 PV cases are
bit exact across two prefix splits, two padded strides, reordered repeats and
three independently labelled score sources. The host guards and cleanup pass.
The first attempt stopped before GPU execution because its D: output drive was
below the 10 GiB reserve; the identical-image replay completed on P:. Thus
compiler/disassembly differences found earlier are not the cause on these
captured operands. This narrows the investigation to generation or publication
of the historical K word, including projection input and K normalization.
[Historical-K comparison and actual-image replay](../benchmarks/correctness/layer19-historical-k-20260924.json)
SHA256 `ea4ae4244203a5ee8e2527a1fba265134a512693450371f54faa39a51004d96d`.
The original KV capture and standalone checks remain in
[layer19 original-operand evidence](../benchmarks/correctness/layer19-original-kv-native-attention-20260923.json).
Neither a diagnostic replay nor a one-bit explanation grants real-model
continuation, performance or release acceptance.

## Actual AMD operator evidence

[Native component and repeat evidence](../benchmarks/correctness/ordered-q8192-native-components-20260923.json)
SHA256 `0000b68a4386832dad24dbc44bfed34b248d3721f60457199bc1dc93c32ca31c`.
This binds original input hashes, actual gfx1151 images, host guards, workers,
commands, model references and source commits. Component results are not
real-model inference or retained performance acceptance.

- Original64 GDN from `e15c6cd1a0e91243ebd330fa55ac24348603e151` passes all
  eight surfaces at q8192 first64 and complete q7169, including113 continuously
  carried original BF16 state checkpoints. Eight further first64 repeats pass.
- Unsigned32 GDN from the declared e03 source fails. Eight repeated identical
  first64 inputs and image sets produce eight different wrong W hashes while
  the interleaved64-bit control remains exact. A synchronization/compiler
  scheduling defect is a hypothesis; deterministic rounding cannot explain
  this observed variability. The mixed native trial remains failed, exit6.
- Dense unsigned32 replay passes54 full-shape QKV/Z/OUT comparisons, including
  all three tile sizes, full queues, original WMMA-selected queues and packed
  operands. It is slower than the current same-run replay: selected QKV
  baseline41–42ms versus198–210ms even for the faster packed candidate.
- Complete layer3 ordered attention passes both64-slab runs: each checks
  1073741824 stored QK FP32 and33554432 BF16 context values. Its component
  wall is6615.7985/6920.8060ms for one layer; it is not selected for product use.
- Routed gate/up and weighted down pass12 full-q8192 route comparisons across
  three tiles and two passes, using actual Windows model tensors and original
  routing inputs. The optional route remains disabled.

The optional combined sources are committed separately at
`2e7cf269b6885055c60c2b5c13c195744644bf3e` on
`codex/ordered-q8192-operators`; all three new options default off. Existing
all-layer original-model CUDA evidence and ABI checks are in the preceding
history and [operator experiment directory](../tools/experiments/ordered_q8192/README.md).
Those results did not predict the AMD performance or the GDN nondeterminism.

## Current execution and next decisions

A separate clean worktree binds the exact original64 GDN source and executed
images. Its explicit selection records both passing64-bit cases and both
rejected32-bit cases; it does not relabel the mixed trial as passing. Windows
compilation and the q8192/out512 product run complete cleanly. All512 original
outputs and callbacks match, first144/logit10.375 has zero error, and model
plus engine load is27595.8828ms. TTFT40680.6294ms regresses from the qualified
20086.8502ms baseline. The route is retained as numerical evidence only.
[Product evidence](../benchmarks/correctness/ordered-gdn-u64-native-product-20260923.json)
has SHA256 `49250da499d96b800e3dc6bb264bff58a450c4242db968b43fc7907e2789dfd9`.
Source manifest SHA256
`e97905250c8172371a65370a77abdba53a34eed16c8a78c10d73ff8618f54bc2`.
Command file is the source-owned `scripts/baiying_linux_core_probe.ps1` in
`P:/projects/AIMA-public-linux-core-ordered-gdn-u64-20260923-r1`.
The unchanged GB10 checker passes correctness and rejects performance acceptance.
A controller diagnostic initially omitted the emitted `group_major_weights=false`
field; a local postprocessing correction checks that field and all original
activation counts without rerunning or modifying native evidence.

Actual AMD comparisons of explicit group-entry/exit barriers and two warps
also fail: neither gives stable first64 results or a correct continuous q7169
chain. The interleaved64-bit controls still pass. This does not establish the
precise synchronization defect. Existing unsigned32 product selection remains
blocked by its numerical failure.

Two further controls tested signed product reduction, internal barriers,
and an explicit layout with no shared-memory conversion. Their original-input
CUDA arithmetic and gfx1151 compilation checks pass; the source and compiler
artifacts remain in [internal-reduction controls](../benchmarks/correctness/ordered-gdn-internal-reduction-controls-20260923.json)
and [explicit-layout controls](../benchmarks/correctness/ordered-gdn-explicit-layout-controls-20260923.json).
The completed **24-case Windows GPU trial rejects all three routes**. The
interleaved64-bit implementation passes four first64 repeats and continuous
q7169. Unsigned32, signed and internal-barrier cases fail every first64 repeat;
explicit layout passes one of four but fails the others and continuous q7169.
Removing layout conversions does not establish the suspected compiler defect.
All trial inputs, guards and cleanup pass; component failure is numerical.
[Actual AMD continuous controls](../benchmarks/correctness/gdn-native-continuous-controls-20260924.json)
SHA256 `f779a45c53a70a0fa7808549bebd99ee58579fe823b5d16c2e5058a91ebf3ceb`.

The same eight images are now embedded in the opt-in prototype at
`b07bf58e5140ea6e256c477f4aacfd39fc89695d`, branch `codex/explicit-gdn-layout`.
The standalone Triton/Gluon compiler rebuilds identical executable content,
constants, load layouts and ABI. Debug source paths and the ELF section-table
offset differ. An initial comparator misread SHT_NOBITS as file bytes; its
failure is retained alongside the corrected comparison and independent objdump
check. No native-trial image was replaced. Host ASan/UBSan passes all128 chunk
bindings and118 invalid pipeline cases. All327 imports,72 base images,17
overlays and60 compilation units remain the same. Eight candidate images total
743344 bytes,4992 above the unsigned64 control, with no new runtime dependency
or device allocation. The option remains off; Windows build and real-model
acceptance are pending.
[Runtime preparation evidence](../benchmarks/correctness/ordered-gdn-explicit-runtime-preparation-20260923.json)
SHA256 `fe0c41006f4cc713008579f4a1d0d4c1c69a9056a7860558da7c6b6fdb50bbe7`.

The explicit GDN kernels now also pass a fresh original-model q8192 comparison
at all 30 linear layers. The candidate alone matches 1006632960 BF16 core values
and 15728640 FP32 final-state values; an independent original64 control matches
the same surfaces. All original inputs and outputs equal the previous qualified
capture, and the original model preserves 576 token IDs and its full first
logits. Candidate outputs never enter model computation. Host, frozen-cache,
input/guard and cleanup checks pass. This expands CUDA coverage only.
[All-layer evidence](../benchmarks/correctness/ordered-gdn-explicit-all-layers-20260923.json)
SHA256 `05eaa3a3a6453c13677940df86b4d2c7dc59c2ca10d756e4ff3bfbdeec5035e8`.

A separate persistent recurrence experiment assigns each block one head and
eight value rows, keeping all128 state channels in6144 shared bytes. Explicit
phase barriers preserve the incoming state until every output is complete.
The original K16 order, BF16 boundaries and FP32 update fma remain unchanged.
The complete q7169 capture and runtime variants both match original core and
final-state values; the capture also matches all113 incoming BF16 checkpoints
and all V-new values. First64 and a nonzero-seeded65-token continuation pass.
Both gfx1151 images compile with225/226 VGPRs and no spills or private memory.
Two earlier CPU compile failures are preserved. This reduces the proposed
q8192 recurrence from384 launches per layer to one; no speedup is established.
The optional runtime binding is described below. [Persistent recurrence controls](../benchmarks/correctness/persistent-gdn-explicit-controls-20260923.json)
SHA256 `f0570c48dbc5cdbe43f985a656e7f76e69ffca23df2ca8be20d2d27fb15a2e23`.

The persistent recurrence now also passes all30 original q8192 GDN layers:
1006632960 BF16 core values and15728640 FP32 final-state values are bit exact.
Original layer inputs, outputs and preparation match the previous qualified
capture; all576 model output IDs and complete first-logit buffers are preserved.
The459.763-second bounded observation preserves original compiler caches and
leaves no GPU process, with minimum host reserve16159875072 bytes. Its
prepared15-case Windows trial has now executed with the actual original64
upstream images. The unfused continuous q7169 controls pass. The persistent
capture variant passes all first64 repeats but diverges at the second 64-token
chunk's incoming state on continuous q7169; the runtime variant is also not
stable on first64. The option stays disabled.
[All-layer original result and native preparation](../benchmarks/correctness/persistent-gdn-explicit-all-layers-20260923.json)
SHA256 `819265da302b6f403395d99ff80553269e8add819818bf73083d46b1952de906`.

The optional persistent runtime is now committed at
`01d1418b28f55339f28c29b6b47821c29626c984`, branch
`codex/persistent-gdn-prefill`. Its default-off setting requires native GDN
prefill and accepts cold q8192 only. Five exact original64 upstream images
feed the fused recurrence; the six images total1545744 bytes. The original
cumsum plus this pipeline uses7 AOT launches per layer, compared with390.
It reuses existing scratch and adds no device allocation or runtime dependency.
ASan/UBSan checks pass116 rejected bindings, six dispatch failures and the
legacy389-launch pipeline. Dense/routed host checks also pass with the shared
HIP stub. Offline rebuilding reproduces all six images' executable content;
the exact native-trial bytes remain embedded. Actual builder preparation has
327 imports,72 base images,61 compilation units and441 hashed source inputs.
Sixteen overlays remain identical; the sole changed overlay reports actual
persistent activation and launch count. An initial extra rectangular-attention
preparation flag was removed to match the product builder, without runtime
source changes. The conditional product entry binds the full15-case native
trial and independently selected dense/routed/attention results. Three local
selection checks report pending evidence and make zero remote calls. No source
manifest or bundle has been staged, and no new Windows/model result is claimed.
[Runtime and entry evidence](../benchmarks/correctness/persistent-gdn-runtime-preparation-20260923.json)
SHA256 `5d2a76699a0a3e18f29af916eeafcbcbc9e07a46d66b9a430b98bc2f40dd8af2`.

A compact persistent variant splits residual shared storage into16-row tiles
and loops over row groups and K16 reductions. Its runtime instruction section
shrinks from615296 to65408 bytes, with158 VGPRs instead of225; shared memory
remains6144 bytes and no spills/layout conversions appear. The capture image
uses159 VGPRs and67328 instruction bytes. Both variants match every previous
q7169 core/final-state comparison; capture also preserves113 incoming BF16
states and complete V-new values. First64 and nonzero-seeded65 controls pass.
The source, compiler and numerical worker are separately frozen. CUDA timing
includes compilation and is diagnostic only. A fresh original-model q8192 run
now confirms all30 GDN layers:1006632960 BF16 core,15728640 FP32 final state
and7864320 FP32 cumsum values are exact; all576 original model outputs agree.
[Compact all-layer result](../benchmarks/correctness/persistent-gdn-compact-all-layers-20260923.json)
SHA256 `2ce6f74a225a018eb5b22c82abebcacdf615514752099a34b6ff0f2e8198bbbb`.
The subsequent Windows15-case trial passes all nine first64 cases, but both
compact continuous q7169 capture repeats first diverge at checkpoint1, and
runtime/capture final states fail. The unfused64-bit controls remain exact.
The compact runtime is committed separately at `75a212ea75c221b37fa04ff4a77a6529f35c7210`
with the option off. No product build or inference credit follows from the
failed native component.
[Compact recurrence source controls](../benchmarks/correctness/persistent-gdn-compact-controls-20260923.json)
SHA256 `97f0dac618a0d6db91a89852e45c27a0e4b3d5d46dd9f7c1dd55449d67c819ee`.

An explicit-layout dense replay also preserves the existing selected-pair
addressing and ascending K16 arithmetic. Its three gfx1151 tiles have zero
layout conversions, LDS operations or spills, with 64/117/229 VGPRs. CUDA checks
pass 4362240 real partial/queue/layout values and 352321536 full-q8192 QKV/Z/OUT
values. A 54-case native plan retains the previously executed baseline and
harness unchanged, including actual WMMA-selected and complete queues. It is
prepared only, behind the active 256k owner, and has a 900-second process bound.
No runtime integration or speedup is claimed before AMD evidence.
[Dense layout evidence](../benchmarks/correctness/ordered-dense-explicit-layout-controls-20260923.json)
SHA256 `97cabf6fa75fa90e49791cd0c634d3d249ce23786f7a93c613e5f95737da1305`.

Explicit attention layouts now pass all 64 query slabs of the original q8192
layer: 1073741824 stored FP32 QK values and 100663296 BF16 context values across
selected PV and two full-output tiles. Probability/scales, four empty/partial
queue controls, inputs and guards pass. All five gfx1151 images compile without
layout conversions or spills. QK, probability and tiled PV have no shared-memory
instructions; selected PV retains 16 shared bytes for its queue-range reduction.
The first compile exposed a missing Gluon broadcasting API; the corrected API
form preserves the arithmetic and passes the complete numerical run. A paired
300-second AMD plan includes eight full passes, 512 slab comparisons and four
partial-queue controls. No native execution or speedup is claimed yet.
[Attention layout evidence](../benchmarks/correctness/ordered-attention-explicit-layout-controls-20260923.json)
SHA256 `4148cc6dfc533e756854d49cbb45554ed578aa86144442d96525164f322e48b7`.

The routed-expert explicit layouts preserve the original gate-up/down arithmetic
and addressing. Actual q8192 operands and model weights pass 603979776 full BF16
values, 24558 sparse values and 15 malformed-queue controls. The unchanged
baseline must first match each complete original-model output hash before its
tensor supplies expected values. All six gfx1151 images have zero layout
conversions or spills; they retain 16 shared bytes for invalid-queue flags.
A prepared 900-second native plan interleaves unchanged and explicit images
for 24 full comparisons, plus 15 invalid and two partial shuffled queue cases.
It has not been staged or executed while the full256k owner is active.
[Routed layout evidence](../benchmarks/correctness/ordered-routed-explicit-layout-controls-20260923.json)
SHA256 `9c1f831bd0780c2423da25425dd4b857e730204a244d16f26548f242727de3f8`.

A combined candidate at `c85259254b805788dc80680caad37685708c98a8`, branch
`codex/explicit-q8192-operators`, embeds these exact pending images: 13
dense/routed/attention images (521376 bytes) and eight GDN images (743344 bytes).
All four runtime options remain disabled by default. Existing C++ launch and
ownership source is unchanged; three ASan/UBSan host checks pass. Build
preparation preserves 327 imports, 72 existing images and all generated
overlays. Offline recompilation reproduces executable content and launch
metadata for all 15 q8192 diagnostic images; selected runtime bytes remain
unchanged. Two tiled PV variants are outside the current runtime binding.
No Windows build, AMD execution or real-model result is claimed for this
combination. [Composition evidence](../benchmarks/correctness/explicit-q8192-runtime-preparation-20260923.json)
SHA256 `172919606811daf0ca50f0539d9311dc8f63b5865adf28f894d449f7356a9791`.

The combined candidate now also passes the complete original-model CUDA
comparison: 160 dense projections (5452595200 BF16 values), 80 routed
projections (8053063680 BF16 values), and all ten attention layers
(10737418240 stored QK FP32 and335544320 context BF16 values). All250
comparisons are bit exact, as are24558 sparse routed values;15 malformed
queue controls pass. Original operands, weights and outputs match the previous
qualified per-family captures. The original576 output IDs, first-token logits
and complete first-logit buffers are preserved. Candidate outputs never enter
the model. The1128.294-second observation exits0, preserves frozen compiler
caches, stays above the8GiB host reserve and leaves no GPU process. Its six
arithmetic source files match the committed optional runtime. These CUDA
results do not establish native gfx1151 correctness or performance.
[Completed all-layer evidence](../benchmarks/correctness/explicit-q8192-all-layers-20260923.json)
SHA256 `a0466be9a3825d6178d0a8e870984aa0e5e7c00a08a354ba11d325dc8f4fb085`.

The [conditional product entry](../benchmarks/correctness/explicit-q8192-product-entry-20260923.json)
requires selected actual native results and clean completion before staging.
Eleven host controls exercise original-output binding and rejection behavior
using the previous real dense report. The pending GDN stage also had an input
inventory mismatch: generator Python files were included in the C++ build
list despite not being read by that builder. The corrected `stage_linux_core_explicit_gdn_r2.py`
and combined stage evaluate the builder's actual inventory, with generation
inputs bound separately. They reproduce all428 previously executed u64 source
entries and select438 for the combined candidate. GDN and combined source
commits remain unchanged. Both product manifests remain absent, with zero new
remote calls from this preparation. Record SHA256
`ce06d2a8c1c32a44a165e74020f7469cb7e0bfef401c13025760e36fed7bf6f5`.

A bounded output-only observer is committed at
`b2b7c0bea3f3fe41ada0fc2d99abb48079e64478`. It saves one layer/position's logical
prefix and tail KV, original query, padded scores, segment output/max/sum and
context before score scratch is reused. Host ASan/UBSan controls pass, including
copy bounds and failure cases. Windows whole-provider, CLI and prefix-probe
compilation pass in121019.155ms with174 source inputs verified and unchanged
compiler arguments. Build run SHA256
`9ab3e19d41947568ff77f6c82633d5223732b50141e7d019ed214060303ce9fd`.
The whole DLL is `066299c0f40d81601d84e623cd3328f6faaf26a638212e0dbfdac1cd18f226ba`.
The ordinary q8192/out512 control passes all512 original outputs and callbacks,
first144/logit10.375. Load21609.0949ms and TTFT23256.3025ms remain outside the
required prefill performance target. [Build and control evidence](../benchmarks/correctness/q1-cache-observer-native-control-20260923.json)
SHA256 `e94ab27be5a8d062332e3674b2aad9a398b937d761311958479123bee987fded`.
The complete256k observation started at2026-09-23T05:44:42.9729503Z, PID7408,
with the unchanged28800-second process deadline. It is active in cold prefill.
Command `D:/projects/run-q1-cache-capture-prefix256k-r1.ps1 -LongFinalBound 1 -Revision 1`.
Source manifest SHA256 `4d162bb8abbc9721600aa977f8d3b98e1c4b057783ee79a9f2d9318adf344c40`.
Logical KV and other diagnostic files stay on Windows D and will be transferred
directly to GB10; controller disk need not hold another full-history copy.

Windows disk reserve remains10GiB. Two input archives and646012928 bytes of
completed native intermediates were relocated to the reference host, with
source and destination SHA256 checks before removing the old copies. Inputs,
result metadata, binaries and original evidence contents are preserved. The
native report records the new retained paths.
Five old128k layer15 controller copies totaling486539264 bytes were also
removed only after matching their complete SHA256 on GB10. Their original
remote files and all local manifests remain; the retention receipt is
`build/recovery-20260910/controller-evidence-retention-20260923-r1.json`
(SHA256 `6d495ce93d2049df2c43f360c8e515af9a78564e506ba0c95468a67b612271cc`).

## Unchanged performance and release requirements

Real q8192 TTFT must be below10000ms, excluding startup. Retained targets remain
1506.407263tok/s,4187.415605ms TTFT and35.502151ms TPOT; model plus engine load
must be at most30000ms. The fastest qualified Linux-core Windows observation
remains source21e32349: all512 original outputs/callbacks, first144/logit10.375,
load27593.4027ms, TTFT20086.8502ms andTPOT226.47153463796477ms. No new retained
performance is claimed.

Release work still requires correct complete256k continuation/prefix protocol,
context-retirement boundaries, package/runtime inventories, relocation and
protocol controls, cold and streaming matrices, and the3600second soak bound
to the final native artifacts. The [September23 Linux release refresh](../benchmarks/correctness/linux-release-refresh-20260923-r2.json)
still identifies `.10`; the latest five bodies and publication times are
unchanged. Current Windows chat/API source is byte-identical to the prior
b3d75e9 HTTP-qualified protocol, whose evidence retains its original provider
and package identities. Windows VL support is not claimed. Continue autonomously
with measured structural changes while the mission remains open.
