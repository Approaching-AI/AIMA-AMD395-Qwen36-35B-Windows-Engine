# Linux native core Windows experiment

This standalone experiment ports the compute core declared by Linux release
`v1.5.1-native-vl.10`, source `ec9934446911fdf376da8eebcd83e7b137efbb7c`.
It is not enabled in the Windows product. The GDN, singleton/dense projection
and normalization replacement (`6ea9adf`) runs the real model on baiying.
The q8192 first token and logit pass the GB10 boundary, but output 115 diverges.
Complete continuation, performance and release qualification remain open.

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
