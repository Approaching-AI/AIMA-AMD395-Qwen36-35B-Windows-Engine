# Experimental Linux compute core

The optional [Windows core prototype](LINUX_CORE_WINDOWS_PROTOTYPE.md) imports
327 unchanged files (29,415,573 bytes) from Linux native source
`ec9934446911fdf376da8eebcd83e7b137efbb7c`. The concrete benefit under evaluation
is replacing the current multi-second projection, recurrence and MoE route with
the release's complete native resident computation. A native Windows q8192 run
now completes, but its continuation fails GB10 at output 115. Its timings remain
diagnostic, and the product does not select this prototype.

The import includes source, generated schedules, AOT GPU images and upstream
licenses. It reuses Windows HIP, hipBLASLt and the existing CK provider. Python
standard-library code and LLVM assemble registries at build time only; Python,
Torch, Triton and vLLM are not runtime dependencies. The pretokenized text probe
does not link ICU, FFmpeg, curl, image decoders or the Linux server frontend.

The selected COFF registry embeds 1,319,512 bytes across 72 unique GPU images.
The engine also requires its separately hashed vision-attention image and
retains upstream visual weights/warmup and cache allocations. Their actual
Windows loading and memory costs are included in the observed 24,186.8901 ms
command-to-ready time. The first executable is 7,135,744 bytes; its final
redistributable DLL closure is not yet qualified. Existing Windows ROCm
hipBLASLt is present; no separate BLAS installation is introduced. Removing
`third_party/aima_linux`, `native/linux_core_port` and their dedicated tools
removes the experiment without changing the current runtime/package route.

The optional GB10 convolution repair reuses the existing 648,036-byte
model-independent SiLU table, SHA256
`673f8dd1280700578c1e8743afd2e3b4da134b1fbd463c890527e1c4d9f796b8`.
It adds one file, 648,036 bytes of persistent device storage and one temporary
host copy during loading, with no new library or installed package. The table
enumerates FP32 activation behavior independently of model tokens. This avoids
the demonstrated AMD BF16 product truncation and preserves the GB10 activation
boundary. The native convolution now reproduces the intended RNE calculation
on the captured input. The complete model still fails, so this table-backed
experiment has no product or release qualification.

The optional `--gb10-gdn` experiment reuses the existing Windows FLA provider
from source `1d11bf7` and its pinned build assets (1,205,368 bytes), plus existing
gate, sigmoid, exponent and reciprocal-root tables (452,527,712 bytes). Gate
tables enumerate every BF16 input for the pinned model parameters; other tables
describe arithmetic independently of prompts. No captured activation is read
by inference. The benefit under evaluation is preserving the complete original
prefill/state-update contract instead of mixing Linux packed decode with the
GB10 reference's Q2 recurrence.

The adapter owns 404,750,592 bytes of reusable FP32 conversion scratch and one
device copy of those tables. The FLA DLL additionally owns its normal scratch
and exponent/root copies. SHA checks, uploads and provider preparation count
toward command-to-ready; first-use provider scratch allocation remains in the
measured request. This experiment adds no installed library or Python/CUDA
runtime. Its memory, loading, correctness and performance costs must be measured
on the real model before product selection or packaging.

The optional `--gb10-projections` experiment reuses the existing Windows
K16/width-26 arithmetic headers for singleton dense projections. It adds no
library. First-layer normalization uses the existing complete 248,320-entry
model embedding inverse table, SHA
`f4e37f759c586bfc8fcc4d74cefdd89235f0f0c0c90cd286147e331e87509e67`.
Its 993,280 bytes and 32,768 bytes of live prompt token IDs are uploaded before
command-ready; the temporary host table is then released. Decode selects the
actual previously emitted token. No captured activation or expected output
is a runtime input. The table is pinned separately from the later dynamic
normalization table packaged by the other runtime. This experiment requires
real-model qualification and complete model/table binding before release.

The additional `--gb10-prefill-projections` option changes q8192 dense GEMM
destinations to FP32 and reuses existing SM121 midpoint selection and staged
scaled-half exact-replay headers. Its default-stream owner reserves 598,360,324
bytes for FP32 outputs, lossless operand staging, row norm bounds and a bounded
candidate queue. This allocation counts toward command-ready; operand
preparation, classification and correction count toward measured prefill.
The maximum candidate window has 1,048,576 cells, so even complete selection
fits its buffer. Counts remain on the GPU. Seven earlier overlay modes are
unchanged. No external library or offline asset is added; real-model numerical,
memory and performance qualification remain required.

The [installed Windows dependency inventory](../benchmarks/correctness/linux-core-windows-dependencies-20260922.json)
finds a6,012,312-byte hipBLASLt DLL. Its installed data directory contains1078
files totaling434,067,673bytes across GPU architectures. The complete96-file
gfx1151/shared-metadata selection is18,849,143bytes, with every SHA recorded.
The later [structured data review](../benchmarks/correctness/linux-core-windows-data-graph-20260922.json)
decodes all48 MessagePack files and resolves all46 lazy sublibraries without
another nested placeholder. Their1271 solution entries name1061 distinct
kernels; every function and64-byte descriptor is defined in its matching
gfx1151 code object. All11 extended-operation functions/descriptors also exist.
The common1030-kernel object is retained conservatively. The46 compressed
containers have empty host entries and only a gfx1151 GPU payload.

DLL plus selected data would add24,861,455bytes before the executable, existing
HIP/driver requirements, Microsoft CRT and redistribution notices. COFF imports
explicitly include amdhip64_7, MSVCP140 and VCRUNTIME140/140_1, plus Windows API
libraries. The actual relocated process must establish the complete runtime
dependencies before that selection is packaged. No installed files or queued
experiment settings have been changed by this inspection.

The installed hipBLASLt MIT notice is1080bytes, SHA256
`b185aaa652b0bf066c37a0d6314ce4bf4521e4a3c9bf46edd2f6a777ac522223`;
it must accompany any redistribution. The structured review reused cached
MessagePack, Zstandard and ELF parsers on the controller. They are evidence
tools only and add no Windows runtime dependency. Resolving explicit file and
symbol references does not establish the final process's dynamic dependencies
or relocated execution.

# Experimental packed Q1 normalization table

The non-speculative Q1 recurrence uses a model-independent SM121 square-root
table, SHA256 `4f40ec04656a43948813e188f914e7e2f78d8f6b09b45e647027978aa520191a`.
Its 17,039,392 bytes reproduce the original square-root instruction; ordinary
rounded square root does not reproduce the captured normalized Q/K values.
The existing reciprocal artifact remains byte-identical, with its normal
exponent range now checked exhaustively from -126 through 125. The builder
reuses the reference host's installed Torch/NumPy/Triton; these remain offline
tools and add no Windows DLL or Python dependency.

Packaging adds the square-root data file and about 16.25 MiB of persistent
device storage, plus one same-size temporary host load buffer. The Windows
CNG SHA check binds the file before use. `QRT_QWEN36_Q1_SM121_SQRT_TABLE` selects
its path; it is loaded only for requests whose retained prefix reaches the
reference drafter's 262144-token limit. The speculative path retains its current
tables. Native replay now matches every original state/core bit in both memory
layouts for all six captured cases. The actual q8192 model also passes all512
outputs, but its short request does not load this table. The integrated packed
load cost and original full256k product boundary remain under verification.
Removing the packed compatibility path removes this artifact and its memory
cost; it also removes the demonstrated arithmetic repair for that path.

Evidence: [arithmetic domains](../benchmarks/correctness/sm121-packed-normalization-arithmetic-20260920.json)
and [recurrence candidate](../benchmarks/correctness/prefix256-packed-recurrence-arithmetic-repair-20260920.json),
with [Windows verification](../benchmarks/correctness/packed-q1-native-regressions-20260920.json).

The [near-limit reference](../benchmarks/correctness/gb10-draft-limit-transition-20260920.json)
also observes requests crossing the limit during generation. Their final
speculative batch may contain a token at 262144, so the generic switch cannot
be inferred from absolute position alone. Current runtime selection still
uses the initial retained prefix; this unresolved selection issue is separate
from the verified arithmetic table and does not change its dependency cost.
The [prepared portable profile](../benchmarks/correctness/packed-q1-portable-preparation-20260920.json)
adds the fixed-SHA table path plus text-only/ordered-fixed loading options.
It has 535 options and 34 relative paths. The proposed 269-artifact R9 runtime
has not been assembled or qualified as an archive.

The optional original MTP observer reuses the reference container's installed
Torch and Python standard library. It copies at most 64 MiB of selected payloads
per request, plus a shared weight snapshot bounded to 32 MiB. Source hashes
pin the unchanged original drafter and model implementations. These files are
offline diagnostics and add no native Windows library, model tensor or package
artifact. Removing the observer and its optional CLI flag removes this tooling
route without changing native inference.

Its optional norm-launcher observer also reads the existing Inductor launcher
and generated cache files. It preserves the selected config and arithmetic,
copies at most 32 kernel identities per request with a 256 KiB limit per
PTX/IR/metadata file, and records original tensor layouts without extra tensor
payloads. This adds no installed package or native runtime dependency.

The optional complete-prefill observer adds a separate 320 MiB copied-data
ceiling per request, restricted to the first complete batch of at most8192
tokens. q8192 uses310411264 bytes, including the actual shifted input IDs.
It reuses Torch and standard-library file/hash operations, preserving the
existing selected-row and shared-weight ceilings. These offline artifacts
support whole prompt-cache validation and are not packaged with the runtime.
The completed two-control capture saves582059012 full-frontend bytes; its
complete diagnostic download is522991645 compressed bytes. The new original
K16 FC/KV baseline reuses the existing HIP integer accumulator and requires no
projection workspace or added library. Its bounded prompt-cache probe remains
unrun on Windows. The explicit split1024 pre-FC norm reuses the same arithmetic
table as the prior reduction and adds no data artifact.

The isolated MTP draft-limit state machine and its offline schedule probe use
only standard C++17. They add no runtime dependency and do not load model data.
The helper requires actual acceptance counts from its eventual caller; reference
counts used by its diagnostic probe must never feed native generation.

The separate native MTP normalization and fusion-input kernels reuse HIP and
the existing SM121 reciprocal-root table. Their CPU probe uses standard C++17.
They introduce no library or data artifact. The concrete arithmetic benefit is
matching the selected original stride-512 norm reduction and fusion order on
all 322 qualified reference rows. Windows launches and complete native MTP
integration remain unmeasured. A future full prompt KV route must account for
its own cache and selectively loaded MTP weights; these kernels allocate neither.
The separate K normalization/RoPE writer also reuses the existing model-derived
BF16 rotary table and single-round BF16 FMA helper. Its proposed token-major
K512/V512 cache requires 536870912 bytes at 262144 tokens. This cache is caller
owned and not yet allocated by inference. The offline GPU probe deliberately
allocates the full cache plus two host images to check every untouched cell;
that diagnostic host storage is not a proposed runtime dependency.

`QRT_PREFILL_DESCRIPTOR_BATCH_RESIDENT_MODEL_MTP=1` optionally retains all 19
original `mtp.*` tensors in ordinary or ordered text storage. The default is
off. The real-model metadata plan adds exactly 1689281536 device bytes
(about 1.57 GiB), with unchanged target fixed-weight positions and vision still
omitted. The weights already exist in the model shards, so no package artifact
or dependency is added. The concrete benefit is making complete MTP weights
available to a future drafter and its prompt-cache builder. Model-load time,
Windows memory use and native MTP inference are unmeasured for this option.
Borrowed MTP views now use a storage epoch so reloading the same model with a
different resident scope cannot publish aliases from the previous allocation.

The separate prompt-cache owner uses the existing HIP runtime and tables, with
no added package dependency. At 262144 cached tokens and 8192 scratch rows it
owns 536870912 cache bytes, 150994944 scratch bytes, a four-byte device flag and
a four-byte pinned host flag: 687865864 bytes total. Weights and the caller's
projection workspace are additional. Failed completion permanently disables
the request and retains potentially live allocations until process teardown.
This owner is host-tested but is not yet used by native model inference.

The additional MTP Q/RoPE and residual-normalization kernels reuse these same
HIP/math/table components and add no dependency or runtime data artifact. Q/gate
outputs require 16384 bytes per row; a residual norm has two 4096-byte outputs
per row and supports corresponding input replacement. Its full-pipeline memory
reuse is not yet wired or measured. CPU arithmetic checks cover the original
322 rows; the expanded HIP probes remain uncompiled and unrun.

# Optional caller-side document checker

`scripts/check-agent-documents.py` uses only Python 3.10+ standard-library
ZIP, XML, JSON and hashing modules. It detects missing or malformed DOCX
outputs and optionally recovers text without accepting a fake container.
It runs on the caller's machine and adds no inference runtime dependency.
The portable archive includes the script, `docs/API.md` and
`docs/AGENT-INTEGRATION.md`; their exact sizes and hashes are recorded in
`FILE-SHA256SUMS.json`. No Python interpreter or Office library is bundled.
The helper comes from the Apache-2.0 Linux sibling `.9` tag identified in
the integration guide. Its container checks do not substitute for rendered
layout or content review.

# Optional SM121 exponential compatibility data

The optional `QRT_QWEN36_Q1_SM121_FULL=1` decode path also consumes the
existing model-parameter rotary cache through
`QRT_QWEN36_Q1_SM121_ROPE_TABLE`. The loader verifies 33,554,432 bytes and
SHA-256 `ba12ce218327d4cf23aac7dfacd8e9efbc99fd207611a8466227089838ef0e80`
using the existing Windows CNG dependency. It adds 32 MiB of device storage
and a transient host buffer, reusing the existing reciprocal-root and
sigmoid tables. It introduces no new offline artifact or runtime library.
The concrete benefit is matching every observed first-decode Q/K rotary
value for q8191 and q7169 with the original BF16 multiply/FMA sequence.
This is a diagnostic route pending native product qualification.

The runtime-tail variant contains 264,736 rows (33,886,208 bytes), SHA-256
`1c4d86d492b587a5f4f433dad84c99722d2fee702bcf27d7af77c7fe6a5d2d6a`.
It adds 331,776 bytes to the original host/device table and packaged data file,
covering a 256k prefix, real suffix inputs and resident decode storage. Both
layouts remain recognized by fixed length and SHA-256; Q1 also checks the
requested position against the loaded layout. The original 262,144 rows
are byte-identical. All 16,943,104 extended BF16 values match the original
GB10 MRoPE constructor and a separate construction at the requested extent.
Model configuration remains 262,144 positions. This variant reuses Windows
CNG and the existing offline CUDA reference tools; it adds no runtime library.
The rebuilt Windows runtime passes the real q8192/out512 and 16k-prefix
regressions with this table. The current portable package also passes relocation,
all 268 runtime-asset hashes, real HTTP/protocol/prefix and five short cold cases.
The expanded RoPE retains its generic packaged basename; the interpolated CK
exponential table adds one 38,909,480-byte artifact. The complete unpublished
ZIP is 131,997,808 bytes, containing 281 release files. This packaging change
adds no library dependency. Larger context and sustained-soak qualification
remain open. See `benchmarks/correctness/current-portable-http-20260915.json`,
`benchmarks/correctness/current-portable-protocol-prefix-20260915.json`,
`benchmarks/correctness/current-portable-short-matrix-20260915.json` and the
separate real q8192/16k-prefix evidence.

## Optional long-context prepared QK workspace

`QRT_CK_SM121_LONG_PREPARED_DECODED_QK=1` extends the enabled prepared-QK
route to multi-query calls beyond q8192, with at most 8192 consumed query
rows. It uses the range component validated against original native K16
scores, retains the original BF16 fallback, and keeps 32-query long slabs.
The intended benefit is avoiding repeated Q/K decoding across long-history
score tiles; real-model qualification is still required for this integration.

The separate owner grows by 8192 key tokens up to 264736. Its fixed compact
Q region plus complete K metadata uses at most 679039232 device bytes.
Growth allocates the replacement before retiring the old owner, so both
allocations coexist briefly; failed allocation preserves the previous owner.
Every selected call refreshes its Q interval and complete K history under the
existing provider mutex. The fixed cold owner and single-query decode keep
their existing behavior. No new DLL, table, Python, CUDA or third-party
runtime dependency is introduced. Default-off source and package settings
remain until correctness-attached product measurements support selection.

## Optional long-context V transpose workspace

`QRT_CK_SM121_LONG_TRANSPOSE_VALUE=1` extends the existing
`QRT_CK_SM121_COMPACT_PV_TRANSPOSE_VALUE=1` operand view to long calls.
It uses the existing HIP transpose kernel and preserves original integer
PV arithmetic, admission bounds, candidate ownership and ordered reduction.
The intended benefit is more contiguous reads during long-history PV replay;
native numerical and product measurements determine whether to retain it.

The independent buffer grows in 8192-token increments, capped at 264736
tokens or 271089664 device bytes. It is allocated only for selected long
multi-query calls and refreshed for every layer and invocation. A failed
growth preserves the previous buffer; successful growth briefly holds both
allocations before releasing the previous owner. The fixed 8192-token short
buffer and single-query decode keep their existing allocation behavior.
The option defaults off. It adds no artifact, library or runtime dependency.
Native component checks and the q8192/16k-prefix product boundaries pass
at `7fcf8a1`. Larger contexts and package qualification remain pending. See
`benchmarks/correctness/long-transposed-value-prefix16k-20260915.json`.

`QRT_QWEN36_Q1_SM121_ATTENTION=1` adds the original 32-token online
attention reduction to that decode path. It uses the existing reciprocal
artifact through `QRT_QWEN36_Q1_SM121_RCP_TABLE`: 8,388,640 bytes, SHA-256
`d2e557543f6bc51f5141ba6414000cd8ed892e2e915eda19245c3cae22c16b39`.
The loader verifies its layout and hash with Windows CNG and adds one device
copy plus a transient host buffer. It reuses the Q1 exponential table and
keeps the owned prefix and decode-tail KV allocations separate. This adds
no offline artifact or runtime library. The arithmetic matches all 4096
context values in each original-input q8191/q7169 first-decode GPU replay;
the integrated product path still requires frozen-token qualification.

`QRT_FLA_GDN_SM121_EXP2_TABLE` selects an experimental, model-independent
`ex2.approx.f32` table for the optional Blackwell GDN state implementation.
The builder enumerates all 2,139,095,041 nonpositive finite FP32 inputs and
negative infinity on GB10. No prompt, token, weights or expected model output
is an input to construction. Packed integer deltas preserve every output bit.

The generated artifact is 183,174,448 bytes, SHA-256
`f490940df2bd80421159a96424c3e922330b7ca120d5ae7b629a973b9183730b`.
This adds about 175 MiB of resident device memory, the same transient host
buffer during loading, and one optional data file to any future package.
The native loader validates the layout and fixed SHA-256 using Windows CNG;
`bcrypt.dll` is supplied by Windows. No Python/CUDA library enters inference.
The offline builder reuses the existing reference container's Torch, NumPy and
Triton, with the exact image/source/PTX identities retained in its record.

The concrete benefit is reproducing the reference's gate and old-state decay
rounding, which has been isolated from matrix accumulation in a complete
384-token state replay. Ordinary mathematically rounded exp2 is insufficient
to reproduce those raw states. This is not yet whole-model or release
acceptance; Windows component and product tests determine retention.

The table is opt-in and the default arithmetic is unchanged. Removing the
environment binding disables it in a fresh process. Removing the optional
loader, lookup and builder removes its packaging cost entirely. The data is
not committed as a large source-tree binary; a qualified package must include
the fingerprinted artifact and account for its load time and memory.

## Optional SM121 normalization compatibility data

`QRT_FLA_GDN_NORM_BLACKWELL=1` uses the reference's 16-lane, eight-contiguous-
dimension sum order and requires `QRT_FLA_GDN_SM121_RSQRT_TABLE`. The additional
artifact has 17,301,808 bytes (16.50 MiB), SHA-256
`ca0230a8bae9bd101ac368f8a7c34007cda637df6513dbe4714253c36b940850`.
It contains compressed reciprocal-square-root values on [1,4). Exponent
scaling, subnormal flushing and +infinity were exhaustively checked against
all 2,139,095,041 nonnegative encodings on SM121; construction accepts no
model or prompt input. Its packed native lookup also matches all 229,408
independently captured Q/K reciprocal roots from the real q7169 layer.

The concrete benefit is removing two independently measured normalization
differences: the FP32 sum tree and the device reciprocal-root approximation.
High-precision host reciprocal roots still move 98 BF16 outputs in this
capture. The packaging cost is one optional 16.50 MiB file, an equal device
allocation and transient host buffer during SHA verification. The same
Windows CNG and offline builder dependencies described above apply. The
normalization route passes the complete frozen q7169 Windows component
comparison and remains opt-in pending full-model qualification.


## Optional model-parameter GDN gate table

`scripts/capture_sm121_gating_table.py` extracts the original fused gate
function from a fingerprinted source file. A real-token terminal control
must match exactly before enumeration. Table construction then uses every
BF16 input encoding and the actual model's A_log/dt_bias parameters; prompts
and expected model outputs do not select table entries.

The initial layer-zero table contains 2,097,152 FP32 values in an 8,388,608-byte
head-major file, SHA-256
`fb8afb17901d4c6a49a7be47bb19059f93f2aa3caec3625e7300f02e439e8bc4`.
The shared BF16 sigmoid file is 131,072 bytes, SHA-256
`32923b94eca938cd0f966f39efb5fcbda40f2c2bb748cdd90bfd3ddeff9e8f97`.
The concrete benefit is matching SM121 softplus, exp and sigmoid rounding
at the projection-to-GDN boundary. These tables and independent direct
execution both reproduce all 32 G and beta control values bit for bit.

The existing optional `QRT_QWEN36_GB10_GATE_LUT_DIR` diagnostic uses host
lookups. Its current per-layer cache retains 8 MiB plus a 128 KiB sigmoid
copy. All 30 linear-attention layers have now been constructed, using 240 MiB
plus a shared 128 KiB file on disk; the current loader duplicates the
sigmoid cache per layer. The offline dependencies are the same pinned
Torch/NumPy/Triton image; no CUDA/Python dependency enters Windows inference.
Diagnostic commands must pin fingerprints and parameter provenance because the existing raw
loader validates size, not model identity. Release use remains contingent on
model binding, artifact verification, load time and real-model qualification.

The optional Q1 decode implementation reuses those 30 model gate files and
the existing exponent, reciprocal-root, SiLU and gated-normalization tables.
It adds a 262,144-byte model-independent FP32 sigmoid file, SHA-256
`cafc4dc1012d534a6a99739f9c4a327bc4052ad2d3441a79dff18f9a2ea7ebc5`,
enumerated over all BF16 inputs using the pinned original decode expression.
The benefit is preserving FP32 beta and normalized Q/K through the recurrent
update; the former BF16 beta route changes real captured states. Native HIP
replay matches every saved output and state in the q7169 operator control.

The Q1 loader validates the three common files with Windows CNG SHA-256 and
layout checks, allocating 200,738,400 bytes (191.44 MiB) on the device. It
also uploads 8 MiB per model gate layer, up to another 240 MiB, retaining the
existing host cache. SiLU and gated-normalization tables share their existing
process caches. Common exponent/root tables can duplicate allocations owned
by the separate prefill provider. The incremental offline artifact is the
256 KiB FP32 sigmoid file; Windows inference gains no CUDA, Python or new
third-party runtime dependency. Q1-specific path variables isolate this
choice from prefill. Model-file binding, total load cost and full-model
qualification remain release requirements.

The optional Q1 MoE path reuses the existing model-independent CUDA router
fraction and BF16 SiLU files. Q1-specific aliases are
`QRT_QWEN36_Q1_SM121_ROUTER_TABLE` and
`QRT_QWEN36_Q1_SM121_MOE_SILU_TABLE`. Their sizes are 33,554,432 and
131,096 bytes, with SHA-256
`b2a42c4a626469c986e33e43f16f41bde9d84de94347cd9ceaa1bd68d36bcdf0` and
`97a2a729266bb0681983aa5b2c6ddffafeaab1bab99c7658a0524b09331a11ac`.
The concrete benefit is reproducing all 176 original decode routing weights
and all shared activation endpoints in 22 observed rows. The Q1 loader uses
Windows CNG and adds 33,685,528 bytes of device storage, potentially duplicating
the separate prefill provider's copies. It reuses Q1 sigmoid/root storage.
No new offline artifact or Windows runtime library is introduced; packaging
and load-time qualification still apply.

## Optional model embedding inverse scales

`scripts/capture_sm121_embedding_scales.py` precomputes one FP32 inverse
RMSNorm scale for each of the model's 248,320 immutable embedding rows.
This removes the observed layer-zero reduction/reciprocal-root discrepancy
without a prompt-specific correction. The proposed file is 993,280 bytes;
the native loader keeps that host table and uploads four bytes per prompt
token. It uses the existing optional
`QRT_QWEN36_GB10_LAYER0_RMSNORM_SCALE_LUT_PATH` surface.

Construction uses the pinned SM121 Torch/NumPy/Triton environment and reads
the embedding tensor on CPU in bounded chunks. It first validates the
complete real-token norm control, then computes all entries solely from model
parameters and the fixed epsilon. No CUDA, Python or new third-party library
is added to Windows inference. Release packaging still requires table/model
fingerprints and real-model qualification. Construction passes with table SHA
`f4e37f759c586bfc8fcc4d74cefdd89235f0f0c0c90cd286147e331e87509e67`.
Both complete real-token controls pass, including a second application after
all vocabulary rows are enumerated. The native command verifies the complete
embedding and norm-weight tensor fingerprints before use; its captured full
input normalization matches every reference BF16 cell. The raw runtime loader
still needs equivalent artifact/model binding for release packaging.

The optional SiLU endpoint builder enumerates every finite FP32 input of the
original SM121 Triton expression, after a real convolution control passes.
It stores every BF16 output transition, including nonmonotonic transitions,
and a page directory, then rechecks the packed lookup over the full domain.
This would replace cross-vendor exponential/division differences without
depending on model weights or prompt values. The complete table is 648,036
bytes, with SHA `673f8dd1280700578c1e8743afd2e3b4da134b1fbd463c890527e1c4d9f796b8`.
The optional Windows loader validates its layout and SHA using the operating
system's existing BCrypt API, and shares one immutable device copy per provider
process. CUDA/Python is confined to offline construction. Native qualification
and release inclusion remain pending.


## Optional attention reciprocal compatibility

The captured-input attention diagnostic can reproduce SM121 approximate
reciprocals using an 8,388,640-byte coefficient file, SHA
`d2e557543f6bc51f5141ba6414000cd8ed892e2e915eda19245c3cae22c16b39`.
It stores signed one-ULP deltas from correctly rounded reciprocal for all
8,388,608 normalized FP32 mantissas. Construction independently verifies all
159,383,552 positive inputs at exponents 0 through 18; the same coefficients
cover attention denominators throughout the supported context range. No model,
weight, prompt, hidden value, logit or generated token enters construction.

The concrete benefit is matching the `div.full.f32` reciprocal-multiply
endpoint of the pinned original attention implementation. Native diagnostic
use adds 8 MiB of immutable device storage and an equal offline artifact,
validated with existing Windows BCrypt and explicit format bounds. It adds no
CUDA, Python or third-party runtime dependency to Windows. The current retained
profile is unchanged; component and real-model qualification are separate.
