# Optional SM121 exponential compatibility data

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
