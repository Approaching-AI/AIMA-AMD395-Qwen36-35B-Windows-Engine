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
copy. Extending to all 30 linear-attention layers would add 240 MiB plus a
shared 128 KiB file on disk; the current loader would also duplicate the
sigmoid cache per layer. The offline dependencies are the same pinned
Torch/NumPy/Triton image; no CUDA/Python dependency enters Windows inference.
Only layer zero has been constructed in this control. Diagnostic commands
must pin its fingerprint and parameter provenance because the existing raw
loader validates size, not model identity. Release use remains contingent on
model binding, artifact verification, load time and real-model qualification.
