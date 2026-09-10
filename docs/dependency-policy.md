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
normalization route remains opt-in pending Windows and full-model evidence.
