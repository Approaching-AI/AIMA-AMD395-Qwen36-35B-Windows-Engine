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
