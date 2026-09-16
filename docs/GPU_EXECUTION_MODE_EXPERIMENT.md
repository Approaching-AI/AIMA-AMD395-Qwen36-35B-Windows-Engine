# Native GPU execution-mode comparison

Source `4f60cfdbefadeb768baf9afbfd90a614726105aa` adds an optional
`-GpuExecutionMode default|wgp|cu` to the whole, CK, FLA and MoE builders.
The default passes no new flag. Explicit modes pass `-Xarch_device` with
`-mno-cumode` or `-mcumode`, and record the choice in build provenance.
No GPU-global setting, arithmetic source or numerical threshold changes.

On baiying, all eight bounded native builds pass. The actual embedded kernel
descriptors confirm 485 whole, 66 CK, 29 FLA and 57 MoE entries in the requested
mode, all wave32. Duplicate CK symbols in separate device objects remain
separate entries. Mode is read from the documented descriptor field;
see [LLVM's AMDGPU descriptor documentation](https://llvm.org/docs/AMDGPUUsage.html#code-object-v3-kernel-descriptor).
Static resources change with compilation, but these records do not measure
occupancy, bank conflicts or stage-level runtime. Twelve FLA and ten MoE
precompiled AOT/header artifacts remain identical between modes.

The same-source cold q8192/out512 pair uses `D:\models\Qwen3.6-35B-A3B`, the
existing CLI and external libraries, Dense1000/MoE512, original K16 replay,
staged/expert-order MoE and coarse full-attention OUT. Coarse linear OUT,
GDN interval/fusion and selective QK tail stay explicitly disabled in both
arms. Each process starts fresh. Every original GB10 prompt and output ID,
first logit within0.125, all512 actual callbacks and host guard checks pass.

| In-tree HIP mode | Load ms | TTFT ms | TPOT ms | First logit | GB10 IDs |
| --- | ---: | ---: | ---: | ---: | ---: |
| WGP | 21407.9495 | 27936.2702 | 100.365068 | 10.375 | 512/512 |
| CU | 21216.1972 | 27730.1676 | 101.981920 | 10.375 | 512/512 |

This single pair observes206.1026 ms lower CU TTFT and1.616852 ms higher CU
TPOT. Keep the build default and retained stack unchanged. The difference
does not establish a repeatable improvement sufficient for the mission.
Both TTFT values exceed10000 ms; the retained4187.415605 ms /
1506.407263 tok/s target remains unchanged. No other prompt, prefix,
long-context, package or release qualification transfers.

Local validation covers C ABI smoke, public hygiene and the diff. Windows
parsing and all native compilations pass; the MoE builders also complete
their existing full-shape smoke checks. This change touches only four build
scripts, so the earlier full code suite is not claimed as a new test run.

Evidence: [native builds, descriptors and both product boundaries](../benchmarks/correctness/gpu-execution-mode-product-20260916.json),
SHA256 `e840d43358e81e1eda862569466ac48b58b74b4de42f64070f8fa716d5ea9c4e`. The record includes the actual commands, model, commit,
DLL hashes, original numerical oracle, observations and scope limitations.
