# Building on Windows

## Validated toolchain

- Windows 11 x64 on AMD Ryzen AI Max+ 395 (`gfx1151`)
- Visual Studio 2022 Build Tools with MSVC x64 and Windows SDK
- Rust 1.95 (`x86_64-pc-windows-msvc`)
- AMD ROCm HIP SDK 7.1
- WSL2 Ubuntu 24.04 with Python and Triton 3.6
- AMD Composable Kernel source checkout
- Python 3.10 or newer

The WSL/Triton environment is needed for q1024 AOT generation and for optional
q8192 regeneration experiments. The release build reuses the tracked,
correctness-accepted q8192 code objects and recompiles their host provider;
freshly regenerated q8192 bytes require a new real-model GB10 acceptance before
they can replace that inventory. The resident process itself is native Windows
and does not run under WSL.

## Full runtime build

From a Developer PowerShell:

```powershell
git clone https://github.com/skyguan92/AIMA-AMD395-Qwen36-35B-Windows-Engine.git
Set-Location AIMA-AMD395-Qwen36-35B-Windows-Engine

.\scripts\build-runtime.ps1 `
  -CkRoot C:\src\composable_kernel `
  -RocmRoot 'C:\Program Files\AMD\ROCm\7.1' `
  -WslDistribution Ubuntu-24.04 `
  -TritonPython /opt/qwen36-vllm/bin/python `
  -OutDir build\runtime
```

The orchestrator builds:

1. the native whole-model provider;
2. the exact q1024 arbitrary-length selected-MoE provider and kernels;
3. the q8192 selected-MoE provider against the accepted tracked kernels;
4. CK FMHA and AITER fused-GDN providers;
5. the complete tracked q1/base AOT inventory under `aot/gfx1151`;
6. the Rust resident server/lifecycle CLI; and
7. `runtime-manifest.json` with commit, dirty state, paths, sizes, and SHA256.

The q8192 build submits 17 asynchronous calls through a 16-event ring and
requires all outputs to match the synchronous fixed hash. This catches stale
providers that reject valid long-context work instead of applying bounded
backpressure.

## Component builds

Each component script accepts explicit output/toolchain paths:

```powershell
.\scripts\baiying_build_whole_provider.ps1 -OutDir build\whole
.\scripts\baiying_build_triton_moe_q1024_exact.ps1 -OutDir build\q1024
.\scripts\baiying_build_triton_moe_q8192.ps1 -BuildDir build\q8192 -OutDir build\q8192
.\scripts\baiying_build_ck_fmha_q8192.ps1 -CkRoot C:\src\composable_kernel
.\scripts\baiying_build_aiter_fused_gdn_q8192.ps1 -BuildDir build\gdn
.\scripts\baiying_build_qrt_server.ps1 -OutDir build\engine
```

Use each script's `Get-Help`/parameter declaration for optional paths and
timeouts. Builds are bounded and record provenance rather than invoking an
unbounded remote job.

## CPU-safe validation

On macOS/Linux or Windows with Make, a C11 compiler, Rust, and Python:

```shell
make check
```

This runs the C ABI smoke, Rust tests, clippy with warnings denied, Python
contract/API/evaluation tests, and the public-tree privacy/license hygiene
scan. It does not claim GPU inference success.

## Model files

The runtime expects `config.json`, `tokenizer.json`, tokenizer metadata, and
the BF16 safetensor shards in one model directory. Model data is never copied
into build output or release archives.

## Reproducibility boundary

Generated code-object bytes can change with Triton/ROCm compiler revisions.
The release therefore publishes AOT objects, source generators, metadata,
hashes, and numerical smoke gates. `build-runtime.ps1` passes the tracked
`native/aot/gfx1151` directory through `-ReuseAotDir`; invoking the component
script without that option deliberately regenerates AOT for investigation. A
byte-different rebuild is acceptable for release only after the same real-model
external correctness boundary passes; self-hashes alone are not authority.
The copied base AOT directory is a runtime dependency, not merely build
provenance: pass it as
`QRT_PREFILL_DESCRIPTOR_BATCH_Q1_MOE_TRITON_0626_MODULE_DIR` as shown in the
top-level quick start. Related q1 projection/attention loaders resolve their
qualified objects from the same directory.
