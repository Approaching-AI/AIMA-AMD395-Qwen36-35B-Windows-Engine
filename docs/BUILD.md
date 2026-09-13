# Building on Windows

## Validated toolchain

- Windows 11 x64 on AMD Ryzen AI Max+ 395 (`gfx1151`)
- Visual Studio 2022 Build Tools with MSVC x64 and Windows SDK
- Rust 1.95 (`x86_64-pc-windows-msvc`)
- AMD ROCm HIP SDK 7.1
- WSL2 Ubuntu 24.04 with Python and Triton 3.6
- AMD Composable Kernel from ROCm/aiter v0.1.13, pinned below
- Python 3.10 or newer

The WSL/Triton environment is needed for q1024 and q32..q4096 smooth-tail AOT
generation and for optional q8192 regeneration experiments. The release build reuses the tracked,
correctness-accepted q8192 code objects and recompiles their host provider;
freshly regenerated q8192 bytes require a new real-model GB10 acceptance before
they can replace that inventory. The resident process itself is native Windows
and does not run under WSL.

## Full runtime build

From a Developer PowerShell:

```powershell
git clone --branch v0.1.13 --recurse-submodules `
  https://github.com/ROCm/aiter.git C:\src\aiter-v0.1.13
git -C C:\src\aiter-v0.1.13 checkout cdcfa833bdf554ca75594c90dde4316ea9b50199
git -C C:\src\aiter-v0.1.13\3rdparty\composable_kernel checkout `
  fdf4bb7fcc984811cef48ce817d89aac064b984a

git clone https://github.com/Approaching-AI/AIMA-AMD395-Qwen36-35B-Windows-Engine.git
Set-Location AIMA-AMD395-Qwen36-35B-Windows-Engine

.\scripts\build-runtime.ps1 `
  -CkRoot C:\src\aiter-v0.1.13\3rdparty\composable_kernel `
  -RocmRoot 'C:\Program Files\AMD\ROCm\7.1' `
  -WslDistribution Ubuntu-24.04 `
  -TritonPython /opt/qwen36-vllm/bin/python `
  -OutDir build\runtime
```

The orchestrator builds:

1. the native whole-model provider;
2. the exact q1024 arbitrary-length selected-MoE provider and kernels;
3. the q8192 selected-MoE provider against the accepted tracked kernels;
4. q32, q64, q128, q256, q512, q1024, q2048, and q4096 smooth-tail
   selected-MoE providers used to keep non-aligned prompt lengths continuous;
5. CK FMHA and AITER fused-GDN providers;
6. the complete tracked q1/base AOT inventory under `aot/gfx1151`;
7. the Rust resident server/lifecycle CLI;
8. the native `qrt-product.exe` token/logit/prefix acceptance CLI with a
   256 MiB Windows stack reserve; and
9. `runtime-manifest.json` with commit, dirty state, paths, sizes, and SHA256.

When `runtime.env` is loaded from a built runtime, `qrt` validates the complete
`smooth-tail/q32` through `smooth-tail/q4096` inventory and binds all provider
paths automatically. A partially copied tree is rejected during startup rather
than silently falling back to a length-discontinuous route. An unpackaged tree
can be selected explicitly with `--smooth-tail-moe-root`.

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
.\scripts\baiying_build_smooth_tail_moe.ps1 -Tokens 256 -OutDir build\smooth-tail\q256
.\scripts\baiying_build_ck_fmha_q8192.ps1 `
  -CkRoot C:\src\aiter-v0.1.13\3rdparty\composable_kernel
.\scripts\baiying_build_aiter_fused_gdn_q8192.ps1 -BuildDir build\gdn
.\scripts\baiying_build_qrt_server.ps1 -OutDir build\engine
.\scripts\build-product-cli.ps1 -OutDir build\product-cli
```

Use each script's `Get-Help`/parameter declaration for optional paths and
timeouts. Builds are bounded and record provenance rather than invoking an
unbounded remote job.

The qualified CK source boundary is AITER commit
`cdcfa833bdf554ca75594c90dde4316ea9b50199` with CK commit
`fdf4bb7fcc984811cef48ce817d89aac064b984a`. Other CK revisions may change the
FMHA host/device API or architecture tags; treat them as unqualified until the
component smoke and native Windows GB10 product gate are rerun.

### Experimental Triton/FLA GDN

`baiying_build_fla_gdn.ps1 -OutDir build\fla-gdn -TimeoutSeconds 240`
builds only the optional chunk-64 FLA provider and q64/q65/q7169 probes. It
does not change the packaged AITER route or launch inference. Run each probe
separately with `baiying_guarded_inference.ps1`, then attach the GB10 numerical
comparison before any product measurement. Generated `qrt_fla_gdn_kernel_specs.inc`
binds the native launch sizes to the exact new AOT compiler output; old
FlashInfer-order binaries have a different ABI and cannot be substituted.
If WSL is unavailable, pass `-AotDir <directory>` containing the same generator's
Linux cross-compilation output. The builder checks target, generator hash,
every kernel hash/size and the generated launch-header hash before using it.

The September 9 reference-service audit found GB10 uses `forward_native`
Triton/FLA, not the SM90-only FlashInfer backend. The restored optional route
therefore uses log-gate cumsum, BF16 beta/K and W/U boundaries, and the FLA
state/output decomposition. Every recurrent dispatch is limited to 1024
tokens; aligned segments hand off unrounded F32 state, and only a ragged final
chunk is padded with neutral inputs. This is a route correction under
qualification, not a claim that the q7169 model gate has passed.

## CPU-safe validation

On macOS/Linux or Windows with Make, a C11 compiler, Rust, and Python:

```shell
make check
```

This runs the C ABI smoke, Rust tests, clippy with warnings denied, Python
contract/API/evaluation tests, and the public-tree privacy/license hygiene
scan. It does not claim GPU inference success.

### Guarded Windows experiments

Use `scripts/baiying_guarded_inference.ps1 -SpecPath <spec.json> -OutDir
<new-directory> -TimeoutSeconds 90` on baiying for a recovered candidate.
The JSON spec contains `executable`, `working_directory`, and an `arguments`
array; include `repo_commit` to check source identity. `-PreflightOnly` checks
the inputs and host without launching the executable. Use absolute paths.

The runner holds a machine-wide experiment mutex, rejects an existing engine
or compiler, reserves 8 GiB of host memory and 20 GiB of commit headroom,
limits logs to 64 MiB, and places the process tree in a kill-on-close Windows
job. Logs are written during execution. Its record distinguishes process exit,
CLI summary status, and post-run host checks; exit zero with no passing
summary is not inference success. Attach the external GB10 oracle separately.

Hawkeye correction admission now covers every common-launcher caller,
including QKV/Z. It counts candidates before exact-dot work and rejects more
than 131072 candidates or 64 candidates in one block. After each synchronized
dispatch it stops further work above 100 ms per dispatch or 10 seconds for
the correction. These diagnostic limits cannot be raised through environment
overrides. They leave accepted arithmetic unchanged. A supervisor cannot
interrupt an already hung GPU kernel or recover a hard-locked Windows host;
candidate admission is required before submitting expensive GPU work.
Set `QRT_QWEN36_HAWKEYE_CORRECTION_COUNT_ONLY=1` to count and stop at the first
correction without launching any exact dots, including when admission passes.
This intentional diagnostic failure produces no accepted inference output.

F32 projection overrides must allocate their output even when the selected
convolution consumer normally fuses a BF16 input. The layer-2 QKV override
previously wrote through an omitted F32 buffer, before reaching correction
admission. The current source allocates the producer buffer, validates both
producer and required consumer pointers before launching WMMA/dot2, and
materializes the BF16 consumer after projection. On 2026-09-09, source
`c36e267` passed native Windows synthetic safety checks and three real-model
layer-2 count-only exits, including two using the exact original asynchronous
failure profile. All counted 6134883 candidates and rejected the excessive
work before exact-dot dispatch, with post-run host checks passing. This is a
verified repair of the known reproduction, not a proof that every possible
driver or engine failure is prevented. The same provider passed the q8192
32-token/logit/stream boundary, while ordinary q7169 remains numerically
incorrect (220 versus GB10 82). No arbitrary-length acceptance follows from
safe diagnostic exits.
An additional full-forward run with detailed route logging confirms all six
QKV/Z WMMA projections across layers 0--2 reach downstream model work and
normal cleanup. It still returns 220 and is not GB10-correct inference.

The model-free native regression builds the actual provider translation unit:

```powershell
.\scripts\baiying_build_whole_provider.ps1 -ProjectionSafetyTest -OutDir build\projection-safety
```

Run the resulting `qrt-projection-safety.exe` through the guarded runner,
with exactly one of `--host-only`, `--small`, or `--full-shape` as its argument.
Use separate output directories and inspect each record before increasing the
shape. `--host-only` checks invalid launch arguments without calling HIP;
`--small` checks eight WMMA/F32-to-BF16 cases across row/token tile boundaries;
`--full-shape` checks a synthetic 8192-row, 7169-token projection without model
loading or Hawkeye exact-dot correction. Output redzones and every conversion
cell are checked; exact BF16 reference projection checks cover every small-case
cell and 512 full-shape cells. Unrounded WMMA-versus-host F32 differences are
reported separately, not mistaken for BF16 endpoint failures. The initial
strict-F32 synthetic probe found such differences and was not accepted as a
passing test. Each producer is synchronized before its consumer.
These are safety regressions, not GB10 or real-model inference acceptance.

The experimental dense replay option
`QRT_QWEN36_HAWKEYE_PREPARED_OPERANDS=1` prepares lossless BF16 operand views
for unchanged-plan projections with1024–8192 tokens,1024–9216 rows and
K16-aligned reduction sizes up to4096. Each ineligible row uses original
arithmetic. The option defaults to0 and does not apply to packed candidates,
device-count replay, short changed dot plans or decode. Its bounded workspace
is released after every correction call.

For its native checks, compile `tests/native/prepared_projection_selftest.cpp`
with the same DPP/compact-normalization settings as the provider. The existing
`-ProjectionSafetyTest` executable also accepts `--prepared-correction` with
the option enabled; it checks the actual selector and replay entry point,
including mixed fallback rows and guarded outputs. Follow component checks
with the original QKV reference and complete GB10 product request before
retaining a performance result.

`QRT_QWEN36_HAWKEYE_ABSOLUTE_PRODUCT_BOUND=1` enables experimental per-cell
absolute-product metadata on the prepared route when both L2 bounds are
available and the caller has not supplied absolute sums. Excluded BF16 rows
receive infinite bounds; midpoint radii and PPB do not change. Its WMMA
implementation has passed native and original QKV checks but exceeded a
q8192 dispatch deadline before producing any token. Leave it disabled for
the selected product stack.

The follow-up `QRT_QWEN36_HAWKEYE_ABSOLUTE_PRODUCT_HIPBLASLT=1` option uses
the existing resident matrix dependency with immutable BF16 magnitude views.
It is active only with the bound option. A flat window owns up to64MiB plus
two partial token columns, and a separate bounded allocation holds magnitude
operands. Its completed matrix window has a250ms deadline, including backend
setup; exact-dot/WMMA dispatches retain100ms and the aggregate correction
limit remains10000ms. All setup remains in actual product TTFT.
The native `--absolute-product-hipblaslt` mode checks independent
double sums, magnitude views and guards; `--absolute-bound-correction` checks
the actual launcher with either backend. The native and original QKV checks
pass, but the completeq8192 continuation fails from output index115. Keep
absolute-product admission disabled. A tested upper bound on the product sum
does not independently validate the projection-error coefficient.

The optional `QRT_CK_SM121_FINAL_PV_BOUND=1` route accumulates nonnegative PV
error metadata and applies its conservative inflation at the end. It defaults
to 0 and is active only for compact PV replay modes 1/3 (layouts 22/24), with
the complete query span at or below 8192 tokens. Other spans and Q1 retain
their original route. Native matrix accumulation, online K32 rescaling,
reciprocal and exact replay remain unchanged; exceptional metadata selects
exact replay. No additional dependency or workspace is required.

Compile `tests/native/final_pv_bound_selftest.cpp` for the numerical envelope
check and `tests/native/final_pv_kernel_selftest.cpp` for the complete native
PV check. The latter takes the SHA-verified reciprocal table as its argument.
Use the provider's DPP/compact-normalization/compact-exp2 compiler settings.
The attention replay tool accepts `QRT_ATTENTION_REPLAY_FINAL_PV_BOUND=0|1`
for original-tensor comparisons. Both settings have passed the complete
q8192/out512 GB10 request; this does not qualify a new package or release.

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
The q8192 component build runs the ROCm-bundled `llvm-strip --strip-debug` on
its selected-MoE code objects, rewrites the base AOT metadata hashes and sizes,
and rejects any packaged HSACO that still contains a private home path. This
removes compiler debug paths only; the retained objects must still pass the
dynamic-length smoke and the real GB10 numerical boundary before release.
The copied base AOT directory is a runtime dependency, not merely build
provenance: pass it as
`QRT_PREFILL_DESCRIPTOR_BATCH_Q1_MOE_TRITON_0626_MODULE_DIR` as shown in the
top-level quick start. Related q1 projection/attention loaders resolve their
qualified objects from the same directory.

## Portable profiles in the unreleased server

Runtime paths in an env file can start with the literal `${RUNTIME_DIR}`.
The server resolves this marker against the directory containing that env
file, including when `start` launches from another working directory. It does
not expand other environment variables. For example:

```dotenv
QRT_FLA_GDN_SM121_EXP2_TABLE=${RUNTIME_DIR}/tables/sm121-exp2-negative-f490940d.bin
```

Use forward slashes after the marker. Parent traversal and rooted suffixes
are rejected; external model paths remain explicit command arguments.
Literal values and later `--set-env` overrides retain their existing behavior.
All entries are parsed before any profile variables are applied.

For a verified runtime containing the separate SM121 attention DLL, pass
`-CkProviderRelativePath 'ck-fmha/qrt_ck_fmha_sm121.dll'` to
`scripts/package-runtime.ps1`. The chosen DLL must be present in the hashed
runtime manifest. The packaging default remains the existing continuous
attention DLL. Source support and local tests do not establish archive
relocation or model acceptance; run the extracted artifact on baiying.
