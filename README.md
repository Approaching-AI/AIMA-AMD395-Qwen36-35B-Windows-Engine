# AIMA AMD395 Qwen3.6 35B Windows Engine

A native, model-specific Windows inference engine for
`Qwen3.6-35B-A3B-BF16` on AMD Ryzen AI Max+ 395 (`gfx1151`). It provides a
resident batch-one runtime, an OpenAI-compatible HTTP API, and a structured
`qrt` command-line lifecycle.

[中文说明](README.zh-CN.md) ·
[Linux companion](https://github.com/skyguan92/AIMA-AMD395-Qwen36-35B-Linux-Engine)

## What is included

- Native C/C++/HIP inference and model loading, with generated `gfx1151` AOT
  kernels.
- Rust `qrt serve`, `qrt start`, `qrt status`, and `qrt stop` commands.
- `/v1/models`, `/v1/completions`, and `/v1/chat/completions` compatibility.
- JSON and SSE streaming responses, including streamed function tool calls.
- Deterministic greedy decode, tool-result continuation, and 512-token output.
- Continuous prompt lengths up to the configured total-context limit.
- Prefix-cache seed, copy-on-write reuse, resident reuse, and isolation checks.
- A bounded FIFO request queue: one active batch-one request and configurable
  waiting depth/timeout, with explicit `429`/`503` overload behavior.
- Source-build, verification, performance, and MMLU-Pro evidence.

This is intentionally not a generic graph runtime. The kernels, layouts, and
provider contract target one model family and one AMD GPU architecture.

## Qualified platform

- Windows 11 x64
- AMD Ryzen AI Max+ 395 (`gfx1151`)
- AMD ROCm HIP SDK 7.1 and a compatible driver
- `Qwen3.6-35B-A3B` BF16 model files obtained separately

Model weights and AMD runtime binaries are not included.

## Quick start

Download the v1.0.1 runtime archive from
[Releases](https://github.com/Approaching-AI/AIMA-AMD395-Qwen36-35B-Windows-Engine/releases),
or build it with [the source-build guide](docs/BUILD.md). Then run from
PowerShell:

```powershell
$rt = Resolve-Path .\AIMA-AMD395-Qwen36-35B-Windows-Engine-v1.0.1
$state = Join-Path $rt 'service.json'

& "$rt\engine\qrt.exe" start `
  --model 'C:\models\Qwen3.6-35B-A3B' `
  --provider "$rt\whole-provider\qrt_qwen36_whole_provider.dll" `
  --arbitrary-moe-provider "$rt\q1024-moe\qrt_triton_moe_q1024_exact_provider_slots64.dll" `
  --arbitrary-moe-kernel-dir "$rt\q1024-moe\moe-kernels" `
  --env-file "$rt\runtime.env" `
  --set-env "QRT_PREFILL_DESCRIPTOR_BATCH_Q1_MOE_TRITON_0626_MODULE_DIR=$rt\aot\gfx1151" `
  --set-env "QRT_PREFILL_DESCRIPTOR_BATCH_FULL_ATTENTION_CK_FMHA_DLL=$rt\ck-fmha\qrt_ck_fmha_continuous_long.dll" `
  --set-env "QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_TRITON_SELECTED_MOE_DLL=$rt\q8192-moe\qrt_triton_moe_q8192_provider.dll" `
  --set-env "QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_TRITON_SELECTED_MOE_KERNEL_DIR=$rt\q8192-moe" `
  --set-env "QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_AITER_FUSED_GDN_DLL=$rt\aiter-gdn\qrt_aiter_fused_gdn_q8192_provider.dll" `
  --set-env "QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_AITER_FUSED_GDN_KERNEL_DIR=$rt\aiter-gdn" `
  --model-id qwen3.6-35b-a3b `
  --host 127.0.0.1 --port 8000 `
  --max-model-len 262144 `
  --max-queue-depth 64 --queue-timeout-seconds 600 `
  --state-file $state --log-file "$rt\service.log"
```

The process remains resident after the launching terminal exits. Operate the
exact recorded PID through the shared state file:

```powershell
& "$rt\engine\qrt.exe" status --state-file $state
& "$rt\engine\qrt.exe" stop --state-file $state --wait-seconds 120
```

Use `--api-key` before binding beyond loopback. Unauthenticated non-loopback
listeners are rejected unless explicitly enabled.

## OpenAI client example

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="placeholder")
stream = client.chat.completions.create(
    model="qwen3.6-35b-a3b",
    messages=[{"role": "user", "content": "Explain prefix caching briefly."}],
    max_completion_tokens=128,
    temperature=0,
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

See [API and operational behavior](docs/API.md) for tool calls, errors, queue
semantics, context limits, and health fields.

## Published qualification

The retained q8192 real-model result on the qualified Windows platform was:

| Metric | Result | Acceptance bound |
|---|---:|---:|
| Model + engine load | 19,940.245 ms | <= 30,000 ms |
| TTFT | 3,852.909 ms | <= 4,187.416 ms |
| Prefill | 2,126.186 tok/s | >= 1,506.407 tok/s |
| Decode | 30.551 tok/s | >= 28.168 tok/s |
| TPOT | 32.732 ms | <= 35.502 ms |

The v1.0.1 boundary repair was additionally qualified at q8191, q8192, and
q8193 with one- and two-token continuations, three cold-prefix repetitions per
shape. All 18 outputs matched the GB10 BF16 oracle exactly. q8192 TTFT was
3,872.773–3,928.488 ms; the worst neighbor ratio was 1.423x and the worst
positive residual was 1,642.653 ms.

The repository publishes all 12 retained performance rows, GB10-anchored token
correctness, maximum-context and prefix-continuation digests, OpenAI surface
acceptance, and the complete sanitized MMLU-Pro candidate/reference parity
data. See [performance](docs/PERFORMANCE.md),
[evaluation](docs/EVALUATION.md), and [benchmarks](benchmarks/README.md).

MMLU-Pro exact-answer accuracy was `7,486 / 12,032` (`62.2174%`) for both this
engine and the BF16 authority, with zero projection mismatch across all 12,032
questions.

## Build and test

```powershell
.\scripts\build-runtime.ps1 `
  -CkRoot C:\src\aiter-v0.1.13\3rdparty\composable_kernel `
  -OutDir build\runtime
```

CPU-safe checks also run on macOS/Linux:

```shell
make check
```

Target builds additionally require Visual Studio Build Tools, Rust, ROCm 7.1,
WSL2, and Triton 3.6. Details are in [BUILD.md](docs/BUILD.md).

## License

Project-authored material is Apache-2.0. Adapted or redistributed upstream
material remains under its stated permissive license. See [LICENSE](LICENSE),
[NOTICE](NOTICE), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
