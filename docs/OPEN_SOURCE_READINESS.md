# Open-source readiness audit

Audit date: 2026-08-15

## Decision

Version 1.0.1 is suitable for public source publication for its declared,
model-specific Windows target. The runtime source, AOT inventory, build entry
points, lifecycle/API tests, licenses, and redacted real-model evidence are
present in this repository. Model weights and vendor runtimes remain external.

## Release gates

| Area | Result | Published basis |
|---|---|---|
| Native source and ABI | Pass | C/C++/HIP core plus Rust server/CLI |
| Windows build | Pass | clean MSVC/Rust/HIP build scripts and manifests |
| Local regression | Pass | C smoke, Rust tests/clippy, Python tests, hygiene |
| Lifecycle | Pass | detached start, independent status, graceful stop/PID exit |
| OpenAI surface | Pass | JSON/SSE completion/chat, tools and continuation |
| Request pressure | Pass | bounded FIFO concurrency, queue metrics, overload errors |
| Context behavior | Pass | continuous inputs, maximum boundary, explicit overflow rejection |
| Prefix cache | Pass | seed, COW hit, resident hit, A-B-A isolation |
| q8192 product target | Pass | 3,852.909 ms and 2,126.186 tok/s |
| q8192 neighbor continuity | Pass | 18/18 GB10 matches; worst ratio 1.423x |
| Load bound | Pass | 19,940.245 ms, below 30 seconds |
| Correctness | Pass | external BF16 first-token/logit and long continuation |
| MMLU-Pro | Pass | both 7,486 / 12,032; zero projection mismatch |
| Licensing/privacy | Pass | Apache-2.0, upstream notices, clean public-tree scan |

## Scope limitations

- Only Windows 11 x64, Ryzen AI Max+ 395 (`gfx1151`), the declared BF16 model,
  and batch size 1 are qualified.
- Greedy sampling is supported; unsupported OpenAI parameters fail explicitly.
- ROCm runtime DLLs, model data, CK source, and build toolchains are not
  redistributed by the source repository.
- Performance evidence applies to the recorded platform/profile and is not a
  universal hardware claim.

## Evidence discipline

Real inference claims identify the platform, command family, model reference,
output/token result, commit, and correctness attachment. Transport-only tests,
stubs, estimates, proxy kernels, and engine self-hashes are never presented as
inference success. Public evidence is sanitized to remove private addresses,
personal paths, credentials, raw model data, and copyrighted evaluation text.
