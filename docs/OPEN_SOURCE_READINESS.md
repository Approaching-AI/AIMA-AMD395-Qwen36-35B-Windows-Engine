# Open-source readiness audit

Historical audit date: 2026-08-22

## Current unreleased candidate, 2026-09-13

The current candidate is **not release-ready**. The repaired r4 archive passes
all eight cold cases, including q7169, all five short HTTP fixtures, and the
declared HTTP and prefix-state checks. A broader attention correction bound
subsequently repairs the original 32k prefix plus 1024 suffix and 512-token
continuation. That newer configuration still needs renewed package, HTTP and
soak qualification. The experimental chunked cold route also passes the
declared 16k and q8192 boundaries, but remains disabled by default.

Current q8192 callback TTFT is about 62 seconds, above the unchanged 4.187-second
target and 10-second operating threshold. Larger contexts and retained
performance remain open. See [current measurements](PERFORMANCE.md),
[HTTP token/logit and stability evidence](HTTP_STABILITY.md), and
`benchmarks/correctness/prefix32k-admission-product-20260913.json`. The older
decision and tables below describe their recorded historical basis; they do
not approve promotion of the current candidate or imply a new publication.

## Decision

Version 1.0.1 remains suitable for public source publication for its declared,
model-specific Windows target. The runtime source, AOT inventory, build entry
points, lifecycle/API tests, licenses, and redacted real-model evidence are
present in this repository. Model weights and vendor runtimes remain external.

The current unreleased candidate is source-publication clean but is not yet
eligible for promotion to `main` or a release. Its nine q8192 selected-MoE AOT
objects reproduce byte-for-byte from neutral build roots after deterministic
debug stripping; its 32-shape component smoke, real layer-3 GB10 comparison,
Windows artifact transport check, and local C/Rust/Python/hygiene gates pass.
A fresh native Windows build and correctness-attached real-model product run on
`baiying`, from the exact candidate commit, remains mandatory. Component and
transport evidence do not substitute for that product gate. The candidate
record is
`benchmarks/performance/prefill-diagnostic-public-complete-aot-r1187-r1190.json`.

## Version 1.0.1 release gates

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
| q8192 neighbor continuity | Pass | 18/18 GB10 matches; worst ratio 1.036619x |
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
