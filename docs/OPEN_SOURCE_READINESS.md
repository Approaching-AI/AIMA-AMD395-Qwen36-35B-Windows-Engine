# Open-source readiness audit

Historical audit date: 2026-08-22

## Current unreleased work, 2026-09-18

The mission and release gates remain open. The latest experimental q8192
control is FLA `7b20c90`, whole `6e4908b`, CK `df2ea51`, MoE `9235750` and
CLI `24c4304`. A same-DLL OFF/ON/ON/OFF comparison matches all 2,048 original
GB10 output IDs, prompts, first logits and actual callbacks. Enabled median
TTFT is 24709.6139 ms and load median 21518.53165 ms. The 10000 ms operating
boundary and retained 4187.415605 ms target remain unchanged. These components
are not yet an updated portable package; full long-context and release
qualification do not transfer from earlier binaries.

The last assembled archive, `v1.0.2-current-stack.20260915.r4`, remains
unpublished. Its declared cold/prefix, protocol and one-hour soak evidence
belongs to its original server `4fca963` and R3 runtime assets. The later
Linux `.10` review found and repaired text-part tool results that could reset
failed retry windows. Latest server `b3d75e9` passes 54 native Rust tests and
45 real HTTP requests with the unchanged R4 runtime, including all 512 original
q8192 GB10 IDs, first logits, early callbacks and normal shutdown. Its cold
TTFT is 30231.4271 ms and load is 21529.5239 ms. New server packaging and soak
remain open. Windows continues to reject visual tool results explicitly.

See [current performance and component identities](PERFORMANCE.md),
[the Linux `.10` source comparison and protocol checks](SIBLING_REVIEW.md), and
[the latest server HTTP record](../benchmarks/correctness/tool-text-parts-bounded-native-http-20260918.json).
Earlier context results remain attached to their exact configurations; none
lowers the required context/performance targets. The historical audit below
is not approval to publish the current work.

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
