# Open-source readiness audit

Historical audit date: 2026-08-22

## Current unreleased work, 2026-09-19

The mission and release gates remain open. The latest experimental q8192
control is FLA `7b20c90`, whole `ddacdc9`, CK `1c2770d`, MoE `9235750` and
CLI `24c4304`. A same-DLL OFF/ON/ON/OFF comparison matches all 2,048 original
GB10 output IDs, prompts, first logits and actual callbacks. Enabled median
TTFT is 23353.80795 ms and load median 21299.42725 ms. Final-query and output
liveness preserve complete KV capture and remain default off. The 10000 ms operating
boundary and retained 4187.415605 ms target remain unchanged. These components
are not yet an updated portable package; full long-context and release
qualification do not transfer from earlier binaries.

The long CK control `ea6faff` passes the q8192/out512 inactive-scope regression
and four same-DLL cold16384/out32 runs. Its deferred bound reduces cold16k
TTFT from76932.886950 to73840.139151 ms, with all128 GB10 IDs matching.
The same DLL also passes actual16k/32k/64k prefixes plus1024 suffix tokens:
both512-token continuations, first logits, callbacks, state restoration and
changed-prefix rejection match the original references. Their hit TTFTs are
8459.984999/13210.9695/25254.1475 ms; original speed ceilings remain unmet.
These earlier component configurations do not qualify later whole-provider
or FLA binaries. Separate long cold continuations remain their own gates.
[Complete scope and evidence](LONG_FINAL_PV.md).

Whole `32a1b96` fixes a double-rounding error in BF16 RoPE shared by prefill
and decode. With FLA `1d11bf7`, CK `ea6faff`, MoE `9235750` and CLI `6d9602c`,
the repaired stack passes original q8192/out512 and the complete 128k prefix
case. Both original suffix512 requests, combined owner32, first logits,
actual callbacks, restoration, changed-prefix rejection and final host checks
pass. A separate cold16k/out512 request also passes every original ID and
callback. The 256k prefix test is active; separate cold32/64/128k tests remain
pending. These are functional results. The qualified q8192 median and all
performance targets above remain unchanged. See the
[complete correction and evidence](FLA_COMPLETION_GUARD.md).

The last assembled archive, `v1.0.2-current-stack.20260919.r6`, remains
unpublished. It pins server `b3d75e9`, CLI `6d9602c`, whole `ddacdc9`, CK
`ea6faff`, MoE `9235750` and FLA `2b33665`. Its 284 inventoried release files
and 268 runtime artifacts pass packaging and relocation. Its actual archive
passes 45 protocol requests, saved-prefix branches and 55 control-plane
requests. The same archive still requires the 13-case CLI matrix and a full
one-hour soak. Earlier R4 soak evidence belongs to its own runtime. See the
[R6 protocol, prefix and control-plane evidence](../benchmarks/correctness/current-portable-r6-protocol-prefix-controlplane-20260919.json).

R8 configuration now pins the repaired whole32a/FLA1d pair and the other R6
components, with the qualified q8192 and 128k prerequisites. The R8 archive
and its native regressions have not yet run. The standalone
[archive inventory verifier](PORTABLE_ARCHIVE_VERIFICATION.md) supports
checking the resulting ZIP against its accepted manifest. Windows continues
to reject visual tool results explicitly; the Linux `.10` protocol repair
does not supply a Windows visual inference qualification.

See [current performance and component identities](PERFORMANCE.md),
[the Linux `.10` source comparison and protocol checks](SIBLING_REVIEW.md), and
[the server's original HTTP record](../benchmarks/correctness/tool-text-parts-bounded-native-http-20260918.json).
Earlier context results remain attached to their exact configurations; none
lowers the required context/performance targets. The historical audit below
is not approval to publish the current work.

## Historical publication decision

Version 1.0.1 remains suitable for public source publication for its declared,
model-specific Windows target. The runtime source, AOT inventory, build entry
points, lifecycle/API tests, licenses, and redacted real-model evidence are
present in this repository. Model weights and vendor runtimes remain external.

The candidate described by the August audit was source-publication clean but
not yet eligible for promotion to `main` or a release. Its nine q8192
selected-MoE AOT objects reproduced byte-for-byte from neutral build roots
after deterministic debug stripping; its32-shape component smoke, real
layer-3 GB10 comparison, Windows artifact transport check, and local
C/Rust/Python/hygiene gates passed.
A fresh native Windows build and correctness-attached real-model product run on
`baiying`, from that candidate commit, was still required. Component and
transport evidence do not substitute for that product gate. The candidate
record is
`benchmarks/performance/prefill-diagnostic-public-complete-aot-r1187-r1190.json`.

## Historical source and release evidence

These rows summarize historical audits from multiple source revisions.
The exact downloadable v1.0.1 archive was published on August 15 from
`2bf04571`; its acceptance records 19658.8225 ms load and q8192 TTFT of
3840.4419–3877.0237 ms. The 1.036619x neighbor result belongs to the later
August 16 source `09bd96fd`, not that archive. See the
[source and archive attribution](PERFORMANCE.md#q8192-neighbor-continuity-gate).
Neither historical record qualifies the current candidate.

| Area | Result | Historical basis |
|---|---|---|
| Native source and ABI | Pass | C/C++/HIP core plus Rust server/CLI |
| Windows build | Pass | clean MSVC/Rust/HIP build scripts and manifests |
| Local regression | Pass | C smoke, Rust tests/clippy, Python tests, hygiene |
| Lifecycle | Pass | detached start, independent status, graceful stop/PID exit |
| OpenAI surface | Pass | JSON/SSE completion/chat, tools and continuation |
| Request pressure | Pass | bounded FIFO concurrency, queue metrics, overload errors |
| Context behavior | Pass | continuous inputs, maximum boundary, explicit overflow rejection |
| Prefix cache | Pass | seed, COW hit, resident hit, A-B-A isolation |
| q8192 product target | Pass | retained source row: 3,852.909 ms and 2,126.186 tok/s |
| Published v1.0.1 neighbor continuity | Historical limits pass | 18/18 GB10 matches; worst ratio 1.426480x, residual 1640.221 ms; historical limits 2x / 5000 ms |
| Later source neighbor continuity | Pass | `09bd96fd`: 18/18 GB10 matches; worst ratio 1.036619x, residual 142.288 ms; limits 1.10x / 500 ms |
| Published v1.0.1 load bound | Pass | 19,658.823 ms, below 30 seconds |
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
