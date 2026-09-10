# Correctness and evaluation

## Correctness authority

The numerical authority is a separate BF16 Qwen3.6-35B-A3B service. Product
acceptance binds real prompt token IDs, the first generated token, and the
first-token logit within 0.125. Decode and prefix continuation are compared
token-for-token. Engine self-hashes are diagnostic only.

The unreleased arbitrary-length gate additionally freezes the failing q7169
fixture in `contracts/arbitrary_q7169_gb10_oracle.json`: prompt hashes, first
token 82, raw logit 9.25 at the unchanged 0.125 tolerance, and all 32 reference
tokens. This is captured GB10 authority, **not** Windows acceptance. The default
q8192 contract and its targets are unchanged. `scripts/capture_gb10_q7169_oracle.py`
is the byte-identical fixture adapter used for the September 9 capture; stage it
beside the two existing q8192 capture modules in the opt-in BF16 reference
container. It changes fixture constants only and does not alter model math.

The published OpenAI acceptance additionally submits deterministic raw-token
prompts across the continuous-length matrix and requires every returned text
and finish reason to match a frozen authority response. It separately tests
the exact maximum accepted input and context-limit rejection.

The production `engine/runtime.env` scopes BF16-window high-ID arbitration to
the two continuous maximum-context provider shapes that require it. The formal
MMLU run explicitly enabled the same policy globally, as recorded in the
summary (`max_ulps=2`, eligible count 2, count-three shape mask 12) and in
`benchmarks/eval/mmlu-pro-runtime-overrides-v1.0.0.env`. Keeping global
evaluation arbitration explicit prevents a score-calibration tie break from
changing ordinary OpenAI completions; production HTTP acceptance always runs
with the global override disabled.

## Unreleased FLA diagnostics (updated September 10)

The optional FLA route at `33e0492ee17013aa897eb19aa092c15cf56df2bf` now
matches the saved real q7169 layer-0 recurrence exactly on native Windows:
29,364,224 V-new BF16 values, 59,244,544 checkpoint BF16 values, and all
524,288 raw FP32 terminal-state values. Shorter q64/q384 views also pass.
These replays use fingerprinted GB10 intermediate inputs and load no model.
They establish a component repair, not an accepted inference route.

A new GB10 q384 capture preserves saved BF16 parity and same-run raw terminal
parity before recording 12,480 independent exponent instruction samples.
The CPU probe covers all 4096 head/value rows without feeding reference
checkpoints into its carried trajectory. Exact exponents, continuous
Blackwell K128/K64 accumulation and fused state update match every raw bit;
host exp2, split projection and unfused-update controls do not.

The optional `QRT_FLA_GDN_SM121_EXP2_TABLE` artifact exhaustively enumerates
all 2,139,095,041 nonpositive FP32 inputs through negative infinity. Its
builder accepts no model or prompt input. Lossless packing produces
183,174,448 bytes, SHA256
`f490940df2bd80421159a96424c3e922330b7ca120d5ae7b629a973b9183730b`.
All earlier independent samples also match the actual native lookup.
Windows validates the fixed SHA through system CNG before uploading it;
[dependency policy](dependency-policy.md) records the memory/packaging cost.
The published default profile is unchanged.

The r15 Windows build reuses unchanged, source-validated r14 AOT artifacts;
its FLA DLL SHA256 is
`f8339eadfca6a25379e9455eaa4b091948cccdb3887fac80108a0f4680038b0a`.
The compiler-owned state kernels retain explicit dispatch/shape bounds,
disjoint state ownership and scratch accounting. Full q7169 isolated replay
uses 514,663,728 bytes and 226 dispatches, maximum 2.060 ms. All native guards
complete with healthy cleanup. The launcher still validates compiled ABI,
including Triton's auxiliary scratch slots. These times are diagnostics.

The subsequent real-model q7169 test on `baiying`, using
`D:\models\Qwen3.6-35B-A3B` and command file
`prepare-fla-model-q7169-exact-exp2-r2.ps1`, **still fails**: output token 220,
logit 9.25, versus GB10 token 82 / logit 9.25. Prompt FNV matches
`c900c18703532d6a`. Engine load is 20122.1765 ms and TTFT 13267.5151 ms.
The FLA source is `33e0492`; the unchanged whole-provider source is
`c36e2674a5ade5544ef49a3756dbe99454b00991` and CLI source is `f544cbe`.
Run-record SHA256 is
`e8ca72baaf49064c47caee2c329442d3d5c02c2793d1b559cbaa8684bff13da1`.
This mixed-component diagnostic is not an all-component package build, a
passing continuation result, or accepted performance.

The extension at `0b10c4571c5403f972a4238d2a43720a80466c42` applies the
general exponent table to KKT gating and adds optional native Blackwell W/U
and output kernels. On the full captured q7169 inputs, W and U each match
all 29,364,224 BF16 values exactly, with the production U=V alias. Each CTA
reads its complete V column tile into shared memory before writing U. The
standalone output comparison improves to 177 differences / 29,364,224
values, relative L2 8.8967834e-8; it is still not bit-exact. The q64 component
controls pass, including the output kernel. Dispatches remain bounded to
64 tokens with per-call timing admission. Output reuses dead state residual
scratch; the standalone harness reads back one chunk at a time.

The integrated real q7169 replay still has 1,904,743 output differences
(relative L2 0.0009123102), which are much larger than the isolated output
kernel's residual. The following real-model test also **fails** with token
220 / logit 9.3125; the diagnostic top five are
`220:9.3125, 82:9.125, 64:9.0625, 144:8.9375, 83:8.8125`.
Load is 20004.9421 ms and TTFT 21870.2059 ms. The full route is not promoted.
The r16 FLA DLL is
`78aa91c9f64a611606ef1a3cee5f8fdc7013e24ad393f940e25113a256beeec0`;
the whole-provider and CLI retain their preceding source identities. Command
file `prepare-fla-model-q7169-blackwell-aux-r1.ps1` runs on the same Windows
host/model and oracle; run-record SHA256 is
`df4b3ee3bcaeda11b63ff090ab71e63615aa8b5ed681b868f769adde9d1238e0`.
All native build/replay/model guards complete with passing host checks.

The remaining investigation follows actual input propagation through
normalization, gating and inverse arithmetic; isolated kernel success does
not erase an incorrect predecessor. Full local checks pass 243 Python tests
(two platform/dependency skips), 45 Rust tests,
clippy, C ABI, seven q16 contracts and public hygiene. The skipped NumPy
packing tests pass in the existing reference container; these source checks
cannot replace real-token, prefix, HTTP/archive and retained-performance
qualification. The retained q8192 target and numerical tolerance remain fixed.

The later normalization correction at `3d82d6950b082f36b866cd248845e3973684a1ab`
reproduces every Q/K BF16 output in the complete frozen q7169 component case.
It uses the independently characterized reduction order and the general
reciprocal-root table described in `docs/dependency-policy.md`. Integrated
q64 output differences fall from 31 to one, but the complete model still
emits 220 / 9.3125 instead of the authority's 82 / 9.25. Native load is
20,023.1357 ms and prefill 21,866.8483 ms. This remains an unqualified opt-in
combination; the triangular inverse and output arithmetic remain under
investigation. Source checks pass 254 Python tests (two known skips), 45
Rust tests, clippy, C ABI, q16 contracts and hygiene. Native component and
model runs completed with passing host health/cleanup checks. No release
or retained-performance qualification follows from the component correction.

The next inverse implementation is controlled by
`QRT_FLA_GDN_INVERSE_BLACKWELL=1`. A fresh capture of the original reference
kernel reproduced the saved real q64 inverse exactly with two warps; four
and eight warps differed at eight and nine BF16 elements respectively. Its
emitted PTX identifies the diagonal FMA reduction order and the continued
FP32 accumulator across off-diagonal matrix products. Replaying the actual
new arithmetic header on the host matches all 14,682,112 BF16 inverse values
of the full q7169 capture (SHA-256
`7c6a9c8445ab3496c7468282aa7cd2d6cc47ed6849aa57762b84d56e57cecc62`).
The native kernel uses 33 KiB of shared memory per 64-token/head block and
adds no table or runtime dependency. Native Windows r18 at
`fdace5e1dfd7b5e40a78a59b8e2098b8fb4ea091` also matches every inverse element;
its maximum dispatch is 1.070 ms. The integrated real q64 prefix is exact
at every recorded stage and terminal state. Full q7169 terminal FP32 state
is now exact, while output retains precisely the standalone kernel's 177
differences. All host checks pass; full local checks pass 259 Python tests
(two known skips), 45 Rust tests, clippy, C ABI, q16 and hygiene.

The corresponding real-model command
`prepare-fla-model-q7169-blackwell-inverse-r1.ps1` on baiying, using the same
real model, prompt, oracle, CLI and whole provider, still **fails**:
220 / 9.3125 instead of 82 / 9.25. Load is 20,133.0916 ms and TTFT
21,836.5686 ms. Model run-record SHA-256 is
`f1c5de3a870f0673fc398e1f785e25364b1ee001e78a621e8ffcb1b898c8ef1e`;
the r18 FLA DLL is
`5e91a6450cd1a641cd0edcf65aa9ec4b2515711e9e798923e30918ea83080a21`.
This component correction does not qualify the combined model or performance.

The remaining output correction rounds the scaled prior before fusing the
local term, following the reference's evaluation order. A CPU attribution
with the general exponent table covers all 177 residual cells and 3,585
fixed-stride controls: all 3,762 match with this order. Reversing the FMA
retains all 177 differences; separate scaling retains 74. Windows r19 at
`992b04becc8057c2fc27477c4b40f6e748308184` now matches all 29,364,224 output
BF16 cells and 524,288 terminal FP32 state cells in the integrated q7169
capture. The q64 full-stage control remains exact. Its DLL SHA-256 is
`42c9e97fac635bd2caf773341d1d8f104cebb39dc73bd8b0df46d28b542d8640`.

The matching real-model run `prepare-fla-model-q7169-blackwell-output-r1.ps1`
still **fails** with 220 / 9.3125; token 82's logit is 9.125. Native load is
20,034.3477 ms and TTFT 21,868.1872 ms, with passing cleanup/host checks.
Run-record SHA-256 is
`a3cb14d4d190367bf6469e65de803b92404cc13e32a7b89bbefc33bf0285c2ac`.
The original layer-zero traces were suppressed by the required-marker and
deferred/compact log filters. Disabling those filters makes the existing
traces observable. The exact saved-input GDN case is component evidence; the
model remains unqualified.

`QRT_FLA_GDN_CAPTURE_FIRST_DIR` captures the first complete native GDN call's
actual raw inputs, gates, outputs and final state for 1..8192 tokens. It uses
one 1 MiB host buffer, completed stream reads, an exclusively created output
directory and a completion record after every operation succeeds. It never
supplies values to inference. Failed capture stops the call and cannot be
retried in the same prepared provider. The optional hook avoids reliance on
whole-provider trace branches. Host fault/tail tests and full local checks
pass 262 Python tests (two known skips), Rust, clippy, C ABI, q16 and hygiene.


The native r20 first-call control matches the existing q64 inputs and outputs.
The real model on baiying, `D:\models\Qwen3.6-35B-A3B`, at FLA commit
`831c1699c3c956ace00365bb42c2deb2045a7ea7` and whole-provider commit
`c36e2674a5ade5544ef49a3756dbe99454b00991` exposes differences before GDN:
7,014 Q, 8,852 K and 16,121 V BF16 cells differ from the frozen layer-zero
GB10 inputs. Raw gates differ as well. Command
`prepare-fla-model-q7169-first-call-r1.ps1` still emits **220 / 9.3125**
against **82 / 9.25**, with load 20,033.6432 ms and TTFT 22,046.4739 ms.
Run-record SHA-256 is
`2f5c27c099f630f0f7c0d20dbb2da54762f5df1143f643cabf0268850d6735fb`.

A BF16 projection-boundary control, using the same safe base profile and
`prepare-fla-model-q7169-bf16-endpoints-capture-r1.ps1`, reduces G relative
L2 to 1.20422e-7 (all BF16 endpoints match) and beta mismatches from 19,057
to 90. Q/K/V remain unchanged; GDN output mismatches fall from 7,223,511 to
4,308,499. The terminal input RMSNorm, Z, A and B projections match their
GB10 BF16 endpoints exactly, while QKV differs in 49 / 8,192 cells. The
model still emits **220 / 9.3125**; load is 20,076.6755 ms and diagnostic
TTFT 23,104.1936 ms. Run SHA-256 is
`ff049cd32bd2ec3f42808f3ccdd8d00b40d913dbb8ad52a78d70638a636ba752`.

CPU recomputation uses the verified terminal input and actual model QKV
weights (SHA-256 `b06edcc9973be74f862d072f89a195e2eecea067ee615144be8510656122401f`).
The production characterized accumulator at width 26 with continuous K2048
matches all 8,192 reference endpoints; split K1024 retains five differences.
This is a projection diagnostic, not whole-model acceptance. Its record SHA
is `274b785da42728dc7ac0f589e3a6fa82e0ff264305cf8a4ec971dca78189d1e3`.

The initial real-model WMMA/correction probe stops before correction because
915,149 candidates exceed the former whole-projection quota of 131,072,
although its densest original block has only 12 candidates. It emits no
model token. Run-record SHA-256 is
`3e46fa77950447f3438021dd8f258aa35306522f8fca6aaaff5a811a5549a931`.
All three model commands pass host and cleanup checks. No result qualifies
continuation, prefix reuse, product performance, or a release.

The correction launcher now streams 65,536-element collection windows into
constant 512 KiB index scratch, then executes only compacted candidates.
Indices remain absolute across token/row boundaries. Collection uses at most
256 blocks; exact-dot dispatches retain eight blocks, completed-stream
synchronization, the 100 ms dispatch deadline and 10 s aggregate deadline.
The 131,072 candidate and 64-per-source-block limits apply to each window.
Count-only mode leaves outputs unchanged. Host tests execute the production
launcher with mocked HIP to check cross-window tails, no repeated indices,
admission failure, asynchronous error cleanup and count-only behavior.
Native `qrt-projection-safety.exe --correction` adds an irregular K2048
case and a synthetic full q7169 geometry with more candidates than the former
global quota. Native and real-model evidence for this new launcher is pending.

## MMLU-Pro full evaluation

| Measure | Windows engine | BF16 authority |
|---|---:|---:|
| Dataset rows | 12,032 | 12,032 |
| Exact correct | 7,486 | 7,486 |
| Accuracy | 62.2174% | 62.2174% |
| Parsed predictions | 12,026 | 12,030 |
| Projection mismatches | 0 | 0 |
| Candidate/reference prediction agreement | 11,180 | — |

Identical aggregate accuracy does not mean every free-form prediction string
is identical. The publication includes each sanitized row so the 11,180
agreement count, parsing differences, correctness flags, and zero projection
mismatch can be independently recomputed.

The row file contains only stable question IDs, input hashes, answer key,
candidate/reference prediction and correctness, usage, and timing fields. It
does not contain dataset question text, answer choices, private endpoints,
machine paths, credentials, or raw service identifiers.

## Reproduction

Obtain MMLU-Pro separately, then run the included OpenAI evaluator against the
resident engine:

```powershell
python .\scripts\eval_mmlu_pro_openai.py `
  --base-url http://127.0.0.1:8000/v1 `
  --model qwen3.6-35b-a3b `
  --output .\candidate.jsonl
```

The harness fetches the pinned dataset revision from the Hugging Face dataset
server and caches it under the output directory. Use `--dataset-cache PATH` for
an existing cache. Candidate-only evaluation is the default; authority parity
requires `--side both --reference-url URL` and should never embed a private
reference endpoint in committed output.

Use the exact v1.0.0 command options recorded in
`benchmarks/eval/mmlu-pro-summary-v1.0.0.json`. Dataset/harness revisions,
prompt templates, few-shot policy, and answer extraction can materially alter
results, so comparisons must retain those fields.

For parity reproduction, pass every assignment from
`mmlu-pro-runtime-overrides-v1.0.0.env` as a `qrt start --set-env KEY=VALUE`
override after loading the production `runtime.env`; `--set-env` values take
precedence over the profile.

## Evidence files

- `benchmarks/eval/mmlu-pro-summary-v1.0.0.json`: formal aggregate acceptance.
- `benchmarks/eval/mmlu-pro-full-parity-v1.0.0.jsonl`: sanitized 12,032 rows.
- `benchmarks/correctness/`: GB10-bound first-token and continuation summaries.
- `benchmarks/openai/`: service/API/queue/prefix acceptance summaries.

All JSON/JSONL evidence is newline-terminated and accompanied by SHA256 in the
release manifest.
