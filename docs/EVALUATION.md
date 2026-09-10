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
adds no table or runtime dependency. Host parity is component evidence;
Windows GPU and real-model qualification remain required.

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
