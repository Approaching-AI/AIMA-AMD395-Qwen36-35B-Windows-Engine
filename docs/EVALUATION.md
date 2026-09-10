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

Other isolated stages still have known mismatches: inverse, W/U and output
arithmetic. The complete state repair provides a firmer boundary for their
replacement; it does not erase errors entering the recurrence. Full local
checks pass 238 Python tests (two platform/dependency skips), 45 Rust tests,
clippy, C ABI, seven q16 contracts and public hygiene. The skipped NumPy
packing tests pass in the existing reference container; these source checks
cannot replace real-token, prefix, HTTP/archive and retained-performance
qualification. The retained q8192 target and numerical tolerance remain fixed.

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
