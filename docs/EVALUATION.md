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

## Unreleased FLA diagnostics

The unreleased FLA replacement is a separate diagnostic route, not a change
to the published acceptance below. It now follows the reference worker's
Triton/FLA decomposition and bounds native dispatches to 1024-token segments.
The q64 stage comparison identified BF16 elementwise products truncating
before a same-dtype cast. Promoting operands before explicit nearest-even
rounding removes 1824 first-token product discrepancies and reduces output
relative L2 from 0.0158573 to 0.000409875. Remaining differences are reported
as failures of exact component parity, not inference success. Native q64,
q65 and q7169 synthetic runs exit safely; only q64 has this stage reference.
`scripts/compare_fla_gdn_capture.py` validates the saved reference manifest
and distinguishes component diagnostics from real-token product acceptance.
The q7169 token/continuation contract and release gates remain unchanged.
The subsequent real-model q7169 test safely exits but still produces token
220 instead of reference token 82, so this repair alone is not a passing
product route or retained performance result.

An isolated follow-up keeps BF16 input products but accumulates U in IEEE
F32. It removes all 50 q64 two-term midpoint discrepancies and reduces q64
output relative L2 to 0.000271804. A fingerprint-verified saved real q7169
layer-0 input replay has output/state relative L2 0.001072256/0.000771976.
Both component runs safely exit, but exact parity still fails, and this
variant has not passed a new real-model gate. The build-generated launcher
metadata tracks its changed shared-memory requirement (16384 bytes).

An opt-in continuous Blackwell K128 accumulator now matches all 131072
captured q64 pre-decay KKT cells byte-for-byte. CPU attribution also matches
28225 sampled cells spanning saved real q7169 layer-0 input. Native dispatch
is serialized in 64-token chunks with elapsed-time admission before the next
chunk; no high-cost correction ceiling or runtime default is relaxed.
Gate scaling and later stages remain different. The real component replay
has output/state relative L2 0.001061918/0.000791691, not a consistent state
improvement and not a passing model result. The comparison tool includes the
pre-decay surface when present, rather than silently omitting this boundary.

The W midpoint repair is subsequently Windows-verified: all 50 known
two-term discrepancies disappear and the other q64 surfaces stay unchanged.
On the saved real replay, output relative L2 is 0.001046221 but state relative
L2 worsens to 0.000912814. It is not promoted as an overall improvement or
real-model acceptance. A separate CPU-only U probe identifies continuous
Blackwell K64 as bit-exact for 262144 q64 cells and 114709 sampled real cells;
this does not bypass upstream inverse errors or authorize unbounded native
correction. The repaired WSL AOT launcher disables GPU visibility and retains
compiler/thread/time limits. No system configuration change is required.

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
