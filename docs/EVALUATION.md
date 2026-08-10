# Correctness and evaluation

## Correctness authority

The numerical authority is a separate BF16 Qwen3.6-35B-A3B service. Product
acceptance binds real prompt token IDs, the first generated token, and the
first-token logit within 0.125. Decode and prefix continuation are compared
token-for-token. Engine self-hashes are diagnostic only.

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
