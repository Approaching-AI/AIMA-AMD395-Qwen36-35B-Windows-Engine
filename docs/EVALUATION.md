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

The standalone `fla-output-capture-replay` diagnostic now isolates the last
GDN stage using fingerprinted reference intermediate inputs. Native q64 has
one mismatch in 262144 cells; the full real q7169 layer-0 surface has 2205
in 29364224 cells (relative L2 2.3199896e-5), much less than the full-chain
input replay. This localizes a major upstream contribution; it is not a
runtime improvement, a zero-error result, or product acceptance. The harness
is not linked into the model engine and retains bounded allocation/dispatch.
Its initial missing auxiliary-pointer ABI bug is repaired; both corrected
runs exit normally with numerical-difference status and healthy cleanup.
Compiler metadata now distinguishes the source signature from the actual
launch ABI, including Triton's two trailing scratch pointers and verified
offsets/size. No driver or system setting was changed for this repair.

Standalone upstream replay now isolates solve, W/U and state recurrence with
the saved real q7169 input. Six guarded native runs (64-token parent-capture
views, then full length) exit normally with numerical-difference status,
passing host checks and at most 2.719 ms per dispatch. With correct inputs,
inverse/W/U differ on 729/1577/1093 cells, but the state kernel still has
2917457 V-new and 4636318 chunk-state differences. It is a major independent
source of drift, not a passing replacement. Complete compiled argument
layouts, bounded allocations and segmented/tail addresses are checked; a
CPU fake HIP test double with sanitizers validates caller safety only.

A CPU first-update replay matches all 524288 BF16 state cells with continuous
Blackwell K64, and continuous K128 projection matches 114704 sampled V-new
cells when supplied reference chunk state. However, all sampled full-length
carried-state variants still fail without reference-checkpoint injection.
The probe explicitly separates this negative trajectory result from the
input-isolated projection control. Host exponent arithmetic is not an SM121
SFU emulation; state-update/raw-F32 and gating attribution remain open. No
runtime arithmetic, default profile or real-model release gate is changed
based on these CPU controls.

Offline CUDA state-IR audits at BV32 and BV64 show continuous K128
projection, a separate zero-seeded K64 update and F32 fused decay/addition.
They do not verify the live reference worker's autotune choice. An opt-in
IEEE state build is implemented and safely replayed, but not retained:
first-update BF16 differences fall from 185 to 14, while full-length V-new
and chunk-state differences rise to 3235232 and 5139527. Default state math
is unchanged. All nineteen build artifacts and the eleven-slot launch ABI
are checked; both native replays exit normally with passing host checks.

CPU-only exponent controls infer unique values from independent captured
F32 products, rejecting ambiguous, conflicting and underflowed constraints.
They cover only part of the inputs and still fail the carried trajectory;
these reference-derived values are never a production SFU implementation.
Per-row first-boundary traces identify an early sampled cancellation without
injecting reference state. Final local checks pass 184 Python and 44 Rust
tests, clippy, ABI smoke, transaction tests and hygiene. No model, retained
performance, package or release acceptance is added by these controls.

An explicit, default-off `QRT_FLA_GDN_STATE_BLACKWELL=1` control now uses
native compiler-launched K128 projection and K64 update kernels. Separate
64-token calls, disjoint carried-state buffers, checked tail ownership and
post-dispatch admission preserve the bounded diagnostic route. The first
native update matches all 524288 raw F32 cells of its CPU control. On saved
real input, isolated V-new/H difference counts fall to 2024619/3156778,
but full-chain state relative L2 worsens to 0.0009255941. These are not
passing full-length component results.

A new real-model q7169 first-token run covers all 30 eligible linear layers
with this state control and the optional KKT accumulator. It completes
normally with healthy cleanup but still emits 220/9.3125 instead of the
frozen authority's 82/9.25. Load is 20234.0053 ms and TTFT 12972.0762 ms;
the route is rejected as a product or retained-performance improvement.
Default runtime arithmetic and correction limits remain unchanged. Local
checks pass 188 Python and 44 Rust tests, clippy, ABI/transaction tests and
public hygiene. No continuation, release or issue-closure gate is promoted.

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
