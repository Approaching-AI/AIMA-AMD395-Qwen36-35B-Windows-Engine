# Benchmarks and correctness evidence

The evidence bundle is deliberately compact and reviewable:

- `performance/`: all retained Windows product rows and acceptance bounds;
- `correctness/`: external BF16 first-token and continuation bindings;
- `openai/`: API, lifecycle, queue, context, and prefix-cache acceptance; and
- `eval/`: MMLU-Pro aggregate acceptance plus 12,032 sanitized parity rows.

Evidence never includes model weights, evaluation question text, private
endpoints, credentials, or personal deployment paths. SHA256 values bind each
published artifact; diagnostic self-hashes are not correctness authority.

The current unreleased runtime has separate correctness and performance
status. `correctness/prefix32k-admission-product-20260913.json` records the
broader correction bound's original 32k and q8192 acceptance, including the
permanent original-row regression. `correctness/cold-prefill-chunks-20260913.json`
records the opt-in 8192-input cold chunks, native builds, cold17408 and q8192
controls, both 16k prefix transactions, callbacks and state restoration.
Neither record passes the retained performance targets or qualifies a release.
See [current measurements](../docs/PERFORMANCE.md).

`correctness/attention-integer-core-20260913.json` records a separate exact
integer attention experiment: independent host arithmetic checks, original
Q/K coverage, three native builds and seven complete operator replays. All
declared numerical comparisons pass, but the new schedules remain slower than
the control. These are component results; product dispatch keeps its existing
attention route.

`correctness/q8192-current-wall-profile-20260913.json` attaches all 512 GB10
outputs and callbacks to the current corrected q8192 profile. It preserves
negative GPU event values as invalid evidence, separates host walls from
overlapping GPU stages, and records the source-derived routed submission
count. Its instrumented 64.37-second callback does not replace the ordinary
62.48-second control or qualify retained performance.

`correctness/moe-wide-compaction-20260913.json` records a wider routed
candidate window with at most 1024 persistent replay blocks. The same-binary
q8192 pair passes all 512 outputs and callbacks in each run, with callback
TTFT 62231.9179 → 59348.223499 ms. A separate instrumented run also passes.
The larger window is retained for experiments; broader contexts, packaging
and the immutable performance targets remain open.

`correctness/moe-parallel-gate-20260913.json` records an independent M64/N64
matrix schedule. Its 19-shape native comparison is bitwise exact and its
q8192 run passes all 512 GB10 tokens and callbacks. The 134.8 ms observed
callback difference from the qualified wider-window configuration is too
small to retain as a performance improvement; the matrix option stays off.

`correctness/moe-absolute-selector-20260913.json` preserves a rejected
admission experiment. Removing the three fixed routed midpoint bands while
keeping the norm-scaled bound produces 384 incorrect continuation tokens out
of 512, despite the exact first token and logit. Its timing is ineligible for
performance acceptance; subsequent experiments restore all three radii to 512.

`correctness/dpp-exact-reductions-20260913.json` records native masked integer
reductions, adversarial BF16 dots, emitted GPU instructions and two complete
q8192 GB10 boundaries. Extending DPP transport from MoE to the whole, CK and FLA
providers gives an observed 57617.9907 ms callback, with unchanged admission
and arithmetic. This is an experimental baseline; retained performance and
release qualification remain open.

`correctness/attention-compact-pv-20260913.json` records globally compacted
attention PV correction, twenty native safety cases, three original q7169
operator comparisons and the complete q8192/out512 GB10 boundary. All external
BF16 cells, product tokens and callbacks pass. The observed callback is
56604.0518 ms, compared with 57617.9907 ms for the preceding DPP configuration.
This single-run comparison supports the next experiment, without qualifying
retained performance, broader contexts or a release.

`correctness/attention-parallel-probability-20260913.json` records 216 native
probability/scale comparisons and two complete original q7169 operator runs.
Eight-wave probability generation preserves every checked bit and external
BF16 endpoint. Its 5.39 ms observed operator difference is too small to retain;
no full-model trial selects the new mode.

`correctness/moe-staged-dot-20260913.json` records K64 operand staging ahead
of ordered routed correction, 73,782 independent CPU/native dot comparisons,
emitted GPU instructions and a complete q8192/out512 GB10 boundary. Actual
callback TTFT is 55453.937 ms; all tokens, callbacks and the first logit match.
The measured 1150.1148 ms reduction supports further experiments, while
retained performance and release acceptance remain open.

`correctness/canonical-normalize-20260913.json` records a common exact
normalization simplification across whole, CK, FLA and MoE. Independent wide
CPU arithmetic checks 4,194,304 magnitudes and native dots check 73,782 outputs.
All q8192/out512 tokens, callbacks and the first logit match GB10 at an actual
53818.8428 ms callback. The 1635.0942 ms observed reduction compares one run
each; no retained performance threshold or release is qualified.

The v1.0.1 q8192-neighbor and wide-length repair adds four bounded artifacts:

- `correctness/gb10-q8192-neighbor-continuation-reference-v1.0.1.json` is the
  external BF16 token-ID oracle for q8191/q8192/q8193, one- and two-token
  continuations, and three cold-prefix repetitions per shape; and
- `performance/q8192-neighbor-provider-smoke-amd395-v1.0.1.json` records the
  isolated CK-FMHA and fused-GDN q8191/q8192/q8193 provider checks on gfx1151.
- `performance/q8192-neighbor-product-gate-amd395-v1.0.1.json` records the
  native Windows real-model 18-case publication gate, exact token outputs,
  TTFT cohorts, continuity bounds, source hashes, and source commit.
- `performance/prefill-length-smoothness-amd395-v1.0.1.json` records the
  72-case q4096-through-q16384 cold sweep, all 24 TTFT cohorts, local boundary
  gates, the global throughput envelope, runtime artifact hashes, and GB10
  token-ID agreement.

The provider record is synthetic component evidence only. The product record
was generated by `scripts/verify_q8192_neighbor_continuity.py` against the
native Windows real-model service on `baiying`; all 18 token sequences matched
GB10, and every neighbor cohort passed both continuity bounds.
`scripts/verify_prefill_length_smoothness.py` independently compared every
request in the wider sweep to GB10; all 72 token IDs matched and all eight
length-boundary triplets passed.
