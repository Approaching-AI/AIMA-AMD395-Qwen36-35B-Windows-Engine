# Saved partial prefixes

This implementation is experimental and disabled unless
QRT_QWEN36_PREFIX_CHECKPOINTS=1. Native q7169 cold capture preserves all 32 GB10
outputs and first-token logit, and all three model checkpoints are complete.
Partial continuation is **not qualified**: the saved 7168 prefix and an
independently computed prefix both return 220/9.3125 instead of the GB10 cold
prompt's first token 82/9.25. All 186 captured operator, carrier and KV files
in those continuation routes match bit-for-bit; repeated rollback and the
unrelated-prefix guard pass. The next arithmetic investigation is seeded FLA
prefill versus the current decode recurrence. This is a hypothesis, not a
completed repair. See benchmarks/correctness/model-prefix-checkpoints-20260912.json.

The separate recurrent producer's native evidence is
benchmarks/correctness/fla-fp32-checkpoints-20260912.json. That component
record does not qualify the model checkpoint or any release.

During a resident prefill of 65–8192 tokens, the provider selects up to three
64-token boundaries. It prefers message boundaries and uses bounded positions
near the end for raw token input. Selection uses the caller's actual tokens.
The optional FLA producer retains FP32 recurrent state at those positions.
Each of the 30 linear layers also retains its four convolution projection
rows in the decoder's existing format and absolute-position ring layout.
The final norm retains each checkpoint's terminal hidden row.

Publication requires complete linear state, convolution history, hidden and
all ten owner KV layers. The immutable store binds the owner's actual tokens,
engine pointer, generation and prompt digest. Missing producer capability or
incomplete capture produces a cache miss. Older providers keep their existing
exact-prefix path.

The qrt_engine_prefix_checkpoint_match_v1 API returns the deepest complete
boundary matching actual input tokens, with a suffix of at most 1024 tokens
and the existing decode-tail capacity. Zero means a miss. The server tries
this query before its existing seed logic. A saved hit performs no cold seed.

Restoration clones the saved linear state and mutable KV tail. Separate KV
storage borrows only the owner's immutable prefix. Contiguous KV storage copies
the shortened K and V from their original pointers into a private layout.
The branch always rolls back to the original owner; it cannot commit over it.
Cancellation, stream/clone failures and owner replacement remain subject to
the existing request serialization and cleanup contracts.

For native qualification, qrt-product run accepts --checkpoint-owner FILE
together with --prefix-tokens N. It first runs the complete owner prompt,
reports its first token/logit and time, requires a query hit at N, and then
executes the usual repeated prefix/stream/negative-guard checks. The owner must
be longer than N and match the requested prefix token-for-token. Owner prefill
time is reported separately and is never a cold TTFT improvement.

The host tests execute the actual query, publication and shadow transaction
code with guarded host allocations, including each allocation failure, both
KV layouts, missing layers/hidden, wrong generation/owner, unrelated tokens
and rollback. They do not establish GPU arithmetic or GB10 correctness.

The optional row-major FP32 seeded FLA entry point now reproduces the original
full operator execution exactly from independently computed prefixes. Native
q65 at prefix 64 and real GB10 q7169 at prefixes 64, 1024, and 7168 have zero
suffix output or final-state bit mismatches. This is operator evidence only;
whole-model suffix integration and key-major state conversion remain pending.
See `benchmarks/correctness/fla-seeded-fp32-20260912.json`.

A separate opt-in probe, QRT_QWEN36_PREFIX_FLA_SINGLE_SUFFIX=1, uses seeded
FLA for one actual suffix input at a 64-token boundary. The scope must visit
all 30 recurrent layers and ends before generated-token decode. Key-major
FP32 state is transposed as bits into a private 2 MiB FLA workspace and back.
The original zero-seed capture hooks are rejected on this entry point.
This probe does not establish multi-token suffix or whole-model correctness;
native operator and GB10 model verification are pending.
