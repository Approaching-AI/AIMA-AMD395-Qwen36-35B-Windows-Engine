# Saved partial prefixes

This implementation is experimental and disabled unless
QRT_QWEN36_PREFIX_CHECKPOINTS=1. Native q7169 cold capture preserves all 32 GB10
outputs and first-token logit, and all three model checkpoints are complete.
With QRT_QWEN36_PREFIX_FLA_SINGLE_SUFFIX=1, a saved 7168 prefix and an
independently computed prefix both return the GB10 first token 82/logit9.25
and all 32 continuation tokens at both repeated hits. All 40 F32 layer
carriers, final BF16 norm and nine operator boundaries match the GB10 cold
prefill row at position 7168. Rollback and unrelated-prefix rejection pass.

The earlier token220/logit9.3125 failure used decode recurrence for the last
prefill input. Seeded FLA preserves the prefill BF16 chunk boundaries and
unrounded FP32 initial state; generated-token decode retains its original
recurrence. This qualifies the tested single-input case only. Divergent
requests, multi-token suffixes, the wider cold matrix and packaged-server
execution remain open. See
`benchmarks/correctness/prefix-fla-single-input-20260912.json`; the earlier
failure remains in `benchmarks/correctness/model-prefix-checkpoints-20260912.json`.

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
whole-model results have their separate record above.
See `benchmarks/correctness/fla-seeded-fp32-20260912.json`.

A separate opt-in probe, QRT_QWEN36_PREFIX_FLA_SINGLE_SUFFIX=1, uses seeded
FLA for one actual suffix input at a 64-token boundary. The scope must visit
all 30 recurrent layers and ends before generated-token decode. Key-major
FP32 state is transposed as bits into a private 2 MiB FLA workspace and back.
The original zero-seed capture hooks are rejected on this entry point.
Both state layouts pass native q65 and real q7169 operator parity at three
independently computed prefixes. See `benchmarks/correctness/fla-seeded-key-major-20260912.json`.
The model claim remains limited to the q7169 single-input proof above. Warm
callback times exclude owner/seed prefill and cannot qualify cold TTFT.

The single-input arithmetic probe is bounded to a prefix below 8192 tokens.
Native source 8b94be9 preserves the q7169 saved-prefix result and the previously
qualified cold q8193 scheduler continuation. Both 32-token/logit gates pass;
all 93 q8193 diagnostic files equal the prior qualified native configuration.
The q8193 generated-input row has internal differences from the GB10 r3 trace;
these diagnostics do not override its passing token/logit boundary. See
`benchmarks/correctness/prefix-fla-bucket-20260912.json`.
The reference capture tool also supports actual-token branches with read-only
runtime boundaries, while retaining both immutable controls and validating
each selected row against the complete actual input/output history.

A new reference capture initially failed the unchanged q7169 control. A
same-input GB10 replay isolates its first-layer drift to the FLA triangular
inverse autotune choice: the historical 2-warp/2-stage configuration reproduces
all original output and FP32 state bits; selecting 4 warps/5 stages alone
reproduces the rejected run. Python kernel sources are identical. The original
reference configuration is restored through its recorded cache choices.
Both complete immutable controls pass again: q7169 token82/logit9.25 and
q8192 token144/logit10.375, with all 32 outputs each. Four actual branches
at prefix6208, 6656 and 7168 now have GB10 captures, including a five-input
suffix. See `contracts/gb10_partial_prefix_actual_tokens_20260912_oracle.json`.
These are reference captures; each native Windows route still needs its own
qualification. A timeout in post-run metadata collection is preserved, and
all 3000 binary capture files were recovered and verified without rerunning
the model. The lost controller memory minimum is explicitly unavailable.
See `benchmarks/correctness/gb10-fla-autotune-drift-20260912.json`.
