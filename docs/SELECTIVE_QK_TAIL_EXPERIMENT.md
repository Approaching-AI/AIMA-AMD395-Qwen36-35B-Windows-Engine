# Selective QK prefix with an original-accumulator suffix

`QRT_CK_SM121_SELECTIVE_QK_EXACT_TAIL` is an isolated numerical experiment.
It defaults to `0`, which preserves the complete original dispatch. A decimal
value from `1` through `8192` requests at least that many final queries through
the original QK/probability/PV path. The earlier queries use the existing
selective probability repair, including its approximate denominator. This is
not an exact attention algorithm or a proof of model-level equivalence.

Only cold calls starting at query zero with at most 8192 queries can use the
mixed route. The selective prefix is rounded down to a complete query batch;
the final partial batch remains original. A tail covering the entire call,
single-query decode, prefix continuation and long-context calls preserve
their original dispatch. Both paths run in order on the same stream and hold
the existing workspace lock until completion. Failures drain submitted work.
The selective scratch adds at most 136314884 bytes and is allocated only when
at least one query batch selects it. Prepared QK still encodes the complete
cold inputs before either path consumes them.

The legacy all-query selective flag cannot be combined with a nonzero tail.
An active mixed call also rejects completed-stage profiling and multi-slab
submission. Compact PV modes 1 and 3 are supported. Parsing, query ownership,
partial-tail coverage, allocation failures and failures on both sides of the
transition are exercised by the host tests using the actual provider code.

The earlier all-query route failed the gb10 continuation contract despite a
passing first token. A longer original suffix is only a new hypothesis. Each
real-model run must attach the original prompt IDs, all captured output IDs,
actual stream callbacks and first-token logit within 0.125. A failed output
invalidates performance retention. The option is absent from release profiles;
no numerical or release qualification is implied by compilation or host tests.
