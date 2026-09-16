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

The scope above applies to individual provider calls. A longer model request
can start with an eligible 8192-query cold chunk, which would select the mixed
route. Consequently this option does not imply an unchanged long-request
prefix state or qualification for any prefix reuse.

## Native product result, 2026-09-16

Source `483545c936d6ce57b1de354212d79dd55e86ba12` passes C/q16 ABI,
54 Rust and 470 Python tests (two skipped), clippy and public hygiene in
217.405328 seconds. Windows HIP compilation on baiying completes in
45976.742 ms with all host guards. Both fresh model processes use
`D:\models\Qwen3.6-35B-A3B`, the same new CK DLL and the retained
Dense1000/MoE512, expert-order and coarse OUT stack.

The disabled control matches all 512 gb10 output IDs and actual callbacks,
first token 144 and logit 10.375. Load is 21496.277601 ms, q8192 TTFT
27833.720601 ms and TPOT 102.159896 ms. The 4096-query original suffix
executes in all ten attention layers but fails at output index 1: expected
255, observed 244. Only 26 of 512 positions match. Its first logit 10.3125
has error 0.0625 and passes the first-token tolerance; that cannot excuse the
failed continuation. All callbacks and host guards complete, with native
exit 6. Its 27917.786599 ms TTFT is diagnostic only.

Keep the option default-off and outside release profiles. This suffix does
not recover the required continuation after changing earlier attention.
Evidence: `benchmarks/correctness/selective-qk-tail-product-20260916.json`,
419569 bytes, SHA256
`84cb0a83bd8b9cd105f9ffdcc8cc488ce5075f7e5eb171a3d4a21f8452454ca9`.
No performance target or gb10 threshold changes; release remains unqualified.
