# Actual adaptive linear OUT replay

The [queued follow-up](OUT_VARIANCE_QUEUE_EXPERIMENT.md) separates replay from
certificate state, preserves the same numerical policy and measures all 30
linear layers. Both variants remain default-off.

Default-off `QRT_QWEN36_Q8192_LINEAR_OUT_VARIANCE_REPLAY=1` implements the
linear shape identified by the [variance-budget observer](OUT_VARIANCE_BUDGET_AUDIT.md).
It admits only 2048 rows, 8192 tokens, K4096, radius 512 and PPB 1000, with the
original corrected BF16 OUT and unrounded vLLM residual/RMSNorm consumer.
Other shapes use the existing route. Conflicting producer, filter and audit
options are rejected. No dependency or model arithmetic is replaced.

The owner produces the original algorithm0 matrix, prepares the unchanged
36-byte K16 rows and replays residual-ambiguous candidates first. Each token's
CTA then refines the remaining squared-magnitude interval, prioritizing large
variance contributions independently of reference or control outputs.
Certification uses the original FMA reduction and enumerates every FP32
variance through the original rsqrt function when the span is at most 4096
ULPs. Every residual and normalized BF16 endpoint must be fixed. The final
of at most 18 rounds replays all remaining original candidates. An observed
interval violation disables early certification and forces that full replay.
The unchanged residual/RMSNorm kernel executes afterward. One descriptor-owned
94,674,944-byte workspace is drained before release, including failed launches.

This certificate transports the existing empirical projection envelope. It
does not prove that arbitrary hardware matrix errors fall within that envelope.
Observed violations force fallback; an unobserved violation is not ruled out
by the implementation. Real GB10 output IDs, first-logit tolerance and actual
callbacks remain the correctness gate.

Source `8d31cff` passes full local checks: 54 Rust tests, clippy, 477 Python tests
(two skipped), C/q16 ABI and public hygiene. The actual asynchronous host owner
passes ASAN/UBSAN with 21 success cases, 252 injected failures and four invalid
report cases. Windows compiles the numerical executable in 116820.680 ms.
All 63 GPU cases pass 903168 independent CPU original dots followed by complete
original residual and normalized F32/BF16 comparisons. Original inputs,
prepared rows, statistics and all consumer redzones remain intact. Generated
cases cover widths 16/32/272, counts 1/3/17, cancellation, subnormals, large
products, nonfinite norms and deliberately invalid envelopes. The invalid
envelope mode observes 128810 failures and fully replays 129024 selected cells.
These generated tests do not qualify matrix-envelope accuracy on arbitrary
inputs or establish inference performance.

The first same-DLL pair preserves every GB10 output and callback ID and first
logit 10.375. However, the new owner runs only 28 of 30 intended linear layers:
the entry unnecessarily requires pointwise fusion, which is disabled in
early layers 0/1. The dispatcher correctly rejects that activation mismatch.
TTFT is 27835.3155/27682.8851 ms, but this incomplete integration establishes
no retained performance. Its 28 calls skip 27880215 of 52226048 original
candidates, with 229376 certified token rows and no observed boundary failures.

Source `5c085c9` removes only that unrelated entry requirement. Early layers
already materialize the corrected BF16 OUT and invoke the exact same
unrounded residual/RMSNorm kernel. The adaptive kernel, allocation owner,
generated tests and original consumer arithmetic remain byte-identical.
C ABI, hygiene and the sanitized owner check pass again. The corrected native
build completes in 93440.136 ms. Its DLL is 13348352 bytes, SHA256
`df903181449db24befd16f3738b33d1c2d52733accef75d6e4b8996f1f0f8922`.
Both builds declare 119 VGPRs, 12328 LDS bytes, zero private bytes and wave32
for the adaptive kernel; these declarations alone do not establish occupancy.

The corrected same-DLL pair activates all 30 intended linear layers and
preserves every original GB10 output ID, actual callback and first logit 10.375.

| Adaptive replay | Load ms | TTFT ms | TPOT ms | Active linear layers |
| --- | ---: | ---: | ---: | ---: |
| Off | 21481.2264 | 27777.4475 | 100.472139 | 0 |
| On | 21307.8358 | 27678.0636 | 100.946673 | 30 |

The actual counters exactly reproduce the observer's aggregate linear result:
58,489,767 original candidates; 13,455,017 first-pass and 15,843,176 additional
replays; 29,191,574 skipped, or 49.9088567%. All 245,760 token rows certify,
with 225 complete-replay fallback rows and zero observed interval failures.
There are 1,712,887 rounds, averaging 6.969755 per row. Completed owner clocks
total 2,395.954 ms and include production, preparation, replay, conversion,
completion and ownership. Other 130 dense corrections and ten coarse FA OUT
per-call counts remain identical to the control. Only the new option differs
between the two environments; all external runtime artifacts remain fixed.

Keep the option default-off and outside the retained stack. The single
99.3839 ms TTFT difference does not establish a repeatable speedup, and TPOT
increases by 0.474534 ms. Do not infer performance from the 49.91% reduction
in candidate dots. The next experiment separates long original K16 replay
from the per-token certificate state to measure the cost of the fused
execution structure. The 10,000 ms TTFT gate, 4,187.415605 ms retained target
and 30,000 ms load limit remain unchanged. No prefix, long-context, package
or release acceptance follows.

Evidence:

- [Initial source, local checks, native build and numerical cases](../benchmarks/correctness/out-variance-replay-native-20260917.json), 75504 bytes, SHA256 `66f245398f0ae3c20c23f277263b30bf3dddd35d86a8c6cc881c37f2faeebc65`.
- [Initial 28-layer product pair and rejected activation scope](../benchmarks/correctness/out-variance-replay-product-20260917.json), 370836 bytes, SHA256 `0cd924333d75e9385d258b6f23f7e2580a3b401e7f2b7ec9effc33070cc92bc9`.

- [Corrected full activation, native build and same-DLL product pair](../benchmarks/correctness/out-variance-replay-product-r2-20260917.json), 390119 bytes, SHA256 `85ef9e62143651119854795ad83539cffb62ae67e4f9d00195583d48d7ccc0d9`.
