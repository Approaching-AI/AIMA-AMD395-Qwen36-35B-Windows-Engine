# Shared inputs across exact QK scores

Source `cbe357d7f1e82fc3ce592251a6f45979c388cd53` assigns each thread
two or four independent QK scores, amortizing cooperative operand staging
and reusing a decoded query operand across two key columns. The best 2x2
schedule reduces complete captured q8192 QK from353.4056 to300.9476ms.
It remains an isolated component with no provider or package change.

All cells retain the original BF16 products, exponent maxima, unsigned
modulo sums, normalization and ascending K16 carry order. Each unsupported
cell is independently marked; the existing complete original-dot scan
replaces it. Both retained inline fallback and unchanged deferred fallback
are controls. Every arm uses256 threads and128-feature shared windows.

## Numerical and memory checks

All120 generated configurations pass164241040 raw score comparisons,
representing32848208 distinct score slots and1536 distinct independent CPU
dots. Cases cover partial31/32/33-row tiles, causal masks, signed zero,
unsupported operands, late fallback, overflowing endpoints and an invalid
operand in only the second cell owned by a thread. Original inputs, complete
CPU-derived prepared encodings, guards and unused tails remain unchanged.

Each captured arm checks every raw score after its warmup and each of three
rotated timing samples. Unique observed score slots are418496528 atq7169
and545259520 atq8192, including masked slots. There are228/256 independent
CPU dots. q8192 repeats the first1023 captured Q/K rows; this supplies no
new model prompt, GB10 output token or attention-context result.

## Complete component timing

Completed host clocks include the score kernel and full fallback scan.
Allocation, resets, references and observation are outside the clock.

| Schedule | q7169 ms | q8192 ms | VGPRs | Shared bytes |
| --- | ---: | ---: | ---: | ---: |
| Retained inline fallback | 268.4733 | 353.4056 | 130 | 16384 |
| Original deferred fallback | 249.6907 | 327.1309 | 81 | 16384 |
| One query, two keys | 239.4427 | 316.0277 | 131 | 24576 |
| Two queries, one key | 279.0684 | 368.0736 | 63 | 24576 |
| Two queries, two keys | 233.8882 | 300.9476 | 117 | 32768 |

All arms additionally incur the same one-time preparation8.0125/10.0689ms
forq7169/q8192. Static compiler declarations show wave32 and zero private
bytes throughout; they do not measure occupancy or explain timing causes.
Three samples from one process per shape do not establish long-term stability.

C smoke and repository hygiene pass. Unchanged runtime/Rust/Python sources
reuse the full suite recorded with register PV integration. Four bounded
baiying build, safety and capture guards pass, using
`run-native-microtile-exact-qk-r1.ps1`. Captures reference the real model at
`D:\models\Qwen3.6-35B-A3B`; the model itself is not loaded by this fixture.

Advance the2x2 schedule to a combined default-off attention experiment with
the previously validated exact native EXP correction. Complete attention
and the real GB10 continuation must qualify any resulting product decision.
The current27328.8799ms experimental TTFT,10000ms gate, retained4187.415605ms
target and release status remain unchanged.

[Structured evidence](../benchmarks/correctness/microtile-exact-qk-native-components-20260917.json):
170818bytes, SHA256
`6705943de4958b60a9c995c12cb4319614a3c4ae8f130212ecf5ae35eee7ad53`.
