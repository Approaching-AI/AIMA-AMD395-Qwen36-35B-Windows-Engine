# Packed probabilities inside consumed score rows

This isolated variant keeps the original FP32 score row stride and stores two
BF16 probabilities in each earlier consumed score word. It removes the same
separate probability owner as the [tagged storage candidate](INPLACE_PROBABILITY_STORAGE.md),
while exact PV replay reads two bytes per probability instead of four.

Each row belongs to one wave. Its existing maximum reduction consumes all
K32 scores before any probability store. All lanes then exchange adjacent
payloads, and each even lane writes one complete score word at `key / 2`
within that row. No word has two writers. Later K32 tiles and other rows remain
untouched. Odd-length rows retain space for the last pair; its unused high
payload is zero. Byte copies preserve representations without arithmetic on
a float carrier or access through an incompatible type.

The original, tagged and packed routes share the probability producer and
exact replay kernel. Softmax reductions, native PV, both error-bound policies,
candidate collection and original K16 arithmetic retain their operations.
Both in-place formats share pipeline orchestration. Template defaults retain
separate BF16 storage, and no provider option selects either candidate.
The active full256k model still uses its existing binaries.

At 128 queries and 264736 keys, both candidates require 2240692228 scratch
bytes instead of 3325050884, a reduction of 1084358656 bytes. These are layout
sizes; native memory residency and speed remain unmeasured.

ASan/UBSan checks under strict aliasing pass 262144 pair encodings, 35016000
consumed cells and 17508240 complete-word stores across interleaved rows,
causal tails and odd strides through the maximum context. All probability
payloads and untouched score words match. The actual probability loop also
runs with 32 host threads and emulated wave transport: 18 cases compare all
three formats and both sum orders, including scales, denominators and tails.
That test uses host EXP evaluation and covers storage lifetimes, not GPU
arithmetic. Shared orchestration passes 96 offset and failure cases. The nine
attention regressions and public hygiene pass.

The native fixture prepares three-format comparisons for 60 generated cases
through 264736 keys. Each candidate repeats 128 independent CPU QK dot checks
per case, giving 15360 checks across both candidates. All probabilities,
untouched score words, raw native/replayed surfaces, candidates, guards,
immutable inputs and five completed owner stages are checked. The original
1024-query GB10 capture supplies 4194304 context cells; the 8192-query extension
repeats input rows and provides component coverage only. Each format has one
warmup and three rotated completed-host samples, with every attempt validated.

Fixture signature checks initially rejected an omitted observer argument after
selection through a function pointer. The corrected fixture supplies it
explicitly and passes; the failed attempt remains recorded. Windows HIP
compilation, native correctness, model integration and release qualification
remain pending.

The [preparation evidence](../benchmarks/correctness/packed-probability-storage-preparation-20260920.json)
pins source `6f54c7bfed4e2ca9ce471ae99a8632487b57426f`, all 56 compilation
inputs and the guard script, passing host checks and the original fixture
failure. PowerShell on baiying accepts the exact command at
2026-09-19T17:02:59Z with zero errors and no script execution. Native limits
are 240/1500/600/900 seconds for build/generated cases/original1024/extended8192.
Both dispatch sides require final cleanup of the current original full256k
run and verify its command identity; model success is not required to begin
component repair. All four native actions remain unrun.
