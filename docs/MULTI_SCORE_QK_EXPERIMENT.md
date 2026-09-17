# Shared inputs across exact QK scores

Source `cbe357d7f1e82fc3ce592251a6f45979c388cd53` assigns each thread
two or four independent QK scores, amortizing cooperative operand staging
and reusing a decoded query operand across two key columns. The best 2x2
schedule reduces complete captured q8192 QK from353.4056 to300.9476ms.
That component is now integrated behind the default-off combined option
described below; the package remains unchanged.

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

## Combined complete attention comparison

Source `c3a7c3b0fc6a94222697fc7552999451b5942788` compares the two
replacements independently and together. All four arms use the qualified
register PV rescaling and original compacted canonical replay.

| Arm | q7169 attention ms | q8192 attention ms | q8192 including preparation ms |
| --- | ---: | ---: | ---: |
| Retained prepared QK and source EXP | 545.0784 | 747.2801 | 751.0101 |
| 2x2 QK only | 500.4656 | 694.0511 | 697.7811 |
| Exact native EXP only | 513.4106 | 705.9507 | 723.5104 |
| Both | 472.0864 | 654.7009 | 672.2606 |

One warmup and three samples rotate across128-query slabs. Completed host
clocks include QK and its full fallback scan, probability, native PV,
collection and all exact replay. Common preparation includes Q/K encoding
and V transpose. Candidate preparation also charges one device table build
and one exhaustive328728576-input validation:8.1665+5.6632ms atq8192.
The derived table occupies82182144bytes in addition to the original source.
Nested probability events have no invalid intervals and are not added to
the complete clock.

Every attempted raw score, probability, scale, output, accumulator,
denominator and error surface matches the independent original control.
All29364224 available GB10 context cells and original candidate counts
2198673/3127598 match atq7169/q8192. Full input/table immutability, guards,
unused tails and228/256 CPU QK samples pass. The repeated-row q8192 scope
and absence of a model token loop remain unchanged.

Advance both replacements together to a default-off provider option with
owned, validated derived storage. Real q8192/out512 GB10 qualification is
still required before any runtime retention. C smoke, hygiene and all three
bounded build/capture guards pass; unchanged core sources reuse the recent
full suite. [Combined evidence](../benchmarks/correctness/combined-exact-attention-native-components-20260917.json):
108869bytes, SHA256
`5ad87a4be1ad93e61157a3d1730e83dc53f52397f1866b52a152f2399cc84af7`.

## Provider integration and real-model qualification

Source `cf0f889bf82bf8e9122a263f79360739d841c45a` adds
`QRT_CK_SM121_EXACT_ATTENTION_PIPELINE=1`. The eligible owner is a cold call
starting at zero with 2–8192 queries, prepared decoded QK, compact PV mode 1,
register PV rescaling and the compiled interpolated EXP backend. Exponent
masking is incompatible. Invalid option strings and conflicting eligible
configurations fail before allocation. Decode, suffix and longer calls retain
their prior dispatch. The exported ABI and source EXP artifact are unchanged.

The provider owns an additional 82182144-byte table under its existing lock.
It builds the two-bit codes on the execution device, separately checks all
328728576 inputs, and publishes the owner only after completion and a zero
mismatch count. Subsequent calls reuse it; release frees it before the source
table. Failed allocation, submission, verification and readback drain work
and discard the unpublished buffer. Local checks pass 54 Rust tests, clippy,
488 Python tests with two existing skips, C/q16 ABI and public hygiene.

Four bounded native build/replay actions pass on baiying. The actual production
callbacks replay the original q7169 capture with 29364224 GB10 context matches
in both modes, identical output/accumulator/denominator files, unchanged
candidate counts, guards and immutable operands/tables. Completed query host
clocks are 551.9448/477.5045 ms OFF/ON. The ON EXP initializer separately takes
11.5977 ms. Legacy device-event preparation totals do not consistently cover
that synchronized initializer and are not used as complete-owner timings.
[Native integration evidence](../benchmarks/correctness/exact-attention-provider-native-20260917.json):
177355 bytes, SHA256
`b8a61c1ba302c407479852157b2a27a1a3952b55829b13fb879c9cab821e4b84`.

`run-exact-attention-product-r1.ps1` then runs the real model at
`D:\models\Qwen3.6-35B-A3B` in four fresh baiying processes. Both modes keep
register PV enabled; only the combined option differs within the same DLL.
All other whole/MoE/FLA/CLI artifacts and numerical options remain pinned.

| Order | Combined option | Load ms | q8192 TTFT ms | TPOT ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0 | 21288.8146 | 27266.7663 | 101.105092 |
| 2 | 1 | 21286.2752 | 26385.5584 | 100.847081 |
| 3 | 1 | 21272.8293 | 26385.8867 | 100.548759 |
| 4 | 0 | 21279.4039 | 27259.2610 | 100.650795 |

Each run matches the original 8192 prompt IDs, all 512 GB10 output IDs, all
512 actual callbacks and first logit 10.375 with tolerance 0.125. Each ON run
has ten eligible prefill activations and one verified EXP initialization,
whose cost is included in first-callback TTFT. Decode has no activation.
All 160 dense and ten coarse correction counts match as diagnostics.

OFF/ON TTFT medians are 27263.01365/26385.72255 ms, a reduction of 877.2911 ms
or 3.2179%. Both ON samples are below both OFF samples. Retain ON in the next
experimental q8192 control. Two observations per mode do not establish long
term stability; code/package defaults stay off. Prefix, long context,
packaged HTTP, the 10000 ms gate and retained 4187.415605 ms target remain open.

The CK DLL is
`D:/projects/AIMA-public-exact-attention-provider-20260917/build/exact-attention-provider/qrt_ck_fmha_sm121.dll`,
1780736 bytes, SHA256
`2636dbf98b7c91028f9eebeacc3beb1920e9921bbb37209713502f11c142c027`.
[Product evidence](../benchmarks/correctness/exact-attention-product-20260917.json):
772770 bytes, SHA256
`1a9d33a1c903efc86e8b0bfc51ec54c3f06fa01d58916347ea7ffd5721d84834`.
