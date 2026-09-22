# Packed gate reduction diagnosis

The current candidate uses 32 logical FMA chains for K2048/N32 gates. Each
physical lane accumulates two independent chains, combines logical lanes
i/i+16, then uses the existing 16-lane shuffle. Launch geometry and carriers
are preserved. This replaces the earlier double-precision midpoint patch.
The actual edited helper matches all 31,744 BF16/widened-carrier comparisons
across the two original 124-step histories. Three captured GB10 regressions
are retained as hash-bound fixtures and pass ASan/UBSan. Native source
`c90feccdfc01661b582c5a1db101f2660dc8642c` compiles on baiying and passes all
865 original component cases, 6,664 configurations and 7,672,332 compared
elements. This includes both complete gate histories, all three known failed
cells, QKV controls and the preceding broad projection/shared-activation
suite. Guards, immutable inputs and host cleanup pass.
[Native component evidence](../benchmarks/correctness/packed-gate32-native-20260923.json).

The whole runtime, CLI and prefix probe compile on baiying in120665.79ms.
All173 source inputs and unchanged compiler flags verify. Compared with356,
the packed gate header changes, together with the existing optional747
projection-stride header; the whole provider uses its unchanged contiguous
default. [Full build evidence](../benchmarks/correctness/packed-gate32-build-20260923.json).

The same artifacts pass five original model controls: ordinary q8192/out512,
native MTP q7169/out32, q8191/out32, q8192/out512 and q8193/out32. All1120
outputs and callbacks match; every first logit is exact. The actual8192+1
cold bridge and native commit boundaries pass. Ordinary q8192 loading is
21386.521ms, TTFT23392.4967ms and TPOT101.240904ms. These functional runs
do not meet the10000ms TTFT requirement or qualify retained performance.
[Short model evidence](../benchmarks/correctness/packed-gate32-products-20260923.json).

The complete original256k owner and both512-token suffix requests are now
running with unchanged GB10 outputs, host guards and28800-second deadline.
Output observations select263291/input471, where the diagnosed state error
was observed, and263356/input279, which precedes the old output189 failure.
All three CLI provenance annotations now match the build verified before
launch; checks reject each stale annotation. Original GB10 tensors at263356
are being captured independently. Full256k, native retirement, final package,
protocol matrix, soak and release remain unqualified.

The preceding c268 256k run first emits the wrong token at suffix output 124.
Its earlier numerical cause is a packed B projection at input position263238:
layer5, head13 produces BF16 `0xbebc` instead of original GB10 `0xbebd`.
This is reproduced on baiying with the original normalized input and model
weight, across all eight projection carrier/chunk configurations. A and QKV
at the same position match completely; guards and immutable inputs pass.

The qualified original reference reproduces all608 output IDs and complete
first logits. It supplies124 consecutive layer5 transitions, with exact state
continuity and251 bit-identical shared surfaces at each of the three older
observation positions. All33,098 downloaded files are verified.
[Reference and immutable boundary](../benchmarks/correctness/gb10-layer5-recurrent-history-20260921.json).

Both independent and chained host recurrence replay match every original
state/core bit for all124 steps. Replacing only the diagnostic B operand at
263238 with the independently measured GPU value then reproduces all four
actual native state endpoints at263290 and263291 bitwise. This is an offline
causality check; no reference activation is supplied to model inference.

The FP32 dot equals `-0.3681640625`, exactly a BF16 midpoint. The compensated
double dot is `-0.36816407192964107`, slightly below it. Rounding the FP32 value
again chooses the even BF16 neighbor and loses that distinction. The original
GB10 endpoint selects the lower value. This changes the head's BF16 beta from
0.408203125 to0.41015625 in the failing native run, and its error accumulates
until the first output-token difference.

The earlier 356 repair resolves exact midpoints only for K2048,32-row packed
gate projections. A compensated double sum determines the side before the
final BF16 rounding. Non-midpoints retain the existing calculation. The rule
contains no token, position, layer, head or reference selectors.

The actual edited header passes all7,936 original A/B endpoints across124
steps; it changes only the one failed output. ASan/UBSan checks also pass37
exact dyadic midpoint, carrier and cancellation controls, including four
failures of the old rounding. Source `35607853eb03487b443ddfed8529b8da5539de86`
then compiles and passes actual gfx1151 GPU replay on baiying: focused A/B/QKV,
all248 history projections and364 prior projection/shared-activation cases.
All615 cases,4,664 configurations and7,477,772 element comparisons match the
original operands bitwise, including the corrected B/head13 value. Guards,
immutable inputs and host cleanup pass.
[Fixed native component evidence](../benchmarks/correctness/packed-gate-midpoint-native-20260922.json).

The whole provider, same-source CLI and prefix probe also compile successfully
on baiying in121916.409ms. Compiler flags are unchanged; the packed gate
header is the only changed input among173.
[Full build provenance](../benchmarks/correctness/packed-gate-midpoint-build-20260922.json).
The ordinary q8192 and four native-MTP short model controls now pass all1,120
original output IDs, callbacks and first logits. Ordinary q8192 load is
21517.6512ms, TTFT23272.0441ms and TPOT101.032956ms.
[Original-model short controls](../benchmarks/correctness/packed-gate-midpoint-products-20260922.json).
The repaired complete 256k run now finishes with native exit 6. Its first 189
suffix outputs match GB10; output 189 is 2468 instead of 8240. The previous
output 124 now correctly equals 4980. Both first-token logits and retry
restoration pass, but the later owner continuation, timed suffix and negative
branches are not reached. No retained performance or release acceptance is
claimed.
[Native failure, causal replay and candidate checks](../benchmarks/correctness/layer5-small-gate-midpoint-diagnosis-20260922.json).

At the same captured input positions 263290 and 263291, all compared surfaces
through layer 7 now match bitwise, including the repaired layer 5 state.
The earliest remaining observed difference is layer 8's incoming recurrent
state. The subsequent continuous-history diagnosis below establishes its
earlier cause; the causal link to output 189 still requires model validation.
The completed run retains all 1,559 raw observations with verified hashes,
passes host checks and leaves no native process running.
[Completed run and remaining mismatch](../benchmarks/correctness/packed-gate-midpoint-prefix256k-remaining-failure-20260922.json).

The new unchanged-compute GB10 capture reproduces all 608 original outputs
and complete first logits. It captures 124 consecutive layer 8 transitions;
all state links are bit-exact. Every raw boundary file is verified remotely,
and 1,379 compact files are verified locally. Full states remain on GB10;
the controller receives head 26 slices and the relevant complete operands.
All 124 normalized inputs and the 11 shared surfaces at each older endpoint
match the preceding reference.

Current Windows source 356 reproduces two gate errors across all eight GPU
configurations: B/head19 at 263196 is 47936 instead of 47935, and A/head26
at 263256 is 15736 instead of 15737. The 250-case replay verifies 194,560
elements with 16 mismatches, intact guards and immutable inputs. Two complete
QKV controls pass; the host also matches all 47,616 target Q/K/V values.

Original independent and chained head26 recurrence match every state/core bit
for all 124 steps. Changing only the A/head26 value at 263256 to the measured
AMD value reproduces every bit of all four native state endpoints at
263290/263291. This establishes the source of that observed state difference.
B/head19's downstream effect remains separate. Neither erroneous gate is an
exact FP32 midpoint. Higher precision alone is insufficient: the compensated
double A dot selects the current endpoint, while GB10 selects its neighbor.
[Reference, native reproduction and causal replay](../benchmarks/correctness/layer8-gate-state-diagnosis-20260923.json).

A bounded reduction-order sweep finds that 32 strided chains with a folding
halves tree match all 15,872 original layer5/layer8 gate endpoints. Applying
the old midpoint correction to that order introduces two errors. The new
candidate implements the matching order directly, with no token, position,
layer or head selectors and no added runtime artifact or workspace.

Three CLI provenance fields in that run's emitted metadata still name the old
build. The original success observer correctly rejects them. The pinned
launch script instead verifies the repaired build metadata, and preflight
records the repaired executable's exact SHA. The diagnostic account preserves
both facts and the original records; it does not promote the failed run.

The [portable package preparation](../benchmarks/correctness/packed-gate-midpoint-package-prepared-20260922-r2.json)
binds the actual repaired whole provider and CLI. The server retains its c268
binary and source identity: all25 build inputs are unchanged in356 and both
Windows checkouts are hash-verified. Six component files and the portable
profile are checked. The original HTTP, cold and one-hour workloads remain
intact, but no new archive, HTTP result or soak result exists yet.
