# Packed gate midpoint diagnosis

The current c268256k run first emits the wrong token at suffix output124.
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

The candidate repair resolves exact midpoints only for K2048,32-row packed
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
The complete256k continuation is running and remains unqualified. No retained
performance or release acceptance is claimed.
[Native failure, causal replay and candidate checks](../benchmarks/correctness/layer5-small-gate-midpoint-diagnosis-20260922.json).
