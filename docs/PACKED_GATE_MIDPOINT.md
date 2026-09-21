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
failures of the old rounding. Windows compilation, fixed GPU replay and the
full original-model continuation are still required. No new inference,
performance or release acceptance is claimed.
[Native failure, causal replay and candidate checks](../benchmarks/correctness/layer5-small-gate-midpoint-diagnosis-20260922.json).
