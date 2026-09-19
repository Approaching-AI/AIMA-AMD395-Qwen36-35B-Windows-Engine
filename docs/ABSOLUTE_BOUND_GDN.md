# GDN dot omission with certified consumers

This isolated candidate uses an absolute dot bound to omit a W/H or Q/H
dot only when its original BF16 consumers are constant across the entire
interval. It is not selected by the runtime dispatcher. Native correctness,
complete component timing and real-model performance remain unmeasured.

`absolute_dot_bound.h` accumulates upward bounds on each operand row's L1
norm and maximum magnitude. The smaller of the two L1/L-infinity products
bounds the dot. Original K16 alignment and normalization truncate magnitudes,
so the sum of absolute products also bounds the original accumulator despite
cancellation and intermediate truncation. Positive FP32 norm sums receive a
`1 + 2^-16` inflation and two outward representable steps. Rows have at most
128 values and reject biased exponents above 174. Subnormal BF16 magnitudes
are replaced by the larger minimum normal FP32 value for the bound only;
norm products below that minimum are also bounded by the minimum normal.
No certificate depends on preserving subnormal GPU arithmetic.

The state kernel checks both BF16(U - WH) and the separately rounded
BF16((U - WH) * decay). The original K64 residual update, FP32 FMA, state
storage and next chunk's BF16 checkpoint remain unchanged. The output
kernel computes the original local S/V dot first and checks the original
two multiplies and final FMA across the Q/H interval. A rejected certificate
uses the original scalar dot. Rejected cells enter a bounded shared queue
so waves execute packed replay work. Queue ordering cannot change a cell's
dot or any recurrent update.

Host checks execute the actual state/output kernel bodies with barriers,
shared queues, partial chunks, unusual magnitudes, zero checkpoints and
canaries. They compare every owned state, checkpoint, residual and output
against the independent integer accumulator. A separate 32,768-dot check
covers full exponent spreads, cancellation, subnormals, zeros and invalid
inputs under ASan/UBSan.

An audit of the two original GB10 q7169 captures checks 29,364,224 W/H
consumers per layer. Both residuals are certified for 8,591,442 layer-zero
cells (29.26%) and 14,760,880 layer-one cells (50.27%). Every admitted
unscaled residual matches its captured value. Per layer, 131,072 independent
original W/H and Q/H dots stay inside their bounds; 65,536 residual samples
also check the scaled result, and 4,096 complete reconstructed outputs match
GB10. The output certificate admits 1,143 and 1,822 of those output samples.
These are admission observations, not measured speedups.

A separate [seeded sequence audit](../benchmarks/correctness/absolute-bound-gdn-seeded-sequence-20260919.json)
reconstructs the complete original 8192-token windows at layer 16/position
90112 and layer 33/position 114688. Each window uses its original initial
state, then carries each computed FP32 state into the next 64-token chunk.
All 33,554,432 output BF16 cells and 524,288 final FP32 state cells per layer
match GB10 bit for bit. Both residual consumers and the output consumer are
checked against every original W/H and Q/H dot, with zero false admissions.

| Original seeded window | State dots admitted | Output dots admitted |
| --- | ---: | ---: |
| Layer 16, position 90112 | 923,629 / 33,554,432 (2.75%) | 698,608 / 33,554,432 (2.08%) |
| Layer 33, position 114688 | 5,821,286 / 33,554,432 (17.35%) | 4,276,959 / 33,554,432 (12.75%) |

These CPU observations show that admission depends strongly on the layer
and its incoming state. They do not measure native GPU speed or qualify
full-context model inference. The independent first-chunk reconstruction
also matches captured normalization, inverse, W/U, residual and checkpoint
boundaries in both early-layer controls.

`absolute_bound_gdn_selftest.cpp` compares the retained complete chain with
bounded state only, then bounded state plus output. Paired scores and W/U
stay original. The prepared native checks include production U=V ownership,
seven generated families, eight boundary shapes, full original q7169
captures and the existing q8192 component extension. A positive component
result still requires a real q8192 model run with the matching GB10 token
and logit boundary before any product performance claim.
