# Independent quantized-map audit

The checked signed64 oracle in `tests/native/checked_quantized_affine_map.h`
represents `floor_grid(x + before, 2^shift) + after`. It checks overflow and
leaves outputs unchanged on rejection. Wide integer division independently
checks its rounding; accepted compositions also crosscheck the existing
modulo64 implementation in `sm121_dyadic_carry_scan.h`.

For grids A and B and `t = first.after + second.before`, composition uses
`floor_B(floor_A(y) + t) = floor_B(y + floor_A(t))` when B is at least A;
otherwise it uses `floor_A(y) + floor_B(t)`. Negative-domain truncation is
expressed as `floor_grid(x + q - 1, q)`. These identities do not establish a
floating-point domain by themselves.

The host K16 audit predicts signs and exponents with ordinary FP32 tree sums
and prefix additions. It constructs maps from original individually aligned
products, then validates every derived input/output sign, alignment grid,
normal FP32 representation and output truncation before admission. Original
wide K16 arithmetic supplies comparison results only. Deliberately incorrect
grid/sign predictions must fail admission.

Sanitized checks pass 6990291 pair evaluations, 1000000 triple evaluations,
1000000 signed rounding steps and 7004 checked rejections. Generated dots
admit 562064 original K16 boundaries without a mismatch; all 32768 deliberately
corrupted predictions reject. The deterministic 65536-dot causal sample from
the original layer3 q7169 Q/K capture admits all 1048576 boundaries, again
without a mismatch. This sample does not qualify all layers or model inference.

This algebra already has a native implementation. Its q8192 component result
was 1066.5525 ms for the fastest scan versus 332.3032 ms for its control.
The new checked oracle supplies independent validation, without evidence of
a faster GPU layout. Keep it in tests and do not repeat that native route on
host coverage alone. The current qualified experimental product remains
24835.3024 ms TTFT; runtime, package and release status are unchanged.

Evidence: `benchmarks/correctness/checked-quantized-map-host-20260918.json`
and `benchmarks/correctness/dyadic-scan-qk-native-components-20260916.json`.
