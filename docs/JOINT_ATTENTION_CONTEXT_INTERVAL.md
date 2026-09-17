# Joint attention-context interval

Source `ed1354c` adds an isolated BF16 output certificate for uncertain PV
numerators and attention denominators. It checks their complete Cartesian
product, rather than requiring an exact numerator before denominator
refinement. No runtime dispatcher uses it yet.

The caller must provide valid arithmetic enclosures and the original
SHA-verified reciprocal table. For denominator intervals in `[1,2^19)`, the
helper encloses possible reciprocal-table excursions with two FP32 output
steps at each endpoint. A point denominator uses its exact table result.
All four rounded numerator/reciprocal corners must produce the same finite
BF16 value. Nonfinite, reversed and subnormal product intervals decline;
nonzero underflow to zero also declines. Signed-zero point intervals preserve
their sign, while mixed signed zeros decline. Declining leaves output intact.

The host audit uses UBSan and independently enumerates representable points
near BF16 rounding boundaries. It covers 65536 intervals, all 19 reciprocal
exponents, both numerator signs, 9732096 admitted Cartesian points and
2162688 reciprocal points. Half the generated intervals admit and half
decline. Exceptional inputs and null pointers are checked separately.

On baiying/gfx1151, all 65550 native cases match the host decisions and
outputs, including 14 exceptional/signed-zero cases. The same 9732096
Cartesian points and 2162688 reciprocal points pass with zero false
admissions or escaped reciprocals. Input bytes, table bytes and all redzones
remain intact. Build/test processes complete normally with every host guard.

This qualifies only the arithmetic helper. It does not establish bounds on
model-derived numerators or denominators, reduce runtime replay, or qualify
model tokens or performance. The next complete component must construct its
own enclosures, preserve original QK/PV fallback, and compare against original
arithmetic and the GB10 context. Golden values remain comparison-only.

[Pinned source and native evidence](../benchmarks/correctness/joint-attention-context-interval-native-20260918.json):
85759 bytes, SHA256
`95c50250e514e5f98dde5b274bc35aeccbc2457b212330da019bea10874454d6`.

The qualified experimental model baseline remains 24835.3024 ms q8192 TTFT.
The 10-second boundary, retained performance, long contexts, package and
release acceptance remain open.
