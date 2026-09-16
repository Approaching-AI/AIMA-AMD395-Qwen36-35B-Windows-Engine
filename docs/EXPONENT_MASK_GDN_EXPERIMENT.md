# Shared exponent masks for GDN

Isolated source `185546a` applies the exact top-three exponent certificate to
the original GDN W/U, state and output matrices. Original packed BF16 words,
ordered K16 carries and exceptional replay remain unchanged. Separate shared
metadata supplies the certificate; an unresolved group scans the original
exponent pairs. A second output tile covers 32 rows instead of 64. An unmasked
32-row ablation separates that tile change from the certificate. No production
dispatcher or runtime option uses these kernels.

The native fixture executes q8192 as eight consecutive 1024-token segments,
matching the current provider's segmentation and preserving FP32 state across
segments. Both distinct W/U storage and the product's U=V alias are checked.
Every warmup and three rotated samples compare full output, final state,
checkpoints, W/U and residual buffers, with input ownership and redzones.
This segmentation differs from the earlier single-recurrence GDN fixture;
timings from those two fixtures are not directly comparable.

Local C ABI, public hygiene and sanitized host checks pass. Host checks cover
all 65,536 BF16 encodings, 327,680 certificates, 262,144 chained groups and
2,048 additional extreme carry-bound checks. Packed metadata is checked
against independently decoded original words. No new full Rust/Python suite
run is claimed. The baiying native build completes in 4,392.268 ms, followed
by successful safety and throughput processes, with all host guards passing.

The 144 safety reports cover lengths 1, 63, 64, 65, 129 and 1024, zero,
finite and subnormal modes, four variants and both alias configurations.
Independent CPU comparisons cover 58,812 safety dots and 172,032 q8192 dots.
The complete q8192 comparisons check 1,073,741,824 output words, 16,777,216
final-state words, 2,147,483,648 checkpoint words and 3,221,225,472 W/U and
residual words. All are bitwise equal to the original operator. These are
generated operator inputs, not a new real-model or GB10 continuation gate.

| Route | Distinct U/V total ms | Product U=V total ms |
| --- | ---: | ---: |
| Original scalar, output 64 | 72.3771 | 69.4084 |
| Masked matrices, output 64 | 73.9003 | 71.4949 |
| Masked matrices, output 32 | 66.9500 | 72.1455 |
| Original scalar, output 32 | 69.9001 | 66.9655 |

Totals are medians of completed W/U, state and output execution over all eight
segments. Allocation, reset and validation are outside the timers. In the
product alias configuration the masked W/U phase improves from 14.099681 to
10.959060 ms, but state increases from 32.659340 to 35.868038 ms and output
from 22.621920 to 24.837101 ms for the 64-row tile. Phase medians need not sum
to the median total. The masked 32-row samples span 63.3043 to 74.3766 ms;
the distinct-storage result alone does not establish a product improvement.

All kernels declare wave32 and zero private bytes. Original/masked W/U use
78/83 VGPRs and 10,816/13,376 LDS bytes; state uses 113/128 VGPRs and
42,820/51,780 LDS bytes. Original output64, masked output64, masked output32
and unmasked output32 use 80/86/74/71 VGPRs and
28,736/35,648/19,776/15,936 LDS bytes. These are static resource declarations,
not measured occupancy.

Keep all variants isolated. Both masked product-alias totals regress, while
the unmasked tile change saves only 2.4429 ms in this component observation.
No provider integration or new model run is justified by this experiment.
The retained stack, 10,000 ms TTFT gate and 4,187.415605 ms retained target
remain unchanged; prefix, long-context, package and release gates remain open.

Evidence: [source, bounded commands and complete comparisons](../benchmarks/correctness/exponent-mask-gdn-native-components-20260916.json),
379,850 bytes, SHA256
`bcb161b0eb06ecee12d4bd971560f1f6f73f79253d01a45c2421fa4c7574c174`.
