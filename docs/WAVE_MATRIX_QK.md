# Wave ownership for exact matrix QK

This component keeps the matrix result and its ordered carries in the wave
that computes them. It is isolated from runtime dispatch. The native Windows
build at804ccf4 passes. The subsequent exact-remainder variant atb9ff29d also
passes host arithmetic checks and native compilation. GPU numerical behavior,
the complete q8192 component and the original-model boundary remain unmeasured.

The preceding [compact matrix queue](COMPACT_MATRIX_QUEUE_QK.md) passes its
numerical checks but costs 989.6595 ms for complete q8192 attention plus
preparation, against a 540.3715 ms retained control. That owner stages two
K16 groups, their expanded operands and all matrix products in shared memory,
then redistributes results to scalar carry owners. These results motivate a
replacement of ownership and synchronization, rather than a larger queue.

## Candidate

Each 128-thread block owns a complete 32-query by 32-key tile. A single block
barrier publishes its complete-domain admission decision. Four waves then
independently own 16 by 16 tiles. Transposing the integer matrix operation
lets each lane retain its query row, eight outputs and their original ordered
K16 carries. The four IU8 matrix products reconstruct each signed16 dot exactly.
No matrix result or carry is exchanged through shared memory between groups.

Only eight words of each key's original carry certificate are broadcast:
control, four exponent words, two trailing-bit words and the first packed
coefficient pair. This 32-byte view supplies every field used by the existing
certificate. It does not replace the lossless 60-byte prepared row, change
matrix coefficients or change their storage.

The original exact-alignment certificate and K16 normalization remain. A
rejected group leaves its input carry untouched and runs the existing original
narrow-domain group. When any output needs fallback, every source lane takes
part in each shuffle before the rejected consumers compute. Unsupported
complete tiles, causal edges and partial tiles retain the existing general
narrow producer and complete-dot fallback.

This differs from the earlier scalar direct/wave-queue variants in both
matrix ownership and data exchange. It adds no reference operand and removes
no original candidate, numerical bound or fallback.

An additional isolated variant retains the original certificate first. For
otherwise rejected positive shifts through16 bits, it subtracts the signed
discarded residue of every original coefficient product from the exact matrix
sum before alignment. This preserves per-product truncation; it does not
round the uncorrected complete dot as one value. The existing compact integer
calculation supplies the residue formula. Unsupported rows and wider shifts
keep the original fallback, and the original certificate keeps its admissions.
All encoded-key shuffles execute uniformly before any fallback consumer.
This variant remains outside product dispatch.

## Verification

The host audit compares the compact metadata view with the full row and the
independent original integer arithmetic over 175680 ordered K16 groups through
K8192. All endpoints and rejection behavior match under ASan/UBSan. It covers
47104 matrix-admitted and 128576 fallback groups, 7495680 lossless words and
trailing metadata, and the preceding 1048156 queue source checks. These are
host arithmetic checks, not GPU output or speed evidence.

`tests/native/wave_matrix_qk_capture.cpp` reuses the existing complete-attention
fixture. Its timed variants are retained narrow QK, the previous matrix wave
queue, and the new wave owner. A fourth safety-only variant forces original
fallback for every new-wave group. All eight shapes and ten data families
retain original raw score, probability, alpha, denominator, output, candidate,
CPU metadata, immutable-input and redzone checks.

The q7169 capture carries all 29364224 original GB10 context cells. q8192
adds 1023 repeated captured rows for a full component shape, checking those
extra outputs against original arithmetic. It is not a new real-model prompt.
Complete attention, metadata preparation and common preparation are reported
separately; every required stage must be included in a comparison. One warmup
and three rotated samples retain their numerical checks.

The [remainder host audit](../benchmarks/correctness/wave-matrix-remainder-local-20260921.json)
compares175680 ordered groups against the independent original integer model
under ASan/UBSan. It preserves47104 existing admissions and adds14336 exact
remainder admissions, leaving114240 original fallbacks; all raw carries match.
The original compact-metadata and queue-source test also passes unchanged.
These generated distributions do not predict model-data admission or speed.
`tests/native/wave_matrix_remainder_qk_capture.cpp` keeps all four timed
variants in one executable: retained narrow, old matrix queue, original wave
owner and exact-remainder wave owner. Forced original-wave replay remains a
fifth safety variant, giving400 generated cases with the same complete checks.

The [remainder build and captured-operand audit](../benchmarks/correctness/wave-matrix-remainder-native-20260921.json)
binds sourceb9ff29d and57 compilation inputs to executable SHA256
fd6070ed104b3eb0c22f63109bf5ef24ddfa3d80e44d35a493eee2b5fd885701.
The baiying CPU compiler completes in41929.904 ms, retaining the declared long
owner and all host checks. Physical reserve stays above20070580224 bytes.
No GPU fixture executes during this build.

An ASan/UBSan host sample checks65536 original-operand dots/1048576 ordered
groups per shape. At q7169, the original certificate admits706470 groups,
the remainder path adds142166 and199940 retain fallback. At q8192, the counts
are706919,140449 and201208. Every carry matches the independent original
integer arithmetic; all remaining fallbacks involve an unsupported compact
row. The q8192 extension still repeats captured rows. The historical producer
exits6; only its independently GB10-matched Q/K surfaces qualify as inputs.
Its downstream mismatch is preserved, and these samples establish neither
GPU numerical correctness nor an inference or speed result.

The [native build record](../benchmarks/correctness/wave-matrix-qk-native-build-20260921.json)
binds source804ccf4,56 compilation inputs, the original compiler flags and
executable SHA256be45deaa2893604679cf3134d61649fd05d837999767a20ac1893a90947cb128.
It compiles on baiying in46403.676 ms. The separately owned CPU compiler job
keeps the declared long diagnostic owner running, preserves all host checks,
and launches no GPU fixture. Its complete q8192, GB10 and safety measurements
must wait for that owner's completion and cleanup.

The compiled gfx1151 kernels have these resources. Every listed kernel has
zero private memory and zero reported register spills.

| Kernel | VGPRs | Shared bytes per block |
| --- | ---: | ---: |
| Retained narrow control | 167 | 24576 |
| Earlier compact matrix wave queue | 73 | 28160 |
| New matrix wave owner | 140 | 4 |
| Exact-remainder matrix wave owner | 166 | 4 |
| Forced original-wave safety | 89 | 4 |

These compiled resources do not establish measured occupancy or a speedup.
The current ordinary q8192 functional TTFT
is 23134.0106 ms; its below-10000-ms gate, retained performance target and
release requirements are unchanged.
