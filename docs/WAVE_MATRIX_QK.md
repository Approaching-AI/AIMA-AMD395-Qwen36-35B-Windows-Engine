# Wave ownership for exact matrix QK

This component keeps the matrix result and its ordered carries in the wave
that computes them. It is isolated from runtime dispatch. The Windows kernel,
complete q8192 component and original-model boundary remain unmeasured.

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

The actual compiler's registers, shared/private memory and spills must be
recorded with the complete native measurements. Static resource declarations
alone do not establish a speedup. The current ordinary q8192 functional TTFT
is 23134.0106 ms; its below-10000-ms gate, retained performance target and
release requirements are unchanged.
