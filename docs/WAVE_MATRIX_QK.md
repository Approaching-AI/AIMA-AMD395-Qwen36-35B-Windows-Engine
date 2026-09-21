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

A further isolated variant keeps each lossless signed16 coefficient while
storing an unrepresentable element's original BF16 word in the same slot.
A16-bit exception mask makes the matrix use zero for that slot. The exact
matrix remainder is aligned normally, and only omitted products are expanded
and individually aligned using original narrow arithmetic. Actual exponents
and nonzero masks include all original elements. The prepared row grows from
60 to64 bytes. Unsupported domains and shifts still use the original group;
no approximate operand or partial maximum is substituted.

The partial wave owner broadcasts every key field uniformly before sparse
correction or fallback. It keeps the same four-wave output ownership and
ordered carries. All existing wave variants and their prepared rows remain
unchanged. This candidate is not connected to product dispatch.

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

The fixture now also accepts `full8192` with all original Q, K, V and context
rows from the qualified `q8192-out512` GB10 transaction. This mode repeats no
rows and compares all 33,554,432 context values. The comparison counts its
actual reference extent separately from the executed shape; shorter historical
captures keep their original boundary, while every executed output must remain
finite. Five native fault controls cover the final original feature, a shorter
reference and a nonfinite unreferenced output. Those GPU controls remain unrun.

The extracted comparison kernel passes 36 controller execution-coordinate
checks under ASan/UBSan. Sampling the original q8192 Q/K tensors also passes
65,536 dots and all 1,048,576 ordered groups: 708,453 use the original matrix
certificate, 143,942 use exact remainder correction and 196,181 use partial
coefficients with 219,139 individually aligned original products. Every carry
matches the independent original arithmetic. This sample does not compare
external raw QK scores or establish GPU context correctness or performance.
[Original-reference bindings, sample and comparison-scope checks](../benchmarks/correctness/original-q8192-fixture-local-20260921.json).

The corresponding r3 native plan freezes source `5d63d5c` and all 59 quoted
include files plus two host guard files. It requires a new build, the existing
560 generated cases plus five reference-extent fault controls, and the complete
original q8192 comparison with all five timed variants. All compilation now
waits for the long owner's cleanup and uses the unchanged exclusive host guard.
Preparation rejects 28 damaged attention records and all three premature
attention actions. Native script parsing passes; actual compilation and GPU
execution remain pending. The shared original tensor archive remains local.
[Frozen r3 plan and shared-input preparation](../benchmarks/correctness/original-q8192-native-prepared-20260921.json).

The unchanged extracted partial QK owner passes30 ASan/UBSan controller
coordinate cases using four modeled waves and a128-thread block barrier.
All10,240 selected scores match independent original integer arithmetic;
2,048 use the original q8192 tensors through query8191. All36,923,424 inactive
cells, metadata and redzones remain unchanged. Causal edges, partial tiles
and domain rejection leave the tile for the original producer. Wrong KV-head
and output-head indices are both detected. The scalar matrix model does not
verify physical WMMA semantics, complete GPU attention or its fallback kernels.
[Host ownership checks and deliberately damaged controls](../benchmarks/correctness/partial-wave-qk-coordinate-local-20260921.json).

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

The [partial-coefficient host audit](../benchmarks/correctness/partial-matrix-local-20260921.json)
checks192322 ordered groups throughK8192 and8251456 lossless prepared words
under ASan/UBSan, including every possible BF16 bit pattern. It preserves47104
original and14336 remainder admissions, adds75386 sparse-correction groups
and retains55496 original fallbacks. All raw carries and rejection sentinels
match. The other two original matrix tests pass unchanged.

On the same deterministic captured Q/K sample, all remaining199940 q7169
and201208 q8192 groups now use exact sparse correction, requiring223379 and
224548 individual products respectively. All1048576 groups per shape match
original ordered arithmetic. This does not predict GPU speed or replace the
complete GB10 context comparison.

`tests/native/partial_wave_matrix_qk_capture.cpp` keeps all four earlier timed
variants and adds the partial wave owner. Safety also keeps forced original
wave replay and adds forced original partial-row replay, giving560 generated
cases. CPU metadata, original raw scores and consumer surfaces, GB10 context,
all candidate sets, immutable inputs and redzones remain checked. Each variant
reports its own complete metadata preparation cost. These complete GPU
comparisons remain pending for the partial owner.

The [first partial native build](../benchmarks/correctness/partial-wave-matrix-native-20260921.json)
at719fae8 compiles in47944.706 ms with59 compilation inputs and all host checks,
preserving the long owner. Executable SHA256
05d7aaa0e8cf5dfc69257834a69bec0ab387ac05217f31b0794bb938688b5f98
contains172 VGPRs and132 private bytes per lane for the partial candidate.
Its forced original partial-row control uses256 VGPRs,140 private bytes and81
reported VGPR spills. All five prior control kernels retain their resources.
This exposes addressable row storage before any GPU performance comparison.

The revised candidate visits16 fixed exception slots, computing only selected
original products; its forced replay expands original rows once per group
and broadcasts those raw words. The [revised host audit](../benchmarks/correctness/partial-matrix-static-local-20260921.json)
preserves all generated and sampled counts and exact endpoints under ASan/UBSan.
Its [native build and resource comparison](../benchmarks/correctness/partial-wave-matrix-static-native-20260921.json)
at3badddb completes in41449.184 ms with the identical CPU guard,59 compilation
inputs and passing host checks. Executable SHA256
fcf7883061a8ed0a93ce0573935cf5a9cb524122532b02df026beaf60aadf5fc
has zero private bytes and register spills in all seven selected kernels.
The partial candidate uses196 VGPRs/4 shared bytes; its forced control uses132/4.
All five prior controls retain their resources. Physical reserve stays above
17804132352 bytes, and the long owner remains running. The unchanged560-case
safety and five-variant GPU comparisons are still unrun. The first build
and all original fallback paths remain available.

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
