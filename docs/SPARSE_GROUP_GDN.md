# Exact sparse K16 groups in GDN

Source `8514689` derives nonzero masks from the original BF16 operands in
W/U, state and output kernels. An empty mask intersection omits its zero
products while preserving the original carry, including zero-sign cleanup.
A second variant computes only the original integer products when at most
four positions remain. Dense groups keep their original arithmetic and
fallback. There is no magnitude threshold or reference-driven selection.
The ordered K16 recurrence, BF16 checkpoints, final FP32 state and U=V
ownership remain unchanged.

UBSan host checks cover all 65,536 BF16 encodings, all 65,536 support masks
and 655,360 raw carry comparisons. Native safety passes 288 configurations
across eight lengths, six input families, three variants and both U ownership
modes. They include subnormals, signed zeros, sparse supports, strong decay
and exceptional encodings. All generated q8192 comparisons also pass.

The original GB10 q7169 layer0 capture matches all 29,364,224 output cells,
the equally sized W/U/Vnew references, 113 BF16 checkpoints and 524,288
final FP32 state values. The q8192 extension uses the first 7,168 captured
rows followed by the first 1,024 rows again. Its first 29,360,128 outputs
and intermediates have the external boundary; every extended output and
state is additionally compared with the original kernels. Independent CPU
dots, source/table immutability, redzones and inactive U storage pass.
This fixture runs captured operators, without loading the model or generating
a new prompt continuation.

| Complete WU/state/output median, ms | Original | Empty groups | Empty and small groups |
| --- | ---: | ---: | ---: |
| Generated q8192, dense, U=V | 75.8898 | 85.3721 | 88.1425 |
| Generated q8192, strong decay, U=V | 84.1672 | 83.4632 | 81.6176 |
| Captured q7169, separate U | 73.8157 | 73.9149 | 72.8283 |
| Captured q7169, U=V | 74.3609 | 75.4860 | 75.8443 |
| Captured q8192 extension, separate U | 83.0970 | 85.2511 | 84.6321 |
| Captured q8192 extension, U=V | 83.5097 | 84.7262 | 85.2952 |

One audited warmup precedes three rotated complete host samples per case.
Timed calls omit diagnostic reductions and stores, while retaining all mask
preparation, original fallback and segment completion. Allocation, resets,
transfers, score preparation and observation are outside the clock. This
measures WU/state/output, excluding the other GDN preparation stages.

At q8192, both candidates skip 275,141,450 of 1,207,959,552 K16 groups.
The second also reduces 43,127,242 groups to 99,679,983 original products.
These counts agree across U ownership modes. Samples overlap and vary;
the smaller amount of arithmetic establishes no complete captured speedup.
Static metadata declares zero private storage; it does not measure occupancy
or attribute the timing difference to a particular resource.

Keep both variants isolated. The qualified model TTFT remains 24,835.3024 ms.
No provider dispatch, package, retained target or release changes follow.

[Source, commands and complete evidence](../benchmarks/correctness/sparse-group-gdn-native-components-20260918.json):
575570 bytes, SHA256
`afe6f61b926ca8c3f06fdad02b25de9932e39245b8564a42d665fb1d7c8b8da2`.
