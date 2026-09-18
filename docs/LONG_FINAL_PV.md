# Deferred PV bounds for long attention

Component source `e0d7e4a` extends deferred error bookkeeping to16546 K16
groups, covering264736 keys. Original probabilities, online rescaling, native
matrix accumulation, reciprocal and selected original replay are unchanged.
The native error coefficient remains2^-19. Calls through512 groups delegate
to the existing short finalizer and preserve its bits.

The existing capped metadata has a per-call comparison factor1+2^-17. For
G groups and at mostG/2 rescalings, M=3G/2 calls have growth bounded by
1/(1-M*2^-17). The denominator is exact in FP32 over this range, and its
reciprocal is rounded outward. The existing2^100 metadata cap and64*G*2^-118
floor remain. Invalid inputs select original replay. The source documents
the derivation; exact rational/FP32 checks cover all8017 long even lengths.

Host ASan/UBSan and native gfx1151 checks each perform23652092 envelope
comparisons, including weakened metadata, broad exponents, cancellation,
zero/subnormal cases and overflow. There are no underestimates or false
admissions. All887986 checked short finalizations retain identical bits.
Native100 configurations across10 shapes and5 data modes pass through the
maximum key extent, comparing10608640 distinct cells with complete original
scalar PV. Fifteen short-template regressions pass. Candidate sets are
supersets of the original; selected and unselected raw outputs, native
arithmetic, denominators, guards and immutable data pass.

| Queries after16k history | Original per-group bound ms | Deferred bound ms | Original / deferred candidates |
| --- | ---: | ---: | ---: |
| 1024 original queries | 318.2968 | 263.5991 | 435091 / 438883 |
| 8192 extended queries | 3924.3890 | 3418.2800 | 7582216 / 7687123 |

Each median sums complete128-query slabs across the full operator, with one
warmup and three rotated samples per slab. QK, probabilities/native PV,
compaction and every selected original replay are included. All candidate
samples are below all controls. Shared range/domain preparation and V
transpose add4.8774/7.7593 ms respectively. References, resets and validation
are outside clocks. Every attempt passes original arithmetic and the original
4194304 GB10 context cells; the added7168 rows have no model-token oracle.
Warmups also check all native accumulators before exact replay overwrites
selected cells. These are complete component timings, without model loading
or token generation.

The first build called a device-only BF16 helper in the host observer. Its
failure is preserved; the corrected observer uses the existing host/device
helper. Arithmetic and bound headers are unchanged across that correction.

The provider option `QRT_CK_SM121_LONG_FINAL_PV_BOUND=1` is default off and
applies only when the validated long-attention pipeline is active. It adds
no workspace and leaves short requests and single-query decode on their
existing paths. Contiguous-owner and real-model qualification are pending.
The qualified q8192 model TTFT remains23353.80795 ms; package and release
gates remain open.

[Pinned sources, commands, failures and native results](../benchmarks/correctness/long-final-pv-native-components-20260918.json):
157778 bytes, SHA256
`3581c1fd7bd0ab49a83cb84e1f1fb6901f4f6160d3fbfd2d8006f95c67aa51f5`.
