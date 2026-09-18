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
existing paths. Provider source `ea6faff` passes the native bound and safety
suite again. Eight captured contiguous owners check output offset3, exact
surfaces, full candidate membership, five ordered stages and rejection of
undersized storage before submission. All301 build inputs match that commit.
The DLL is2101760 bytes, SHA256
`e10b2db6fefcd55dc1848895f0eb3e11e67bdacb0b8368a2e9b993468c355ad8`.

## Real model comparison

Four fresh baiying processes run `D:\models\Qwen3.6-35B-A3B` with the same
DLL and original16384-token GB10 prompt. Only the new bound option changes;
the long pipeline stays enabled, both cold chunks complete, and profiling
is disabled. Whole `ddacdc9`, MoE `9235750`, FLA `7b20c90`, CLI `24c4304`
and all other environment values remain fixed.

| Order / option | Load ms | TTFT ms | TPOT ms |
| --- | ---: | ---: | ---: |
| 1 / OFF | 21322.713401 | 76696.968301 | 155.819545 |
| 2 / ON | 21335.445900 | 73847.508901 | 164.128613 |
| 3 / ON | 21341.499800 | 73832.769400 | 164.606855 |
| 4 / OFF | 21334.744300 | 77168.805599 | 161.040390 |

All128 original GB10 continuation IDs, prompt identities, first logits25.625
with zero error and actual callbacks pass. OFF/ON TTFT medians are
76932.886950/73840.139151 ms, a3092.747799 ms (4.02%) reduction. Both ON
samples beat both OFF samples. Every load is below30000 ms. The TPOT medians
158.429968/164.367734 ms show no decode improvement; this option does not
change single-query decode. Two observations per arm do not characterize
broader timing variance.

A separate q8192/out512 run with both long options requested passes every
original GB10 ID, prompt, first logit10.375/error0 and512 actual callbacks.
The long pipeline is inactive for cold8192 and single-query decode, with all
short-route activation checks passing. Load/TTFT/TPOT are
21328.263301/23251.887700/100.737632 ms. This unpaired regression does not
replace the qualified q8192 median23353.80795 ms or establish a short-route
speed gain.

Retain the enabled deferred bound as the experimental long-context control.
The16/32/64/128/256k prefix gates for this DLL remain open; earlier prefix
results belong to the preceding DLL. Code and package defaults remain off.
The10000 ms first q8192 gate and retained4187.415605 ms target are unchanged.
No package or release is qualified.

[Provider tests and all five real-model runs](../benchmarks/correctness/long-final-pv-provider-product-20260918.json):
1034184 bytes, SHA256
`392367e1f0ef23049448260525f5efbba3bee4325e0da53baea55bb3ba8790b3`.
Commands are `run-long-final-pv-provider-r1.ps1`,
`run-long-final-pv-product-r1.ps1` and
`run-long-final-pv-q8192-product-r1.ps1` on baiying.

[Pinned sources, commands, failures and native results](../benchmarks/correctness/long-final-pv-native-components-20260918.json):
157778 bytes, SHA256
`3581c1fd7bd0ab49a83cb84e1f1fb6901f4f6160d3fbfd2d8006f95c67aa51f5`.
