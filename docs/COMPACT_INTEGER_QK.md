# Compact exact integer QK experiment

The previous scalar integer QK row occupies 156 bytes. This isolated retry
uses a 52-byte row: sixteen signed16 coefficients, the original biased
exponents and one control word. Supported BF16 values are represented exactly
under a common power-of-two scale. Unsupported ranges, exceptional encodings
and negative zero keep the original sixteen BF16 words and use original
arithmetic. Every original bit can be reconstructed; this is not quantization.

Four byte-dot products reconstruct signed16 products. The exact common K16
exponent and each signed discarded remainder reproduce the original
toward-zero alignment. Carries retain width26 and their original order.
Unsupported shifts or ranges fall back to the original group operation.
The host test checks all 65536 encodings in four scale contexts, 4194304
roundtrip words, 109075 independent signed16 products and 500000 raw carries
against `group_sum<26,-133>`. UBSan reports no errors. Its fallback/exact/
remainder path counts are 418314/4076/77610; generated coverage does not
predict captured-model coverage.

The native candidate reuses the current four-score 32x32 tile schedule, with
K64 and K128 windows. Shared row storage is 13312 or 26624 bytes. Register
allocation, spills and occupancy require native inspection. All query
encoding, one-time key encoding and fallback must be included in comparisons
against the current decoded four-score control.

The standalone native harness checks 131072 raw carry states, generated
partial/causal tiles, original score bits, independent CPU dots, complete
row encodings, immutable inputs and memory guards. Its captured q8192 shape
extends the original q7169 layer3 Q/K by repeating the first 1023 rows.
That extension is component arithmetic evidence, not a new GB10 prompt.
Every warmup and timed score is checked outside its timer. Captured encoding
coverage and 65536 independent host carry comparisons are diagnostics.

The initial HIP build rejects default member initialization for shared rows.
Source `eac6932` removes it; local preparation value-initializes every row and
shared tile loads overwrite every word. Host UBSan and all native checks pass:
131072 raw carries, 72 generated configurations, 98544624 generated score
comparisons and 6543114240 captured score comparisons. The failed compiler
run remains in the evidence; it performs no GPU arithmetic.

The full q8192 component is slower: control/K64/K128 totals are
303.1695/2383.5624/2302.1050 ms, including preparation. Supported query/key
groups are 1887537/2097152 and 236179/262144. The 65536 sampled carry paths
contain 12238 fallbacks, 73 exact shifts and 53225 remainder corrections.
All are exact against original arithmetic, but this is not a model run.
The compiler declares 46080/59392 bytes of LDS and 8 bytes of private storage,
which exceed the explicit shared row arrays by 32768 bytes. Occupancy is not
measured. Both schedules remain isolated.

[Complete first native comparison](../benchmarks/correctness/compact-integer-qk-native-components-20260918.json):
161520 bytes, SHA256
`b4c4e2b74a85c8c9b9676463ec98d9f5f6575ff6c405a9b0903f2aea8e9312f0`.
It pins `run-native-compact-integer-qk-r2.ps1`, baiying, every build input and
the original model capture.

The next revision replaces the fallback's wide `Value[17]` temporary with
the established one-word original product encoding and bounded modulo sum.
It retains original products, exponent maxima and canonical normalization.
The same host tests and native safety suite pass at `0813965`, but LDS remains
46080/59392 bytes with 8 bytes private storage. This disproves the wide
fallback temporary as the sole storage cause. No complete capture run was
repeated for that intermediate revision.

Disassembly instead places four 8-byte per-thread carries at LDS offset13312
for K64, adding 32 bytes per maximum workgroup lane; two active flags use
scratch bytes. Source `8310180` names the four carries and flags explicitly
and declares the actual 256-thread launch bound. Native LDS becomes exactly
13312/26624 bytes, with zero private storage and 95/102 VGPRs. All 131072
native raw carries, 72 generated configurations and 6543114240 captured score
comparisons still pass.

Complete q8192 control/K64/K128 times are 313.6175/1427.3859/1521.9112 ms.
Removing the storage problem substantially improves this candidate, but both
schedules remain over 4.5 times slower than the control. Neither is integrated.
Static resource declarations do not measure occupancy or explain all remaining
time. Further scalar scheduling changes are not the current next step.

[Explicit-carry comparison and intermediate diagnostic](../benchmarks/correctness/compact-integer-qk-explicit-carries-20260918.json):
293619 bytes, SHA256
`de71caf247bcc759f78a03f4cbe1618d4c18d054a8853d451f0a1d18fef44d40`.
This record includes both native revisions, disassembly evidence and the
unchanged independent original arithmetic checks.

No provider dispatch, runtime defaults or package changes. The current model baseline remains
23902.4417 ms TTFT; the 10000 ms boundary, retained 4187.415605 ms target
and full GB10 continuation requirements remain unchanged.
