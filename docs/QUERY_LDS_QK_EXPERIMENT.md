# Query shared-memory layouts, 2026-09-16

Source `cbe018d978111454e5b6041c525c0295f5e58777` compares three query
storage permutations with the retained QK kernel and the existing deferred
fallback control. Row padding uses a 132-word pitch; the other candidates
transpose query storage or exchange four-word feature groups on odd rows.
All retain the same prepared operands, 16x16 output tile, 256 threads,
128-feature windows, key storage, masks, scalar products, ordered K16
carries and complete original fallback. Product dispatch is unchanged.

Local checks pass 54 Rust and 471 Python tests (two skipped), C/q16 ABI,
clippy and public hygiene. Sanitized host tests verify all four address
maps, 8192 loaded cells, 131072 consumer reads, padding and guards.
The Windows/HIP build on `baiying` takes 12882.877 ms. All 100 generated
shape/variant comparisons pass, including 6400 independent CPU dots.

| Completed QK including fallback, ms | q7169 capture | q8192 extension |
| --- | ---: | ---: |
| Retained inline fallback | 269.0378 | 355.0120 |
| Existing deferred fallback | 248.3811 | 326.5662 |
| Query row pitch 132 | 245.0511 | 324.6450 |
| Feature-major queries | 253.9373 | 334.2226 |
| Odd-row XOR 4 | 254.5576 | 335.6466 |

Each arm also includes the same measured preparation cost of 7.7182 or
9.2522 ms. Timings use one warmup and three completed samples per query
slab, rotating arm order. Every timed and warmup output is compared with
the original computation. Across both shapes and all arms, 19275120960
score slots and 2420 independent CPU dots pass; original inputs, encodings,
output tails and redzones remain intact. The q8192 case repeats the first
1023 rows of the original q7169 layer-3 Q/K capture. It is an operator
extension, with no model load, token generation or GB10 continuation claim.

Static compiler metadata gives 61/60/64 VGPRs for the three new kernels,
16640/16384/16384 bytes of LDS, no private allocation and wave size 32.
This does not establish measured occupancy or bank conflicts.

Keep the layouts component-only. Padding saves only 1.9212 ms over the
existing deferred control at the larger shape; it does not justify a
product integration while the real q8192 TTFT remains above 10 seconds.
The original retained target of 4187.415605 ms is unchanged.

The complete command, input hashes, native executable, source commit,
guards, samples and scope are in
[`query-lds-qk-native-components-20260916.json`](../benchmarks/correctness/query-lds-qk-native-components-20260916.json),
175864 bytes, SHA256
`a5d29bbe40d878fb6ee081d95efda21a8c599edf38f12556218d8032fd3629d5`.
