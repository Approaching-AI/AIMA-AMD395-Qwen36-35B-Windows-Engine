# Scalar QK wave size, 2026-09-16

Source `6a9f15a5473114a3b29fc718ef3161ad350d0be0` extracts the existing
deferred scalar QK kernel into one translation unit compiled separately
for wave32 and wave64. The shared layout, 256 threads, 16x16 output tile,
prepared operands, scalar arithmetic, K16 order, masks and complete fallback
remain. The reference, preparation and replay kernels remain wave32.
Each native numerical action first verifies device `warpSize` and the
population of an all-active ballot for both objects, with private guards.

The first build compiled both objects but failed at the combined main
compile/link step: the Windows HIP wrapper treated `.obj` inputs as source.
Source `faea6cd2b6d32a440b0bfe0328a466059bbd1660` passes those COFF objects
directly to the linker. Kernels and the fixture are unchanged. A subsequent
capture command used an incorrect input directory suffix and stopped before
native execution. The corrected command restores the original hashed capture
path and reuses the same successfully built and checked executable. Both
failures and their fixes are attached to the evidence.

Full local checks pass 54 Rust and 472 Python tests (two skipped), C/q16
ABI, clippy and hygiene. The driver-only fix additionally passes Python
compilation/help, diff and hygiene checks. Native rebuild takes 15441.899 ms;
80 generated shape/variant cases pass, including 5120 independent CPU dots.
Both wave widths execute as declared. All captured scores, encodings,
original inputs, masks, unused output tails and guards pass on every attempt.
The two captures check 15420096768 score slots and 1936 CPU dots across arms.

| Completed QK and fallback, ms | q7169 capture | q8192 extension |
| --- | ---: | ---: |
| Retained inline fallback | 268.4966 | 349.6946 |
| Existing deferred fallback | 247.5794 | 324.9028 |
| Extracted wave32 | 248.1604 | 324.9551 |
| Extracted wave64 | 244.1893 | 317.8057 |

Add the common measured preparation of 7.4507/9.0757 ms respectively.
Each arm has one warmup and three completed samples per 128-query slab,
with rotating order. The q8192 shape repeats the first 1023 rows of the
original layer-3 q7169 Q/K capture. It is an operator extension, with no
model load, generated tokens or new GB10 continuation evidence.

Wave32/wave64 score kernels declare 81/80 VGPRs, 40/70 SGPRs, 16384 bytes
LDS and zero private allocation. This does not establish occupancy.
Keep wave64 component-only: its q8192 difference is just 7.1494 ms against
the same source in wave32. No product dispatch or retained performance
changes; the real q8192 TTFT and release gates remain open.

Full commands, source and input hashes, compiler steps, executable identity,
failures, device checks and samples are in
[`wave-qk-native-components-20260916.json`](../benchmarks/correctness/wave-qk-native-components-20260916.json),
194334 bytes, SHA256
`0b910851d50bac3856b82867109b22a502648675104dd895b01ab0354dd79765`.
