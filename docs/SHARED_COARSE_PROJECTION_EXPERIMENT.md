# Shared-memory C64 projection experiment

Source `2c020c174b2396a237501ac013c19638dae94e82` preserves every tested
C64 center, error interval, candidate mask and selected raw output. Both
q8192 operator comparisons are slower than the same-executable vector-load
control. Keep the three shared-memory variants isolated; the runtime,
package, retained performance and release gates are unchanged.

## Implementation and correctness

A 256-thread CTA loads K64 operands through coalesced packed BF16 reads into
padded shared memory. Two fragments per wave cover either 64 by 64 or
128 by 32 outputs; a third variant prefetches the next 64 by 64 tile into
registers. All lanes reach the retirement and publication barriers, including
unsupported operands and partial output tiles. Four-byte `memcpy` supports
independently two-byte-skewed operand pointers without an alignment promise.

The original zero-C signed and absolute K16 WMMA operations, ascending FP32
additions and C64 error recurrence are unchanged. Its native coefficient
remains 2^-19 and remains conditional on the hardware error model; finite
comparisons do not prove a universal hardware bound. The original selector
and complete staged2 K16 replay finish every ambiguous or unsupported cell.

Native safety runs 15 generated cases with all five configurations: 150505
output comparisons, 30101 distinct independent CPU full dots and 27700
selected raw-output checks. Tail tiles, partial K64 blocks, signed zeros,
cancellation, unsupported values, zero weights and independent buffer skews
pass. Every attempt checks complete prepared words and eligibility flags,
immutable input spans, redzones and unused candidate tails.

The captured OUT case checks all 16777216 outputs against its GB10 q8192
operator reference. QKV checks all 67108864 outputs, repeating the first
1023 reference rows alongside the corresponding captured inputs. Both
operators start from 7169 real-model input rows and repeat 1023 rows to
reach q8192. Each also checks 256 independent CPU original full dots.
Every warmup and timed attempt verifies all raw centers/errors against the
scalar control, unique complete candidate membership, raw selected outputs,
unselected output identity and all BF16 results. No reference is a compute
input. These are conditional operator comparisons; they generate no model
tokens and do not measure TTFT.

## Completed owner clocks

All variants run in the same executable on baiying/gfx1151. The table gives
medians of three rotated samples after one warmup, in milliseconds. The
owner includes eligibility, preparation, producer, compaction, completion,
count readback and all selected original replay. The prefix column includes
everything before replay. Allocation, reset and verification are outside
these clocks.

| Producer | OUT owner | OUT prefix | QKV owner | QKV prefix |
| --- | ---: | ---: | ---: | ---: |
| Original scalar | 101.2769 | 42.8969 | 172.2322 | 89.6085 |
| Original vector | 93.6715 | 35.0166 | 151.6550 | 69.8629 |
| Shared 64 by 64 | 100.2504 | 40.0383 | 175.5896 | 92.8571 |
| Shared 128 by 32 | 98.0841 | 39.5346 | 162.2570 | 80.4832 |
| Shared 64 by 64, prefetch | 96.3518 | 37.4682 | 164.7544 | 82.4698 |

All configurations select exactly 3124922 OUT and 8285915 QKV cells,
executing 799980032 and 1060597120 original K16 groups respectively.
The retained QKV producer is a different route with fewer selected cells;
this comparison does not establish a QKV replacement.

Static code metadata reports 94 VGPRs for the vector control and
206/227/214 for the three shared variants, with 17408/21760/17408 bytes of
shared memory and no private segment. This establishes compiled resource
requirements only. Actual occupancy, power or the cause of the elapsed-time
differences was not measured.

## Evidence and decision

Build, safety, OUT and QKV guarded processes all complete successfully on
baiying. Source and input hashes, complete reports, command-file text,
executable provenance, all samples and host checks are in
[the structured proof](../benchmarks/correctness/shared-coarse-projection-native-components-20260917.json),
230568 bytes, SHA256
`063af502026857db91eb495dc44b5d4d7d209cb3f43c045e6077abf4bde860bc`.
The executable is 487936 bytes, SHA256
`2abd2630ff2a4a4de36b4ac38361c2c344ea4517952cc2a163875dd0e81c5444`.
The proof includes the pinned `run-shared-coarse-projection-r1.ps1`
commands and the model reference `D:\models\Qwen3.6-35B-A3B`.

Local validation covers the C ABI smoke and repository hygiene. Actual HIP
compilation and all numerical work occur on baiying; the full Rust/Python
suite was not rerun for these isolated files. Investigate the repeated
error-recurrence and canonical-arithmetic checks as broader common costs,
with explicit domain proofs before removing any guard. Further tile tuning
is not supported by these complete comparisons. No inference, retained
performance or release acceptance follows from this experiment.
