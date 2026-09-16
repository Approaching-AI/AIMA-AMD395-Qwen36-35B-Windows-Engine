# C64 finite-domain recurrence specialization

Source `b99fb0079f0057c3077c2e3b24696807632a4165` removes unreachable
exceptional branches from the existing C64 error recurrence while retaining
its exact raw center and error values. All tested results pass. Complete
captured OUT improves from 93.8090 to 87.3077 ms and C64 QKV from 150.4241
to 136.0386 ms against their same-executable vector controls. Keep this as
an isolated building block: it does not establish a seconds-scale model
gain, and existing producer defaults remain checked.

## Arithmetic domain

The existing eligibility pass admits zeros and biased BF16 exponents
80 through 174, and zeroes unsupported operand rows in the matrix producer.
Every admitted operand has magnitude below 2^48. With width at most 8192,
a deliberately loose induction bounds each signed/absolute C64 partial
below 2^103, running centers below 2^111 and cumulative errors below 2^104.
Every intermediate magnitude remains below 2^113. Per-block error increments
remain below 2^96; at most 128 blocks preserve the stated error bound.
These margins include floating rounding and outward representable steps.

Consequently, every original `upper` argument is nonnegative, finite and
far below overflow; every `unit` argument is strictly positive. The new
recurrence removes those repeated exceptional checks. It retains the
original floating expressions, two-ULP outward steps, exact subnormal-unit
construction and native coefficient 2^-19. Widths outside the supported
domain use the original recurrence. This domain argument does not prove
the underlying conditional native WMMA error model.

Host ASan/UBSan checks compare 2097152 random states, 647680 ordered edge
states, 5300 unit cases and 2651 outward-step cases against the original
functions with zero raw mismatches. The existing coarse-bound suite also
passes. Native compilation and all execution on baiying pass, including
2097152 direct GPU recurrence comparisons in each process. The repeated
count in each report describes the same process-level check, not additional
independent tests.

The native operator suite has 19 generated shapes and five configurations:
155395 outputs, 31079 distinct CPU original dots and 27755 selected raw
endpoints. Explicit exponent-80/174 inputs, mixed extremes, cancellation,
zeros, unsupported values, tails, independent buffer skews and width-8208
fallback pass. All prepared words, flags, immutable spans, redzones and
unused candidate tails are checked on every attempt.

## Complete captured operators

One warmup precedes three rotated samples in the same executable. Values
are completed host medians in milliseconds. Preparation, eligibility,
production, selection, count readback and complete original selected replay
are included; reset, allocation and validation are outside the clock.

| Producer | OUT owner | OUT prefix | QKV owner | QKV prefix |
| --- | ---: | ---: | ---: | ---: |
| Checked scalar | 101.6865 | 42.8993 | 167.2052 | 86.5096 |
| Checked vector | 93.8090 | 34.8408 | 150.4241 | 69.7101 |
| Domain vector | 90.0849 | 31.3408 | 136.0386 | 55.2787 |
| Domain shared 64 by 64, prefetch | 87.3077 | 26.9037 | 138.5672 | 57.5300 |
| Domain shared 128 by 32 | 87.6504 | 28.4842 | 139.1461 | 58.3036 |

All raw centers and errors are identical to the checked scalar control.
All configurations retain exactly 3124922 OUT and 8285915 QKV candidates,
the same complete candidate masks and original selected raw outputs. Every
16777216 OUT and 67108864 QKV BF16 output matches its GB10 operator reference
in every warmup and timed attempt. Each capture also checks 256 independent
CPU full dots. As in the prior experiment, these shapes repeat 1023 of the
7169 captured input rows; QKV repeats the matching reference rows. They are
conditional operator checks, with no model load or generated tokens.

Both checked and domain vector kernels declare 94 VGPRs. The two domain
shared kernels declare 213/226 VGPRs and 17408/21760 bytes of shared memory;
all declare zero private segment. Static resource counts do not establish
occupancy or explain elapsed-time differences. The retained QKV producer
uses a different matrix route with fewer selected cells, so these results
do not qualify a QKV replacement.

## Provenance and decision

Build, safety and both captures complete with all baiying host guards.
[The structured proof](../benchmarks/correctness/domain-coarse-projection-native-components-20260917.json)
contains source/input hashes, local validation, complete native reports,
command-file text and binary provenance: 269164 bytes, SHA256
`cd2ed35055ac0db536321f04835b2c97f667ee667c95d6963558f5c666edca9a`.
The executable is 495616 bytes, SHA256
`29c08eccf3085c6e2d3694171ca8018e3e31bfedc0bf1f55a7db793656c5148d`.
Commands are pinned in `run-domain-coarse-projection-r1.ps1`, with model
reference `D:\models\Qwen3.6-35B-A3B`.

The host bound suites, C smoke and repository hygiene pass. The full
Rust/Python suite was not rerun for this isolated specialization. Retain the
verified domain reduction as a building block while pursuing larger
arithmetic/dataflow changes. No runtime default, package, inference,
retained-performance or release acceptance changes.
