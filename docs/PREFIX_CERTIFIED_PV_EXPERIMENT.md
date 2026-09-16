# Prefix-certified exact PV replay

Source `3e41f70997ed7288bce8eab0c8b786f18403614b` reduces q8192 selected
K16 replay work by 31.63% with 512-key rounds and 28.06% with 1024-key
rounds. Every tested BF16 result and available GB10 context cell matches.
Complete PV nevertheless increases from 329.6917 ms to 369.6577 / 336.9665
ms. Keep the implementation isolated; no runtime or package change follows.

## Certificate

The producer retains the original carried native WMMA, K32 alpha rescaling
and final error bound. It additionally saves native centers and deferred
error states. Original selected cells replay canonical K16 arithmetic from
zero in compacted rounds. No approximate checkpoint becomes a canonical
starting accumulator.

Let `E` be the original outward error envelope and `A` the product of all
original alphas after an exact prefix. Unrolling that same envelope bounds
the suffix local-error sum by `E_final - A * E_prefix`. The final canonical
numerator therefore lies around
`native_final + A * (exact_prefix - native_prefix)`.
The certificate uses the existing upper bound on `E_final` and a new lower
bound on `E_prefix`. Nonnegative deferred-state arithmetic, `gamma(4G)` for
at most 512 groups, a conservative relative subtraction and a flushing floor
establish the latter bound. Double intervals enclose the alpha product and
signed center correction. Original reciprocal multiplication and BF16
endpoint comparison determine whether replay may stop.

This argument inherits the existing conditional native PV error model. It
does not assume associative or Lipschitz rounded K16 transitions. Accepted
early values are BF16-correct representatives, without a claim of raw FP32
equality to the complete exact endpoint.

Host ASan/UBSan checks pass 524576 old-envelope sandwiches, 520480 suffix
alpha products, 1040960 signed suffix intervals and 65022 BF16 midpoint
tests. The existing final/error-bound suites also pass. Native testing covers
84 generated cases and three checkpoint widths: 252 configurations,
24920064 output comparisons, 8306688 distinct generated output cells and
672 independent CPU full PV dots. Both butterfly orders, signed zero,
subnormal/large values, abrupt maxima, zero alphas, unsupported NaN values,
128-query batches and the 8192 endpoint are exercised. Unsupported cases
continue complete replay. All checks pass.

## Captured complete PV owners

One warmup precedes three rotated measured samples in the same executable.
The completed host clock includes native production, checkpoint writes,
suffix-weight preparation, collection, counter initialization and every
compacted replay round. Allocation, reset and observation are outside the
clock. QK, original probabilities and V transpose are measured separately.

| Capture / route | PV median, ms | Early certificates | Replayed K16 groups |
| --- | ---: | ---: | ---: |
| q7169 original | 210.9568 | 0 | 637019466 |
| q7169, 512 keys | 217.6138 | 1785735 | 456024726 |
| q7169, 1024 keys | 206.1540 | 1548080 | 481635226 |
| q8192 original | 329.6917 | 0 | 1083940700 |
| q8192, 512 keys | 369.6577 | 2638110 | 741045454 |
| q8192, 1024 keys | 336.9665 | 2357587 | 779760180 |

Initial candidate membership remains exactly 2198673 / 3127598 for
q7169 / q8192. Every attempt checks original native centers and error bits,
complete original candidate membership, raw endpoints of complete replays,
actual exact-numerator coverage of early intervals, queue conservation,
immutable inputs, redzones and unused tails. QK also matches the independent
original tiled producer bitwise. All variants match 29364224 original GB10
context cells; the q8192 extension repeats the first 1023 captured Q/K/V rows
and compares all 33554432 outputs to original arithmetic. Capture observers
add 228 / 256 independent CPU PV dots. References never enter computation.

At q8192 the workspaces occupy 86507588 / 52690980 bytes, including result
observers. The original and checkpointed producers declare 102 / 126 VGPRs
and 16 / 12 private bytes. Original and queued exact replays declare 71 / 72
VGPRs and 12 private bytes each. These static declarations do not establish
occupancy or attribute the measured regression to a particular operation.

## Evidence and decision

All baiying build, safety and capture guards pass. Commands are in
`run-prefix-pv-r1.ps1`, with model reference `D:\models\Qwen3.6-35B-A3B`.
[The structured proof](../benchmarks/correctness/prefix-certified-pv-native-components-20260917.json)
contains source/input fingerprints, complete reports, command text, local
validation and executable metadata: 609091 bytes, SHA256
`a4c3116905a1adc23d965e3945770ba7141fde682b459a29497766f7a882e41e`.

C smoke, the three host numerical suites and repository hygiene pass. The
full Rust/Python suite was not rerun. The small q7169 improvement at 1024
keys does not transfer to q8192; further work returns to common arithmetic
costs. No model token loop, TTFT, retained performance or release
qualification is established.
