# Globally split exact attention probabilities

Source `7ff6cd15a40e3a3f76cf7130cadbc769972e7732` separates tile maxima,
prefix maxima, probability tiles and the original denominator fold into
four GPU dispatches. All tested raw attention values match. Complete
captured q8192 attention takes 796.8697 ms, versus 786.9070 ms for the
existing staged control and 800.9081 ms for the online control. Keep this
variant isolated; the extra parallelism establishes no useful speedup.

## Arithmetic and checks

The prefix scan only reassociates the associative maximum. Independent
K32 waves preserve the original EXP lookup, BF16 probabilities and butterfly
sum order. A final thread per row reads tile-major alpha and sum arrays and
executes the original ascending FP32 denominator recurrence. The probability
producer writes the original row-major alpha surface for unchanged PV.
Neither sums nor the denominator recurrence are reassociated.

The native suite covers 240 combinations of ten shapes, six score families,
two butterfly orders and table/native EXP. All 433191552 probability slots,
13696512 scale slots and denominators match bitwise. Tests include causal
tails, signed zero, subnormal scores, repeated maxima, large changes in
maxima and 2375424 zero-alpha observations. Workspace guards, alpha copies,
constant-score CPU checks, immutable inputs and invalid shape/pointer/
capacity rejection pass. These finite-score cases do not establish behavior
for arbitrary unmasked NaN or infinite scores.

## Complete captured attention

The same executable runs online, existing staged-CTA and globally split
probabilities with unchanged original QK and native plus exact selected PV.
Each 128-query slab has one warmup and three rotated measured attempts.
Every attempt is verified against original tiled QK; the external GB10
context is an observer only. Completed host medians include every dispatch
and selected PV replay. Allocation, reset and verification are outside the
clock; common input preparation is reported separately.

| Complete attention, ms | Online | Staged CTA | Split global |
| --- | ---: | ---: | ---: |
| q7169 | 581.4028 | 573.8247 | 575.6160 |
| q8192 extension | 800.9081 | 786.9070 | 796.8697 |
| q8192 including preparation | 810.8241 | 796.8230 | 806.7857 |
| q8192 probability GPU interval | 125.3133 | 121.9949 | 129.2359 |

All raw scores, probabilities, scales, context, accumulators, denominators
and error bounds match; PV candidate counts remain 2198673 / 3127598 for
q7169 / q8192. Every variant matches all 29364224 available GB10 context
cells. The extended q8192 shape repeats 1023 captured Q/K/V rows and compares
all 33554432 output cells to original arithmetic; it supplies no new model
tokens or TTFT measurement. Each shape also checks 228 / 256 independent
CPU dots, workspace/causal tails, guards and immutable inputs.

The split workspace occupies 5529600 / 6291456 bytes at the two shapes.
Its four kernels declare 8 / 14 / 20 / 10 VGPRs, with 32 bytes of shared
memory only for the prefix scan and zero private segments. All measured
probability intervals are nonnegative. These static resource counts and
elapsed times do not establish occupancy or a memory-bandwidth cause.

## Provenance and decision

Build, safety and both captures finish on baiying with all host guards.
The command file is `run-split-probability-r1.ps1`; the capture's model
reference is `D:\models\Qwen3.6-35B-A3B`. The executable is 1653248 bytes,
SHA256 `9ff171d3929e70ab0243f2397edcf88d6ba59ebb05d4545576fec59514a7fccf`.
[The structured proof](../benchmarks/correctness/split-probability-attention-components-20260917.json)
contains complete reports, input/source hashes, command text and provenance:
302589 bytes, SHA256
`3385990f0e58d6c8452b191e85e7b02b70c783ac2c4d18566921691037b51a07`.

C smoke and repository hygiene pass; the full Rust/Python suite was not
rerun for this isolated component. No runtime default, package, retained
performance or release qualification changes. Further work should address
the larger arithmetic and correction costs using the existing PV bounds
and coarse-transfer results as evidence.
