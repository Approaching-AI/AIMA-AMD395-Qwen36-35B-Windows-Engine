# Compact matrix QK group experiment

This isolated producer tests whether integer matrices can replace a useful
fraction of the original ordered K16 QK arithmetic. It is not wired into a
runtime provider or a release package.

The host capture audit found 714,728 of 1,048,576 sampled K16 groups eligible
for exact integer alignment, but only 402 of 65,536 complete dots eligible in
every group. A mixed implementation therefore needs inexpensive original
groups and carry transfer; whole-dot admission is insufficient.

Each 60-byte prepared row retains the original BF16 words through the existing
52-byte compact representation plus eight bytes of trailing-zero metadata.
Unsupported rows retain their original words. Four IU8 matrix operations give
an exact signed16 dot. The exact paired-product maximum and a conservative
divisibility check certify that its aligned sum equals the original group.
Rejected groups leave the carry unchanged and run the original narrow-domain
arithmetic. Complete original rows must satisfy the existing signed-zero or
BF16 exponent 95..159 domain; this QK implementation has K256.

A 256-thread block owns 32 queries by 32 keys and stages two K16 groups at a
time. Four FP32 carries stay with each output thread. The two experimental
variants either compute rejected groups directly or assign them to consecutive
wave lanes using ballots and register shuffles. Carry values are returned to
their original owners before the next K16 group. A safety-only variant forces
every group through the latter path. Shared operands, reconstructed original
words and matrix products occupy 28,160 bytes. Actual register and private
memory usage require native compiler evidence.

The complete attention fixture compares against the retained 2x4 narrow QK
callback with unchanged fused probability/native PV and original PV replay.
It checks all output surfaces, metadata, inactive regions and input guards.
The native safety plan has eight shapes and ten data families across four
variants, including forced original work. Captured q7169 uses the original
GB10 context; q8192 repeats 1,023 input rows and checks the original captured
context only where available. All other q8192 cells are compared with the
unchanged original arithmetic. Candidate timings include complete attention;
the additional metadata preparation is reported separately and must be added
when comparing totals. These are component measurements, not model TTFT.

Host ASan/UBSan checks cover lossless reconstruction, ordered carry equality,
unchanged output on rejection and wave source-rank selection. Native build and all 320 safety configurations pass on baiying, including
the forced-original queue. Both captures preserve all original score and
attention surfaces, candidate identities, immutable inputs and 29,364,224
original GB10 context values per variant.

| Complete attention plus route preparation | Retained | Direct fallback | Wave fallback |
| --- | ---: | ---: | ---: |
| q7169 | 382.2061 ms | 796.6028 ms | 709.9250 ms |
| q8192 | 540.3715 ms | 1102.1330 ms | 989.6595 ms |

Every candidate timing sample is slower than every control sample. The common
decoded preparation and V transpose, excluded equally from these totals, cost
2.4259/3.7167 ms. Static control/direct/wave/forced resources use
167/71/73/57 VGPRs and 24576/28160/28160/28160 shared bytes with zero private
scratch or spills. Reduced registers do not establish a throughput improvement.
The additional q8192 row metadata occupies 141,557,760 bytes.

Keep this source isolated and continue broader scheduling or arithmetic work.
The qualified model control remains 23353.80795 ms; no provider, package or
release change follows. Native source `7c9628c`, guarded command
`run-native-compact-matrix-queue-qk-r1.ps1` and all run/executable/input hashes
are pinned in the [complete evidence](../benchmarks/correctness/compact-matrix-queue-qk-native-components-20260918.json):
146505 bytes, SHA256
`def3391fc2fb17070fe46ead130a1aa41273c6fda43f63a6934c019b4cfe720a`.
