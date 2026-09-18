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
unchanged output on rejection and wave source-rank selection. Native build,
safety and timing results are pending.
