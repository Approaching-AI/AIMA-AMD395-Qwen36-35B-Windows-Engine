# Exact AVX-512 QK partition experiment

Sources `5986f7f` and `6599691` test whether baiying's CPU can take over
canonical attention score production. The experiment remains isolated;
runtime dispatch, package assets and retained settings are unchanged.

Each SIMD lane owns an independent K256 dot. Exact FP32 products, original
common exponents, per-product truncation, unsigned modulo reduction and RZ
normalization preserve every ordered K16 carry. Unsupported operands or
carried endpoints restart the complete original wide dot. AVX512F/CD and
OS ZMM state are checked before execution. Persistent workers establish
round-to-nearest and gradual underflow in their local floating environment.
The implementation uses the installed HIP Clang and standard C++ threads,
with no added library dependency.

Both revisions pass 30 native numerical cases: 1732160 score comparisons,
409871 complete fallback scores, 115176 accepted raw K16 states and 48664
rejected trace slots. Cases cover zeros, cancellation, modulo-sign overlap,
extreme exponents, unsupported words and causal tails. Single/16-worker
results agree; injected worker failure, subsequent ownership, empty work,
output guards and immutable inputs pass. The preparation tests cover every
BF16 encoding, row flags and complete padding. The revision checks both
layouts in ten shapes. Actual 32-worker arithmetic is additionally checked
throughout both complete captured shapes.

## Captured attention and accounting

Each executable compares a current GPU prepared-QK control with two CPU
variants. The CPU inputs are read back from the actual run's device Q/K
tensors. CPU preparation decodes keys losslessly; every produced score slab
is uploaded before the unchanged original probability, native PV and compact
exact-PV stages. No CPU/GPU overlap is implemented or measured.

An independent original tiled-QK pipeline supplies observer-only raw
comparisons. Every warmup and timed attempt preserves scores, probabilities,
scales, output, accumulators, denominators, error bounds and original PV
candidate counts. Guarded buffers, unused tails, transposes, complete prepared
encodings and immutable sources pass. Candidate index ordering is not a
bytewise invariant. All 29364224 available external GB10 context cells match.
q8192 repeats 1023 captured Q/K/V rows, comparing all 33554432 output cells
to original arithmetic; the extension has no additional external golden.
Neither external contexts nor control scores enter candidate computation.

The initial zero-padding experiment gives these completed attention medians:

| Shape | GPU ms | CPU 16 workers ms | CPU 32 workers ms |
| --- | ---: | ---: | ---: |
| q7169 | 710.5495 | 2316.2956 | 2099.0307 |
| q8192 extension | 1505.0898 | 15787.5933 | 11705.0276 |

The disproportionate q8192 cost prompted a paired layout comparison. Source
`6599691` adds 16 floats of padding between key feature rows in each of the
value and exponent arrays, avoiding the 32768-byte feature stride. It adds
65536 bytes in total. The same executable and 32-worker pool compare both
layouts, with unchanged arithmetic and task ownership:

| Shape | GPU complete ms | Unpadded CPU complete ms | Padded CPU complete ms | Unpadded CPU QK ms | Padded CPU QK ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| q7169 | 711.9242 | 2292.3106 | 2115.1136 | 1379.5141 | 1248.5778 |
| q8192 extension | 982.5672 | 6124.7819 | 3606.5788 | 4278.4413 | 2043.3010 |

Padding materially improves the observed q8192 CPU cost. Cache-set conflict
is a plausible explanation for the stride sensitivity; no cache-event or
power counters were collected to establish its hardware cause. Separate
executables have different absolute clocks, so only their paired arms
establish the quoted layout comparison.

Each table entry is the median of three completed full-shape samples after
one warmup per 128-query slab, with rotated variant order. CPU score and
blocking upload intervals are nested in complete attention. At q8192, padded
uploads total 234.7098–238.1412 ms per sample. Input readback, CPU encoding,
worker startup and common V transpose add 36.4943 ms; the padded complete
total becomes 3643.0731 ms. The corresponding GPU total with its own
preparation is 992.7759 ms. Padded CPU encoding occupies 33767424 bytes.
Allocation, fixture reset and verification remain outside these clocks.

Both native builds, both safety runs and all four captures finish normally
with host guards. Local sanitized layout tests, x86 cross-syntax checks,
C ABI smoke and hygiene pass for both revisions. The Mac checks do not
execute AVX-512; numerical SIMD evidence comes from baiying. No new complete
Rust/Python suite run is claimed for these isolated components.

## Decision

Keep the exact CPU implementation and padded representation isolated.
Even the corrected CPU score producer alone exceeds the complete GPU
attention clock here. There is no case for replacing all GPU QK with this
CPU producer. Partial coexecution remains unmeasured; this experiment does
not establish a universal limit on CPU/GPU partitioning or a pipeline speedup.
Continue a broader GPU arithmetic or dataflow route against the measured
product wall. No new model token loop, TTFT, prefix, retained performance,
package or release acceptance follows. All mission thresholds remain fixed.

[Complete sources, commands, safety reports and captured GB10 comparisons](../benchmarks/correctness/cpu-exact-qk-attention-components-20260917.json),
256266 bytes, SHA256
`1cc5f2ef087e1685a284ce07527dddad162e4dbe2c358b9411dfbb4281febfd7`.
