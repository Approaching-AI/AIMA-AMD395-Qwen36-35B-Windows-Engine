# Narrow-domain QK with checked K16 lookahead

This isolated experiment combines the exact byte-exponent metadata route
with two or four K16 groups prepared before their actual carries are ready.
Native BF16 WMMA computes only a predictor. Each original product is scaled,
converted and added to the original unsigned modulo sum. An eight-byte plan
stores that sum and its alignment metadata.

Before consuming a plan, the original ordered carry supplies the actual
alignment. The plan is accepted only when that alignment matches exactly;
otherwise the complete score is recomputed by the existing original dot
routine. Predicted values never become output carries. The original narrow
K256 domain proof and byte-metadata span admission remain prerequisites.
Other tiles use the unchanged general kernel and deferred original replay.

Unlike the earlier generic predicted-group routes, this version streams
products directly into plans using exact packed exponent maxima, retains
normal float carries under the existing proof, and computes four scores per
thread. A 32x32 tile stages K32 or K64 operands and native partials in shared
memory. Plans remain in thread registers; no global plan table is allocated.
The byte metadata owner is reused, plus one four-byte rejection counter.
Earlier unsuccessful prediction experiments remain relevant performance
priors, but their measurements do not qualify this implementation.

The independent host check passes ASan and UBSan: 3145728 predictions across
524288 original K16 groups exercise 1023402 accepted and 2122326 rejected
plans. All accepted carries match original arithmetic bit for bit. Another
524288 deliberately wrong alignments are rejected without changing output.
The initial fixture compile missed a standard header; that failed run is
preserved separately from the corrected passing fixture.

The native fixture compares the qualified narrow 2x4 callback, byte-exponent
4x4 building block, and two/four-group lookahead over complete attention.
It also forces every admitted score to reject its plan in generated safety
cases and checks both the rejection count and original output. There are
400 generated configurations across eight shapes and ten data families.
Metadata, raw scores, native surfaces before PV replay, complete outputs,
candidate identity, guards and immutable inputs are checked independently.

Captured q8192 repeats 1023 rows from the original q7169 capture. Every
original GB10 context cell remains checked; the extension is compared to
original arithmetic, without claiming an external oracle for repeated rows.
Timing includes complete QK, fused probability/native PV, collection and
original exact PV. Route preparation is charged separately and common
decoded preparation/V transpose is reported. One warmup and three rotated
measurements are used per slab.

Source `774e5e82b8f4077f4aa3e9bb52a6a61bc8c31823` passes the Windows build,
all 400 generated configurations and the complete q8192 comparison on
baiying. Forced mismatches replay 178502656 scores with zero output
differences. Both normal lookahead routes reject only 1679 captured scores.
All 545259520 score slots, 33554432 output cells, 29364224 original GB10
context cells and 3127598 PV candidates match; CPU metadata, independent
original dots, guards and immutable inputs pass on every required surface.

| Route | Attention median ms | Preparation ms | Combined ms | VGPRs | Shared bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained narrow 2x4 | 542.6030 | 1.2672 | 543.8702 | 167 | 24576 |
| Byte exponents 4x4 | 519.0002 | 0.6110 | 519.6112 | 140 | 26880 |
| Two-group lookahead | 597.2289 | 0.6110 | 597.8399 | 95 | 14976 |
| Four-group lookahead | 636.7522 | 0.6110 | 637.3632 | 113 | 29824 |

Common decoded preparation and V transpose add 3.8008 ms to every route.
All kernels declare zero private bytes and spills. Static resource counts
do not measure occupancy or explain the regression. Both lookahead schedules
remain outside runtime dispatch. The same-run byte layout remains a modest
isolated improvement; this experiment adds no model performance evidence.

[Commands, all source/binary hashes, host and native records](../benchmarks/correctness/narrow-qk-lookahead-native-components-20260918.json):
133949 bytes, SHA256
`29e8948fda7f7558b6c22dfad8222c6f5945ba2af9399cebd1316714d9776013`.
The command file is `run-native-narrow-qk-lookahead-r1.ps1`; all 279 compiler
inputs match the source commit. Preserve the 23353.80795 ms qualified model
control. Runtime dispatch, packaging and release acceptance remain unchanged.
