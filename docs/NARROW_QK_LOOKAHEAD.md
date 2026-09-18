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

Native validation is pending. Runtime dispatch and packaging are unchanged;
this is not model inference, model TTFT or release acceptance.
