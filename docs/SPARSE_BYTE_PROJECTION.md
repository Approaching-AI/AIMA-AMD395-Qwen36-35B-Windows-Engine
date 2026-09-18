# Sparse projection tiles with exact byte metadata

This isolated route combines input reuse with the exact byte-exponent
arithmetic. The original selected output identities form a bitmap. Each
32x32 or 64x64 tile collects at most 256 identities into shared storage.
One thread owns one selected dot, while original BF16 operands and exact
twenty-byte K16 metadata are shared across a K64 window. Products feed the
original aligned unsigned sum immediately. Ordered K16 normalization stays
unchanged. This differs from the older cooperative half route, which retained
four lanes per dot and their per-group exceptional checks and reductions.

Fast admission requires every original operand in both complete rows to be
signed zero or exponent95:159, with every K16 nonzero span at most31. For
widths through8192, the triangle bound on original products is below2^79.
Each alignment and normalization truncates magnitude; the smallest nonzero
aligned quantum remains2^-89. Thus carries remain normal and finite. The
existing exact byte arithmetic can use the same scale and normalization
without per-group exceptional checks. Dense tiles with more than256
candidates and any rejected row use the qualified staged2 four-lane complete
original dot. No candidate or numerical envelope is removed.

ASan/UBSan host validation compares468480 original ordered K16 groups over
3072 dots at K16/32/256/2048/4096/8192. All raw carries,936960 metadata records
and asserted carry ranges pass. Tests include signed zeros, cancellation,
domain extremes and mixed exponent distributions.

The native suite adds72 generated shapes/data/selection combinations with
three routes, reversed queues, empty and dense candidates, tails, subnormal
inputs and span32 rejection. It compares every raw output against independent
CPU original arithmetic, with metadata, candidate masks, all prepared words
and guards checked. Captured QKV and OUT retain the original current midpoint
selector, PPB1000/10000, QKV matrix4 and OUT matrix0. Full original GB10 BF16
outputs and all unrounded selected values are checked after each attempt.

Timing includes original half preparation, candidate metadata preparation,
bitmap reset/scatter and complete replay. Allocation, fixture reset, common
matrix production and selection, transfer and validation are outside. One
warmup precedes three rotated samples. Native validation is pending; runtime
dispatch, model baseline and packaging remain unchanged.
