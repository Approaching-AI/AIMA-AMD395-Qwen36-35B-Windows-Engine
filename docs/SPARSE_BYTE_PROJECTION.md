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
and guards checked. Captured QKV and OUT use the current midpoint predicate,
PPB1000/10000, QKV matrix4 and OUT matrix0. The OUT comparison is the original
midpoint owner, not the newer retained coarse-OUT model owner. Full GB10 BF16
outputs and all unrounded selected values are checked after each attempt.

Timing includes original half preparation, candidate metadata preparation,
bitmap reset/scatter and complete replay. Allocation, fixture reset, common
matrix production and selection, transfer and validation are outside. One
warmup precedes three rotated samples.

Source `8ae7879235feba5ede2d62038eb0e43e409e1ab7` passes the Windows build,
all216 generated configurations and both captured operators. Every generated
raw output matches18744 distinct independent CPU dots, including all original
fallbacks. Eight invalid launches reject before submission. Every captured
attempt preserves67108864 QKV and16777216 OUT GB10 BF16 outputs and all
4331635/8471989 unrounded selected values. Complete row metadata, original
half encodings, candidate membership, inactive output and guard checks pass.

| Shape | Original staged2 ms | Sparse32 ms | Sparse64 ms |
| --- | ---: | ---: | ---: |
| q8192 QKV, K2048 | 42.4972 | 115.753 | 256.317 |
| q8192 midpoint OUT, K4096 | 156.823 | 272.651 | 325.829 |

The QKV candidates execute3404873/2470133 fast dots, with the remainder
using original arithmetic. Every OUT tile exceeds the sparse capacity, so
all8471989 selected values use original fallback; its timing measures the
additional owner overhead. The common replacement already regresses QKV,
so no current coarse-OUT or real-model integration follows.

Extra metadata, flags, bitmap and counters require50397200 bytes for QKV or
54566928 bytes for OUT, alongside the original prepared operands and queue.
Original/Sparse32/Sparse64 kernels declare49/72/71 VGPRs and0/14084/27652
shared bytes, with zero private bytes and spills. These static declarations
do not measure occupancy or establish a cause for the regression.

[Commands, native results, host checks and source/binary hashes](../benchmarks/correctness/sparse-byte-projection-native-components-20260918.json):
435801 bytes, SHA256
`4662ab9cf7d9125434c576d4a369bb6c4bcc661c3142183fc884fb3bb5b11f68`.
All717 repository source-inventory entries match the commit; this inventory
also includes files not compiled by this fixture. The native command file is
`run-native-sparse-byte-projection-r1.ps1`. Both routes remain isolated and
the23353.80795 ms qualified model control, package and release state remain.
