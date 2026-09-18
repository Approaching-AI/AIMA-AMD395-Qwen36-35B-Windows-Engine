# Exact QK with byte-packed exponents and immediate product consumption

This isolated route prepares twenty bytes of exponent metadata per K16
operand row. Each nonzero operand's deficit from the row maximum is at most
31; signed zeros use 63. Four deficits occupy each dword. Adding two packed
rows cannot carry between bytes because every sum is at most 126.

Packed byte minima obtain the exact paired-product exponent. A minimum below
63 belongs to two nonzero operands; a minimum of 63 or above proves that
every product is zero. The per-byte subtraction sets a local high bit first,
preventing cross-byte borrow. No approximate maximum or observed reference
value enters the arithmetic.

The complete tile must pass the existing signed-zero / BF16 exponent95:159
domain and the per-K16 span check. The existing K256 proof then guarantees
normal finite carries. Each K16 scale is known before forming products,
which are immediately converted and added to the original unsigned modulo
sum. Original ordered K16 normalization, sign disambiguation and final score
scale remain. Other tiles use the unchanged general kernel and complete-dot
fallback. The control is the retained 2x4 narrow-domain provider callback.

This differs from the earlier packed-exponent experiment, which retained
all sixteen products and used a conditional five-bit maximum summary. The
new route uses full admitted byte deficits, immediate product consumption,
packed original BF16 shared operands and 2x4 or 4x4 output ownership. It also
avoids the earlier two-pass route's rereading of every product operand.
The earlier results remain performance priors, not evidence for this route.

Independent ASan/UBSan host checks pass 1064705 packed minimum comparisons,
all 65536 BF16 encodings, 1122374 metadata checks, and 528416 exact maxima
and ordered K16 carries. Span31/32 admission, disjoint nonzero positions,
signed zero tails, cancellation and the narrow-domain extremes are covered.

The native fixture checks 240 configurations, complete original scores and
attention surfaces, metadata reconstructed on the CPU, candidate identity,
guards and immutable inputs. Captured q7169 carries all 29364224 original
GB10 context values; q8192 repeats 1023 original input rows with original
arithmetic checks on the extension. Complete timing includes QK, unchanged
fused probability/native PV, collection and original exact PV. Metadata
preparation is measured and charged separately.

Source `5f0fcc1b49845da1758984757cf11aee50e0db6e` passes Windows build,
all 240 native configurations and the complete q8192 comparison on baiying.
Every route matches all 545259520 score slots, 33554432 outputs, 29364224
original GB10 context cells and all 3127598 original PV candidates on every
attempt. Both original and candidate tile branches are exercised. All CPU
metadata, guards and immutable input checks pass.

| Route | Attention median ms | Route preparation ms | Combined ms | VGPRs | Shared bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained narrow 2x4 | 533.4854 | 1.1685 | 534.6539 | 167 | 24576 |
| Byte exponents 2x4 | 522.2103 | 0.6626 | 522.8729 | 140 | 20096 |
| Byte exponents 4x4 | 509.9847 | 0.6626 | 510.6473 | 140 | 26880 |

Common decoded preparation and V transpose add 3.7604 ms to every route.
All three kernels declare zero private bytes and spills; these static
resources do not establish occupancy or explain the timing difference.
Candidate metadata and flags use 47775744 bytes at q8192, alongside the
existing generic decoded owner. The 4x4 layout saves 24.0066 ms in this
component. Keep it as an isolated building block: this modest gain does not
justify a provider/model cycle by itself while TTFT is above 10 seconds.

[Commands, source/binary hashes, host audit and all native results](../benchmarks/correctness/byte-exponent-qk-native-components-20260918.json):
122249 bytes, SHA256
`bf640abe6a4e066efa95650a244530353ae9b75bbda7209ad1141f7399bb608b`.
The command file is `run-native-byte-exponent-qk-r1.ps1`; all 277 compiler
inputs are audited against the source commit. Runtime dispatch, package and
the 23353.80795 ms qualified model control remain unchanged. No model timing,
new continuation result or release acceptance follows.
