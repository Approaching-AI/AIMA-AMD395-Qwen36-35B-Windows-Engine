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
preparation is measured and charged separately. Native validation is pending;
runtime dispatch and the 23353.80795 ms qualified model control are unchanged.
