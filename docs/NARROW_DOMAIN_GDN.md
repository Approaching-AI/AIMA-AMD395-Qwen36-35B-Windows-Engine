# Exact narrow-domain GDN matrix experiment

This isolated candidate applies the proved signed-zero / BF16 exponent95:159
domain from [exact QK](NARROW_DOMAIN_QK.md) to the complete GDN score, W/U,
recurrent-state and output sequence. Its K64/K128 dots are covered by the
same K256 carry bound. Every original K16 group remains in order, with its
original modulo sum and truncation. No score, probability or state work is
approximated or omitted.

The retained packed-BF16 row flags gain a second bit for the narrow domain.
They are rebuilt from actual operands, including each newly rounded state
checkpoint and each residual. Bitwise intersection admits a dot only when
both complete operands qualify. Admitted dots use a normal FP32 carry with
no per-group exceptional branch. Other dots use the original scalar routine
and its original broad eligibility and integer fallback. This preserves
extended carries outside the proved domain.

The candidate keeps the retained paired-score ownership, shared-arena
lifetimes, U=V alias rules, gates, BF16 conversions and final FMAs. It adds no
device allocation. Runtime dispatch, provider defaults and packages are
unchanged. The qualified q8192 model baseline remains 23353.80795 ms.

The native fixture compares three routes: retained paired scores and shared
arenas; narrow W/U, state and output with original paired scores; and narrow
arithmetic on all four surfaces. Both independent U and production U=V
ownership are checked. Eight boundary shapes and seven data families include
zero signs, unsupported operands, strong decay, sparse groups, nonfinite
values, exact domain endpoints and nearby excluded exponents. Independent
original CPU dots and every score/output/state/checkpoint/intermediate bit
remain the comparison boundary. The prior all-encoding and K16 host proof
is reused unchanged; new native validation is pending.

Captured tests use the original GB10 q7169 GDN inputs and outputs. The q8192
extension repeats 1024 rows after 7168 original rows; those original rows
retain their external output/W/U/residual/checkpoint boundary. All extended
rows use original arithmetic checks without a new GB10 capture. One warmup
and three rotated completed-host samples cover the full score/WU/state/output
sequence in the retained 1024-token segments. Allocation, reset, transfers
and independent validation stay outside timing. Native component results do
not establish model TTFT, continuation or release acceptance.
