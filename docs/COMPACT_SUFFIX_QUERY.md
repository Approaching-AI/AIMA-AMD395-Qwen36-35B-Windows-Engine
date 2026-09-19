# Compact original queries for long attention

The long suffix path previously reserved Q storage for every historical token,
although it reads only the new query rows. A 262144-token prefix plus a 1024-row
suffix reserves 2155872256 bytes for Q in that staging slab. This candidate
passes the caller's compact Q directly and stages only complete K/V. It removes
that Q allocation and its suffix copy. This is an allocation-size calculation,
not a measured reduction in system memory or model latency.

An explicit query origin accompanies the existing long QK workspace. Original
BF16 query preparation, complete-row admission and original exact fallback all
subtract that origin. Packed query metadata retains its existing logical start;
key addresses, causal positions, score layout and the ordered dot arithmetic
stay unchanged. Origin-zero callers retain their existing entry points.
The compact view is accepted only by the long pipeline. One-token and ordinary
paths continue to use the ordinary staging owner. Both owners are protected by
the existing suffix mutex, and submitted work drains before the caller can
release its Q or the provider can release K/V.

Four local tests pass. Actual provider and suffix functions check both owners,
mode changes, capacity bounds, source identity, four complete K/V copies,
allocation failures and partial-submission drains. Query preparation and long
dispatch verify origins, metadata offsets and failed launches. The actual
fallback function matches 400 independent original integer dots at positions
17, 8192, 131072 and 262144, including subnormal operands, interior query spans,
all query/KV heads, signed cancellation, unchanged nonmarkers and output tails.
The numerical and preparation checks run under ASan/UBSan; suffix staging uses
UBSan. A missing constant in the first host fixture is preserved as a failed
compile and was corrected before the passing execution.

The native fixture adds 24 generated cases and original GB10 16384-prefix plus
1024-suffix replay. Large positions reach the full 264736-token capacity, while
small, partial and multi-slab query spans cover fallback and storage bounds.
Unused full-history Q is poisoned. The fixture compares every score, probability,
scale, original PV endpoint, selected candidate and all five completed pipeline
stages, including a nonzero output offset. It independently reconstructs query
metadata and periodic CPU dots. An optional 8192-query case repeats captured
rows and is explicitly a component extension rather than new model evidence.

Windows compilation, numerical execution and original-token model regressions
remain pending. The active 256k model run uses earlier pinned binaries. This
candidate makes no native correctness, memory-benefit or release claim.
