# Compact original queries for long attention

The September20 wholeb3/CK370 original256k run completes the owner and first
512-output suffix, then fails token correctness at output181 (4222 instead
of264). Both first-token/logit boundaries, prefix restoration and final host
guards pass. The [failed case and diagnostic follow-up](../benchmarks/correctness/resident-ordered-prefix256k-token-divergence-20260920.json)
do not yet identify whether compact Q or another numerical surface is involved.
The earlier qualified component and shorter model cases below retain their
original scopes; full256k and release remain unqualified.

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

The fixture now passes all 24 generated cases on baiying's gfx1151 in
61390.119 ms, followed by the original GB10 capture in 4429.936 ms. All
4194304 captured BF16 context endpoints match, along with both query views,
complete intermediate surfaces, exact fallback and guards. Host and cleanup
checks pass. These are component checks; original-token model regressions,
measured memory benefit and retained performance remain open.

The first Windows compile failed because a nested include replaced the outer
`main` macro. Source `37014953c28e5b5a91573c6233fdda00bb279b85` uses an explicit
fixture entrypoint guard. Only two fixture inputs change among the 73 recorded
inputs; CK runtime inputs remain identical to `ca4610f`. The corrected fixture
build completes in 18306.157 ms. The CK DLL then builds in 49974.954 ms with
retained compiler arguments and all 5350 external CK sources unchanged. Its
2106368 bytes have SHA256
`2ab47f4443fa8bb30b521c75b1c6785e8ce1a998b97021bd252aaed77c3aa64b`.
The [native evidence](../benchmarks/correctness/compact-suffix-query-native-20260919.json)
preserves the failed compile, correction, complete source inventory and all
four completed native runs. It does not qualify a model or release.

The same DLL then passes the original q8192/out512 model request with the
unchanged wholeb35/FLA1d/MoE923 stack: all 512 original IDs and callbacks match,
first logit is 10.375 with zero error, and source and cleanup checks pass.
Load is 21426.4429 ms, TTFT 23360.6183 ms and TPOT 101.208059 ms. The
[q8192 regression](../benchmarks/correctness/compact-query-native-q8192-20260919.json)
is a single functional observation; cold8192 does not activate compact suffix
Q and this result does not replace the retained performance boundary. The
original cold32k/out512 activation check also passes: all 512 original IDs and
callbacks match, first logit is 24.75 with zero error, all four chunks complete,
and all 30 compact-Q records report zero Q staging and four K/V copies. Source
and cleanup checks pass. Load is 21336.5387 ms, TTFT 257541.4421 ms and TPOT
205.329628 ms. See the [cold32k result](../benchmarks/correctness/compact-query-cold32k-out512-20260919.json).

The separate wholeb35/CKea6 control records minimum available physical memory
21916213248 bytes; compact Q records 21889257472 bytes, a difference of
-26955776. Peak process private commit differs by -458752 bytes. The
[system-memory samples](../benchmarks/correctness/compact-query-cold32k-memory-kv-20260919.json)
do not show a physical-memory improvement or qualify full256k capacity. System
available memory, driver residency and per-process private commit are different
quantities; these single-run samples do not isolate the GPU staging allocation.

[Preparation evidence](../benchmarks/correctness/compact-suffix-query-preparation-20260919.json)
pins source `ca4610f`, all 71 local compilation inputs, the two build/guard
scripts, bounded native commands, and the failed and passing local checks.
Baiying's PowerShell parser accepts the exact command file at
2026-09-19T07:33:43Z with zero errors; it did not execute the commands or load
the model. The native runs above started after the original 256k run's completed
cleanup; that earlier run used the pinned wholeb35/CKea6 binaries.
