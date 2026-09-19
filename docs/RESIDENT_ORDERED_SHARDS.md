# Shared resident fixed weights

The current stack retains the original 71,903,776,776-byte model shard store
and a separate 3,879,600,640-byte fixed-weight arena. Simply disabling that
arena makes the combined QKVZ+A/B consumer allocate another 1,416,626,176
bytes. The existing text-only storage experiment removes optional vision and
MTP tensors but leaves this fixed-weight duplication in place.

The default-off `QRT_PREFILL_DESCRIPTOR_BATCH_RESIDENT_MODEL_ORDERED_FIXED=1`
option loads fixed tensors directly into their runtime order. It requires
text-only ordinary resident storage, the complete BF16 layer/LM-head route,
and the existing fixed-arena option. A shared definition supplies the exact
tensor order to both the loader and fixed-weight preload. Every other retained
tensor belongs to its ordinary shard allocation. The original BF16 bytes,
on-disk metadata, numerical kernels and runtime ABI remain unchanged.

The loader reads and validates all shard headers before allocating the fixed
owner. Each shard's header and file size must match that plan when its original
input ring is read. Both ring chunks and buffered tails copy only their
intersections with disjoint fixed/ordinary intervals. Tensor views and host
slices resolve through those same original offsets. The fixed-weight consumer
checks every name, size, ordering offset and pointer before borrowing the
completed owner. Its QKVZ+A/B contiguous view remains available without another
allocation. Stream draining precedes release of either owner; releasing the
fixed consumer clears its alias without freeing the resident store's buffer.

The original metadata from `D:\models\Qwen3.6-35B-A3B` contains 1,045 tensors.
The actual C++ planner maps 611 fixed tensors into 3,879,600,640 bytes and 82
other text tensors into 65,441,632,256 bytes. It excludes the same 352 optional
vision/MTP tensors as the text-only candidate. Total planned device storage is
69,321,232,896 bytes, or 6,462,144,520 fewer bytes than the current raw shard
store plus its fixed duplicate. This is allocation accounting, not a measured
physical-memory reduction. The plan has 54 ordinary and 501 fixed intervals,
with 1,665 beginning/middle/end verification samples.

Host sanitizers cover 1,001 generated layouts, 5,788,709 copied bytes, 37,847
tensor checks and 69,627 failure checks. Extracted runtime functions also cover
cross-shard fixed storage, original buffered tails, changed headers and file
sizes, mismatched fixed aliases, partial copy/event/wait failures, and
drain-before-free with one release per owner. The ordinary independently owned
fixed arena retains its original release behavior.

The native MSVC/HIP build now passes on baiying in 95867.030 ms. It verifies
all 103 build inputs and preserves the retained compiler arguments. The
13549056-byte whole-provider DLL has SHA256
`9164529b6ffe09ca57867a715c916acc2184c72da344fb617bcbcc3a17be6817`.

The original q8192/out512 regression also passes: all 512 output IDs and actual
callbacks match GB10, and first logit 10.375 has zero error. All 26 shard layouts,
1665 byte samples, shared fixed aliases and original QKVZ+A/B consumers pass.
Actual combined storage is 69321232896 bytes, with no duplicate fixed owner.
Load is 21750.9763 ms, TTFT 23250.782 ms and TPOT 101.226319 ms. The
[native build and q8192 record](../benchmarks/correctness/resident-ordered-compact-query-native-q8192-20260919.json)
binds source, actual allocation markers and final host/process cleanup. These
are functional and allocation observations, not retained performance or
physical-memory qualification.

The separate [original cold32k/out512 run](../benchmarks/correctness/resident-ordered-compact-query-cold32k-out512-20260919.json)
also passes all 512 IDs and callbacks, first logit 24.75 with zero error, all
four cold chunks and 30 compact-Q continuations. The same original tensor
samples and shared fixed aliases pass. Load is 21594.5557 ms, TTFT
256706.5187 ms, TPOT 208.317485 ms and native wall 385096.409 ms. Final host
and process cleanup checks pass.

The [cold32k memory comparison](../benchmarks/correctness/resident-ordered-cold32k-memory-20260919.json)
compares that run with the original scratch-reuse control on the same boot,
with identical arithmetic options and only the two declared storage flags
added. Minimum sampled available commit rises from 59165630464 to 65716465664
bytes; minimum available physical memory rises from 21861294080 to 22105653248
bytes. Accounting for their different preflight baselines, the maximum sampled
commit drop decreases by 6492917760 bytes and the physical drop by 102625280
bytes. These are system observations; process private bytes do not measure
GPU residency. The commit difference is consistent with the allocation
reduction but does not qualify full256k capacity or retained performance.

The original 262144-token owner32 and both 1024-token suffix/out512
continuations are now active via `run-resident-ordered-compact-query-prefix256k-r1.ps1`.
Source manifest SHA256 is
`5d9ad636b80efaa7949f26bba5eb1855fa7e56a06008e7e3202d316227da0756`.
The completed q8192 and cold32k boundaries and the previous full256k cleanup
are prerequisites. The original inputs, 8 GiB physical and 20 GiB commit
reserves, 28800-second native deadline and numerical tolerance are unchanged.
No optional row/stage observers are enabled. No output-token or full256k
acceptance is inferred from partial chunks. This candidate is not enabled in
a package or release.

The [preparation record](../benchmarks/correctness/resident-ordered-storage-preparation-20260919.json)
pins source `b3af8b6177c8bbf756be2b63af99bc7bb70b436e`, all 101 local
compilation inputs and two build/guard scripts. It binds the original metadata
reference, passing host checks and the native command's successful PowerShell
parse on baiying at 2026-09-19T11:22:34Z. Compile/native/transport bounds remain
240/300/390 seconds. The build and q8192 regression started after the earlier
whole334/CK370 full256k run stopped at the physical reserve after26/32 chunks
and completed cleanup.

The separate [compact single-query candidate](COMPACT_DECODE_QUERY.md) removes
historical Q storage from the CK suffix ABI. A subsequent call-site audit
corrects its proposed scope: current resident decode calls its own Q1 kernels,
and the chunked long route does not enter the one-query suffix ABI. Its only
current model caller is cold q8192 final-layer liveness, with a 64 MiB allocation
difference. It remains default-off and unrun natively; it does not address the
active full256k memory boundary.

The separate [in-place probability candidate](INPLACE_PROBABILITY_STORAGE.md)
targets the long-attention slab used by that route. It reuses each consumed
FP32 score cell for its original BF16 probability payload, removing a separate
probability matrix. Maximum planned scratch falls by 1084358656 bytes. Host
checks and native command parsing pass, while GPU comparisons and measured
memory benefit remain pending. It has no provider dispatch or runtime option
and does not alter the active full256k experiment.
