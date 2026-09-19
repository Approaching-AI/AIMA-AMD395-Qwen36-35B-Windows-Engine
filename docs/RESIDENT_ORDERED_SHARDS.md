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
physical-memory qualification. Separate cold32k is active; full256k remains
unrun with this candidate. It is not enabled in a package or release.

The [preparation record](../benchmarks/correctness/resident-ordered-storage-preparation-20260919.json)
pins source `b3af8b6177c8bbf756be2b63af99bc7bb70b436e`, all 101 local
compilation inputs and two build/guard scripts. It binds the original metadata
reference, passing host checks and the native command's successful PowerShell
parse on baiying at 2026-09-19T11:22:34Z. Compile/native/transport bounds remain
240/300/390 seconds. The build and q8192 regression started after the earlier
whole334/CK370 full256k run stopped at the physical reserve after26/32 chunks
and completed cleanup.
