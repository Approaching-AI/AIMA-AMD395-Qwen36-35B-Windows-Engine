# Resident text tensor storage

The current loader allocates all71903776776 bytes of the26 original model
shards on the GPU. Read-only metadata captured from
`D:\models\Qwen3.6-35B-A3B` on baiying at2026-09-19T09:51:45Z identifies
333 vision tensors occupying893142496 bytes and19 MTP tensors occupying
1689281536 bytes. The original index SHA256 is
`41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83`.
Reading these headers did not load a model or change the running256k test.

The default-off `QRT_PREFILL_DESCRIPTOR_BATCH_RESIDENT_MODEL_TEXT_ONLY=1`
option packs the retained intervals of affected shards into smaller ordinary
device allocations. It excludes only `model.visual.` and `mtp.` namespaces;
unknown families remain resident. The complete original metadata and disk
offsets remain available. Device tensor views and host slices translate those
offsets through checked retained intervals, rejecting omitted tensors and
cross-hole requests. This option is for text inference without MTP; it does
not supply visual or speculative inference. Raw-shard replacement and alternate
managed, mapped or single-arena storage conflict with this layout and fail
before loading. Ordinary storage remains the default.

The existing32 MiB unbuffered input ring reads the original shards. Its copies
intersect each input chunk with the retained intervals; the buffered tail uses
the same mapping. A copy/event failure retains submitted work until the existing
store cleanup drains its streams, then releases device and ring storage. Each
retained interval receives the original beginning/middle/end byte checks.
All original BF16 bytes and inference kernels are unchanged. This experiment
does not claim reduced disk traffic or an inference timing improvement.

The original1045 tensor ranges pass the actual C++ layout and copy planner.
693 remain resident and352 are omitted. The four affected shards each retain
one contiguous interval; the other22 preserve their complete raw layout. Planned
device storage is69321303784 bytes,2582472992 fewer than the raw store. This
is allocation accounting from metadata, not measured physical-memory savings.
All26 shards still have78 planned verification samples.

Two ASan/UBSan tests pass:2000 generated layouts copy40806028 bytes, check39330
tensor/slice mappings and2006 failed-copy/layout cases, with intact guards and
padding. Extracted runtime copy, device-view, host-slice and cleanup functions
also pass. Their mocks cover partial submission, event and wait failures,
drain-before-free, an actual buffered tail, omitted lookups, and the unchanged
ordinary/host-backed mapping. The C smoke and public hygiene pass. Native
MSVC/HIP build, original GB10 continuations and physical-memory measurements
remain unrun for this candidate. No product or release acceptance follows.

The separate decode-order fixed-weight arena holds3879600640 additional bytes
in the current stack. Disabling it also removes the existing QKVZ+A/B alias and
requires a1416626176-byte combined projection allocation. Its net effect must
therefore include that replacement and be measured with original continuations;
the gross arena size is not an established memory saving. This candidate leaves
that option unchanged.
