# Reuse scratch across cold prefill chunks

The original full256k run stopped at the unchanged physical-memory guard after
20 chunks. Each completed chunk reported 32 freshly allocated temporary blocks,
2756558868 cached bytes, and zero outstanding blocks before release. The sum of
those allocations is allocation traffic, not simultaneously live memory and
not evidence of a leak. The active KV-reservation rerun still uses the earlier
whole-provider binary.

This candidate holds the existing temporary allocation pool across the cold
suffix loop, after the first8192-token seed and persistent KV reservations.
Each suffix completes its existing device synchronization and returns its
temporary handoffs before the next chunk can reuse storage. A checked boundary
rejects a completed chunk with any outstanding temporary allocation before
advancing the session. The pool ends before final decode-tail resizing,
publication and the caller's streaming callback. Success, failure and exception
paths retain the existing scoped cleanup.

Persistent KV reserve, tail resize and nonreserved promotion explicitly use the
existing unpooled allocator. The source is included after a `hipMalloc` macro
that selects transient allocation, so relying on the surrounding pool flag for
persistent storage would make ownership fragile. Recurrent state, BF16 bytes,
causal positions, full chunk arithmetic and output tokens are unchanged.
The ordinary q8192 route does not enter this cold-chunk scope.

Two local tests execute the actual coordinator, allocator, nested scope,
release logic and KV ownership functions with ASan/UBSan. Eleven prompt shapes
cover16k through256k plus genuine1024-token tails. The same two mock temporary
spans serve every suffix, including the32-suffix case, and are gone before
publication and callback. Allocation failure, early/late suffix failure,
exceptions, an outstanding handoff, failed KV reservation, invalid recurrent
counters and cancellation all discard the failed owner and release scratch.
Persistent KV operations are tested under the production allocator macro with
a rejecting transient allocator; allocation/copy/free failures preserve the
prior committed bytes and counters.

Windows compilation and original-token model execution remain pending. The
candidate reduces repeated application allocation requests by construction;
system memory, driver residency and product latency must still be measured.
It makes no native accuracy, performance or release claim.

[Preparation evidence](../benchmarks/correctness/prefill-scratch-reuse-preparation-20260919.json)
pins source `3348836`, the passing local checks and the bounded Windows build.
The command file passes baiying's PowerShell parser at2026-09-19T07:58:54Z;
no build or model command was executed. Dispatch requires the active256k
run's completed cleanup.

The independent include traversal now follows the compiler's `native/src`
search path and covers98 local compilation inputs plus two build/guard scripts.
The previous91-file traversal omitted seven files reached through that search
path. All seven were already fingerprinted in the original full build records
for both32a andb35; their original source bytes and native artifact hashes now
reproduce in the linked evidence. Only `prefill_chunks.h` differs fromb35 among
the100 inputs. This inventory correction does not create new native evidence.
