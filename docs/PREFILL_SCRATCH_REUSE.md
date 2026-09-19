# Reuse scratch across cold prefill chunks

The original full256k run stopped at the unchanged physical-memory guard after
20 chunks. Each completed chunk reported 32 freshly allocated temporary blocks,
2756558868 cached bytes, and zero outstanding blocks before release. The sum of
those allocations is allocation traffic, not simultaneously live memory and
not evidence of a leak. The subsequent wholeb35 KV-reservation run stopped
after19 chunks under the same guard. Neither run produced an output token;
both completed host cleanup. KV reservation alone did not resolve the case.

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

Source `334883600c9ea1a9fc11fb4cf943c103b93403be` now builds on baiying in
94606.459 ms with the retained MSVC/HIP arguments and all100 build inputs
verified. Host and cleanup checks pass. The13510144-byte whole-provider DLL
has SHA256 `91652d9566246289c51df12b822e681672ff2a7cc5616a49325a8e44c70964dc`.
The [native build record](../benchmarks/correctness/prefill-scratch-reuse-native-build-20260919.json)
also attaches the local ownership and C cold/stream regressions. Original-token
model and allocation observations follow below. The build itself supplies no
inference or release acceptance.

The original q8192/out512 model regression then passes with CK3701495, FLA1d
and MoE923: all512 IDs and callbacks match, first logit10.375 has zero error,
and source/cleanup checks pass. Load is21217.0047 ms, TTFT23389.703 ms and
TPOT101.084037 ms. This [q8192 result](../benchmarks/correctness/scratch-compact-query-native-q8192-20260919.json)
does not enter the cold-chunk reuse scope and is a single functional sample.
The separate original cold32k/out512 activation check also passes all512 IDs,
callbacks, four chunks and first logit24.75/error0. All30 compact-Q activations
and the cross-chunk reuse marker pass, with no outstanding temporary handoffs.
Load is21292.5142 ms, TTFT257495.8903 ms and TPOT209.394177 ms. See the
[original cold32k result](../benchmarks/correctness/scratch-compact-query-cold32k-out512-20260919.json).

After the seed, the three suffixes make3234 pool requests but only32 new
allocations, versus three separate32-allocation scopes in the compact-Q-only
control. All3234 handoffs are returned; cached storage stays2756558868 bytes.
This demonstrates native application allocation reuse. Minimum sampled system
available physical memory is21861294080 bytes, compared with21889257472 for
compact Q alone and21916213248 for the b35/ea6 control. The
[three-run memory record](../benchmarks/correctness/compact-query-cold32k-memory-reuse-20260919.json)
does not show a physical-memory benefit or establish full256k capacity. No
retained performance or release qualification follows from these single runs.

The same whole334/CK370/FLA1d stack subsequently reaches 26 of the original
32 owner chunks, committing 212992 inputs, before the unchanged 8 GiB physical
reserve stops the process. Native wall is 14297954.775 ms; sampled available
physical memory reaches 8141103104 bytes and commit reserve stays above
35703902208 bytes. No output token or completed owner/suffix boundary is
produced. All final host checks pass and no engine process remains. The
[complete exit record](../benchmarks/correctness/scratch-compact-query-prefix256k-incomplete-20260919.json)
attaches all 250 completed compact-Q markers and the read-only workspace
observation. It supplies no numerical or performance acceptance. The different
stop points across separate runs do not isolate a leak or prove memory benefit.

The active run confirms that the full raw model store and a separate
3879600640-byte fixed-weight arena both exist. The prepared
[ordered storage candidate](RESIDENT_ORDERED_SHARDS.md) removes that duplicate
owner and unused tensor storage. Its planned allocation reduction remains
6462144520 bytes; native physical-memory benefit is still unmeasured.

[Preparation evidence](../benchmarks/correctness/prefill-scratch-reuse-preparation-20260919.json)
pins source `3348836`, the passing local checks and the bounded Windows build.
The command file passes baiying's PowerShell parser at2026-09-19T07:58:54Z;
that grammar check executed no build or model command. The native build above
started after both the original256k and compact-Q cold32k completed cleanup.

The independent include traversal now follows the compiler's `native/src`
search path and covers98 local compilation inputs plus two build/guard scripts.
The previous91-file traversal omitted seven files reached through that search
path. All seven were already fingerprinted in the original full build records
for both32a andb35; their original source bytes and native artifact hashes now
reproduce in the linked evidence. Only `prefill_chunks.h` differs fromb35 among
the100 inputs. This inventory correction does not create new native evidence.
