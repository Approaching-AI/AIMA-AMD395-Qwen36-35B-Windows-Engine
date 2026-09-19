# Architecture

The repository mirrors the high-level organization of the companion Linux
engine while replacing its runtime and provider integration with native
Windows/MSVC/HIP surfaces.

## Runtime layers

| Layer | Responsibility |
|---|---|
| `engine/qrt-server` | HTTP, OpenAI schemas, SSE, tokenizer/chat templates, queue, lifecycle |
| `engine/qrt-cli` | Thin native product/ABI command surface |
| `native/src` | Stable C ABI, model contract, baseline and transaction orchestration |
| `native/providers/whole_provider.cpp` | Model-specific weight loading, prefill/decode, cache ownership, provider dispatch |
| `native/providers/*` | CK attention, Triton MoE, AITER/FLA GDN, host BF16 helpers |
| `native/generators` | Fixed-shape Triton AOT generation |
| `native/aot/gfx1151` | Qualified generated code objects and metadata |

Rust owns transport, process lifecycle, and thin safety boundaries. The native
provider owns model memory and the timed inference path. The ABI reports
structured status and metrics; Rust never substitutes a stub or proxy result
for native inference.

## Prompt execution

The service tokenizes text/chat input or accepts raw token arrays, validates
the total-context contract, and enters the bounded batch-one queue. The native
provider selects retained q8192 tiles plus exact q1024/tail paths, executes all
40 layers, produces the first-token logits, and then runs decode up to the
requested bound. SSE serialization happens incrementally after each native
token.

Arbitrary prompt lengths are decomposed internally; callers do not select a
shape and are not restricted to benchmark sizes. At `max_model_len=262144`, a
262,143-token prompt can request one output token, while a 262,144-token prompt
requesting output is rejected before native execution.

The opt-in `QRT_QWEN36_CHUNKED_PREFILL=1` route bounds cold activation
carriers to 8192 real inputs, followed by an optional 1024-input terminal chunk.
It seeds the ordinary resident session once, extends original FP32 recurrent
states and convolution rings, and promotes each completed chunk's BF16 KV into
the history under the session mutex. Only the completed prompt's sampled token
crosses the stream ABI; a failed replacement retires its partial state. Decode
scratch is reserved for the full prompt before the first chunk runs. Attention
storage grows lazily beyond 131072 tokens to a 264736-token capacity, covering
the 262144-token owner, 1024 real suffix inputs and resident decode tail. This
internal storage bound is separate from the server's configured total-context
budget and from model qualification. Prefix checkpoint capture is not combined
with the chunked mode.

The single-round RoPE correction passes the original 131072-token owner and
both 512-token continuations of its 1024-token suffix, including first logits,
streaming, state restoration and changed-prefix rejection. The subsequent
ordered-storage stack passes separate original q8192/out512 and cold32768/out512
requests. Its original 262144-token owner and suffix run is still pending;
earlier full256k runs stopped at the physical-memory reserve before producing
output tokens. These experimental routes remain outside the released package,
and the performance targets remain unmet. See the
[128k product boundary](../benchmarks/correctness/rope-single-round-prefix128k-product-20260919.json)
and [current storage evidence](RESIDENT_ORDERED_SHARDS.md). Per-descriptor
metrics describe the last chunk; the provider wall and actual callback clock
include all chunks and KV promotion. This mode does not change the 1024-input
prefix API's complete teacher-prediction contract.

The storage experiments preserve original model bytes and numerical kernels.
[Ordered resident storage](RESIDENT_ORDERED_SHARDS.md) loads fixed weights
directly into their consumer order and shares that owner with the loader;
[scratch reuse](PREFILL_SCRATCH_REUSE.md) retains drained temporary allocations
between calls. [Compact suffix queries](COMPACT_SUFFIX_QUERY.md) borrow only
the new Q rows while staging complete K/V. The separate default-off
[compact single-query candidate](COMPACT_DECODE_QUERY.md) extends that CK ABI
ownership to one query. Current resident decode calls separate Q1 kernels;
this candidate's native and model checks remain pending. Allocation accounting alone
does not establish physical-memory capacity or inference correctness.

The standalone attention replay also exposes experimental layouts 19–21.
They reconstruct K16 dots from four integer matrix products and sparse
corrections for operand decomposition and product alignment. Preencoded Q/K
rows can be shared across matrix tiles. Seven complete original-input runs
preserve the BF16 reference and all qualified replay surfaces, but the best
candidate still trails the exact tiled QK control. Product dispatch does not
select these layouts. See `benchmarks/correctness/attention-integer-core-20260913.json`.

The routed MoE correction scheduler also supports
`QRT_QWEN36_MOE_COMPACTION_WINDOW_BLOCKS` (powers of two from 1024 to 16384).
The default is 1024; wider windows require routed compaction. The maximum
window uses 16 MiB plus one counter and keeps at most 1024 replay blocks.
Each subgroup processes disjoint candidate slots in order, with unchanged
selection, K16 arithmetic and up-finalization dependencies. The 16384-block
experiment passes the declared q8192 boundary and reduces its measured
callback by 2.88 seconds; broader qualification remains pending.

`QRT_QWEN36_MOE_PARALLEL_GATE=1` is a separate opt-in M64/N64/K64 matrix
schedule. It reads the original BF16 weights and computes gate/up tiles
independently, then uses the existing exact corrections. Both original
overflow descriptors are decoded into independent M64 tiles. Its native
FP32 comparison and declared q8192 continuation pass, but the product timing
difference is too small to retain. The default is off; see
`benchmarks/correctness/moe-parallel-gate-20260913.json`.

## Prefix cache

Snapshots are owned by the resident provider. A compatible extension borrows
the longest reusable prefix through copy-on-write, then commits a new snapshot
only after successful inference. Unrelated prefixes and engine instances have
separate ownership; failed transactions do not partially mutate a retained
snapshot.

## Concurrency and shutdown

The hardware route is batch-one. A fair semaphore admits one request and keeps
a bounded FIFO wait set. The detached `start` command records identity and
readiness in a state file. `stop` validates that identity, closes admission,
releases queued waiters, waits for the active request, drains provider streams,
and confirms process exit. Windows detached creation uses a new process group,
detached process flags, and breakaway-from-job behavior so an SSH or terminal
job ending does not kill the resident engine.

## Dependency surface

The installed inference route needs the project executable/providers plus AMD
ROCm runtime DLLs. WSL, Triton, CK source, Python, Rust, and compilers are build
dependencies only. Model weights remain external.
