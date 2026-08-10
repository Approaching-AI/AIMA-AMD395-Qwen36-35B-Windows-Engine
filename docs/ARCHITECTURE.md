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
