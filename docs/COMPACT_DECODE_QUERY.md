# Compact Q for exact single-query suffix calls

The CK suffix ABI omits historical Q storage for multi-query prefill. Its
one-query form still allocates Q/K/V at the complete history capacity, although
its score kernel reads only the current Q row. This interface is used by cold
q8192 final-layer query liveness. The current resident token loop instead
calls its own Q1 score and PV kernels directly, borrowing separate prefix and
decode-tail K/V. It does not call this suffix ABI.

The [call-site audit](../benchmarks/correctness/compact-query-call-scope-correction-20260919.json)
corrects the initial description that this owner grows with every generated
token. The original q8192 run has one final-layer suffix call; cold32k and the
qualified prefix128k run have none. Chunked prefill disables final-query
liveness, and the batch suffix accepts 1024 or 8192 queries. Ten auxiliary
query-count-one CK diagnostics at position8191 belong to terminal corrections
and do not establish calls through the suffix ABI.

The default-off `QRT_CK_SM121_COMPACT_DECODE_QUERY=1` candidate borrows the
caller's single Q row and refreshes four original prefix/suffix KV spans in the
existing compact owner. It grows that owner in 8192-token increments, capped
at 264736 tokens. The extra unused KV capacity is at most 8191 rows, less than
16 MiB. Copy lengths, causal positions and consumed history remain the actual
input extents. The separate long-workspace reservation option does not set
this single-query capacity.

For the current q8192 caller, the candidate changes a single 83886080-byte
ordinary owner to a 16777216-byte compact owner; the Q row already exists.
This is a 64 MiB allocation difference, with no measured physical-memory or
latency benefit. It does not address the current full256k memory limit.

At the ABI's supported 262145-token extent, an ordinary Q/K/V allocation is 2684364800 bytes.
A fresh candidate compact KV owner reserves 542179328 bytes at the supported
maximum, while the caller retains its 8192-byte Q row. If compact KV already
exists after prefill, the same owner is reused or replaced after completion;
there is no additional suffix Q/K/V owner. This larger extent is a component
capacity example, not a call made by the current resident token loop.

Only the exact single-query score address changes. The original score body
retains every wave16 product, ordered K16 carry and scale. The compact kernel
subtracts the explicit query origin from the Q row index. The launcher accepts
this origin only for one query in the existing exact split layout. The same
precomputed-score PV kernel, online softmax, table lookups, accumulator order,
raw output and denominator remain in use. Other callers keep origin zero;
multi-query long attention retains its separate range-aware path.

Host checks exercise actual provider and launcher functions, option rejection,
copy spans, repeated input identities, reuse after prefill, capacity boundaries,
allocation failures, each failed copy and failed producer/consumer cleanup.
A separate ASan/UBSan test executes the actual score body with scalar reference
groups replacing the GPU collective. It checks 27 combinations, 3456 original
dots and 1769472 lane/group reads, including compact allocations at long logical
positions. This checks addressing and host arithmetic, not GPU wave execution.

The prepared native fixture compares four alternating full/compact calls for
each case. It checks every score word, raw PV accumulator, denominator, output,
nonzero output offset, unchanged Q/K/V, unused history and allocation guards.
There are 27 generated cases through 264736 keys and four original queries
from the GB10 16384-prefix plus1024-suffix capture, covering 16384 original
context cells. Each case also checks 128 independent CPU original dots.
Host C++ syntax uses mocked HIP declarations and the actual launcher signature;
native compilation and all native cases remain pending.

The current ordered-storage full256k run continues on CK370. The candidate
and its native commands remain unrun and default-off; it is no longer the
next full256k memory experiment. Existing host addressing checks remain valid,
but qualifying this generic ABI would not show a benefit for the resident
token loop. No model, retained-performance, package or release acceptance
follows from the host checks or prepared fixture.

The [preparation evidence](../benchmarks/correctness/compact-decode-query-preparation-20260919.json)
pins source `3d8f94c552ddb31baf768fda5a054e07411a429e`, 69 checked local files,
26 fixture compilation inputs and 66 CK compilation inputs. The union has
67 compilation inputs; the other two files are build and guard scripts.
`run-native-compact-decode-query-r2.ps1` passes the actual PowerShell parser
on baiying at 2026-09-19T14:22:39Z. Native bounds are 240 seconds for each
build, 900 for generated cases and 300 for captured queries. Dispatch requires
the active ordered-storage full256k run's completed host/process cleanup.
The first preparation remains preserved; revision2 corrects generated-case
metadata to report zero GB10 context cells and its actual query positions.
It does not change executable sources or numerical comparisons.
