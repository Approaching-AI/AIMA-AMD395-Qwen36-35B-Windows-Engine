# Compact Q for exact single-query decode

The qualified compact-suffix route omits historical Q storage for multi-query
prefill. Its one-token continuation still allocates Q/K/V at the complete
history capacity, although its score kernel reads only the current Q row.
That ordinary owner grows with each generated token and is separate from the
compact KV owner left by prefill.

The default-off `QRT_CK_SM121_COMPACT_DECODE_QUERY=1` candidate borrows the
caller's single Q row and refreshes four original prefix/suffix KV spans in the
existing compact owner. It grows that owner in 8192-token increments, capped
at 264736 tokens. The extra unused KV capacity is at most 8191 rows, less than
16 MiB. Copy lengths, causal positions and consumed history remain the actual
input extents. The separate long-workspace reservation option does not set
this decode capacity.

At 262145 tokens the previous ordinary Q/K/V allocation is 2684364800 bytes.
A fresh candidate compact KV owner reserves 542179328 bytes at the supported
maximum, while the caller retains its 8192-byte Q row. If compact KV already
exists after prefill, the same owner is reused or replaced after completion;
there is no additional decode Q/K/V owner. These are allocation sizes, not
measurements of physical memory or latency.

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

The current ordered-storage full256k run continues on CK370 without this
candidate. Original q8192/out512, original long owner and suffix continuations,
memory comparison and final host cleanup are required before using a new
provider in the runtime. No model, retained-performance, package or release
acceptance follows from the host checks or prepared fixture.
