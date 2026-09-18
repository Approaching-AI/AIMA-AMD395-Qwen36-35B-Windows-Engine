# Two-stream attention component

Source `68df394` compares three schedules for the unchanged qualified narrow
QK, fused probability/native PV and original selected replay. Two independent
128-query workspaces retain separate scores, probabilities, scales, errors,
raw outputs and candidate owners. Shared input preparation is read-only.

The control executes both slabs serially on the root stream. The first
candidate gives each complete slab a nonblocking stream. The second runs QK
on one stream and probability/PV/compaction/replay on the other, with a ready
event for each slab. Both workers wait for prepared inputs and join the root
before reuse, observation or release. Numerical kernels, coefficients and
provider dispatch are unchanged. This tests scheduling that permits overlap;
it does not establish actual hardware overlap.

All168 generated configurations pass across eight extents and seven data
modes, including signed zeros, cancellation, subnormals, admitted/excluded
exponents, nonaligned ranges, one-query tails, inactive slots and slot reuse.
The231 group attempts check98 original slabs and392 independent CPU QK dots.
Native build and safety complete in15316.526/11182.910 ms on baiying. All297
compiler inputs match the source; the1850880-byte executable has SHA256
`d2b46732a6c534d29b5a761ad720f5da28c61f8dc8f3471520badf924ea10784`.

The q8192 fixture extends the original7169-row GB10 attention capture with
1023 repeated rows. Every warm and timed group checks complete original score,
probability, scale, error, denominator, selected-raw and final-output surfaces.
Every candidate membership, duplicate check, list tail, guard and immutable
input passes. Warmups additionally compare native accumulators before replay.
All29364224 original GB10 context cells and256 CPU QK dots pass. Repeated
rows have original-arithmetic coverage only, with no model-token oracle.

| Schedule | Complete q8192 median ms | Three samples ms |
| --- | ---: | --- |
| Serial | 533.8146 | 540.6758,533.8146,530.8355 |
| Complete slabs on two streams | 531.8948 | 531.8948,532.1473,529.4945 |
| QK/PV stage pipeline | 532.8859 | 531.0384,532.8859,534.7626 |

All arms select3127598 original PV cells and perform260096 narrow/6144
original QK tiles. One warmup and three rotated completed-host samples run
per256-query group; the table sums32 groups covering64 slabs. Timing includes
all QK, probability/native PV, compaction, exact replay, dependencies and final
completion. Shared preparation adds5.1261 ms. Allocation, resets, reference
generation and verification are outside clocks. Both arms use the same pair
of workspace allocations, so this is a serial/concurrent scheduling comparison.

Keep both candidates isolated: median changes of1.9198/0.9287 ms and
overlapping sample ranges establish no stable material gain. No provider or
model run follows this version. The qualified23353.80795 ms q8192 median,
10000 ms first gate and retained4187.415605 ms target remain unchanged.
Defaults, packages and release status are unchanged.

[Pinned native commands and all results](../benchmarks/correctness/pipelined-attention-native-components-20260918.json):
163446 bytes, SHA256
`2e72572f6d097808b2d106118c58b88c796dc5f66b78f78bf4c02fd496c42793`.
Command `run-native-pipelined-attention-r1.ps1` runs on baiying with explicit
native/transport deadlines. The model is a capture reference, not loaded.
