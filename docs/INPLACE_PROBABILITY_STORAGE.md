# Reuse consumed long-attention scores for probabilities

The long attention slab retains FP32 scores and a separate BF16 probability
matrix until exact PV replay completes. The probability producer reads each
score once. This isolated candidate replaces that same FP32 cell with the
original probability's BF16 payload after the read. Each cell keeps its own
address, row and column; no other row or later K32 tile is overwritten.

A fixed normal-FP32 tag carries the sixteen payload bits. Stores and subsequent
replay loads use the FP32 type and perform no arithmetic on the carrier.
The original score load feeds the probability computation before its store;
exact replay starts after the probability kernel completes on the same stream.
Unused causal score cells remain untouched. The fused native PV still consumes
its original shared BF16 probabilities. Its softmax, native accumulator, error
bound and selected original K16 replay retain their numerical operations.

This removes the separate probability allocation while keeping four-byte
probability reads in exact replay. At 128 queries and 264736 keys, the slab
changes from 3325050884 to 2240692228 bytes, saving 1084358656 bytes. Scales,
errors, indices and candidate count retain disjoint checked ranges. The larger
probability read width may affect speed; allocation sizes do not establish
physical-memory savings, GPU residency or inference performance.

The separate [packed-row variant](PACKED_PROBABILITY_STORAGE.md) preserves
two-byte probability reads while reusing the same score owner. It remains an
unrun component candidate; the original tagged preparation below stays pinned
to its recorded source and command.

The original template defaults still use packed BF16 probabilities. No runtime
option or provider dispatch selects the new path. The active ordered-storage
full256k run uses its existing binaries and allocations.

ASan/UBSan host checks pass all 65536 payload encodings, 1588053 complete layouts
and 34979088 consumed cells under strict aliasing. Randomized row order checks
unconsumed scores, causal tails, original payloads and allocation guards. The
actual new pipeline and shared replay functions pass 48 ownership and failure
cases, including every observation stage, QK failure, each failed kernel,
candidate-count initialization failure and invalid/undersized inputs. Existing
nine attention regressions and C ABI smoke also pass. These host tests do not
execute GPU kernels.

The native fixture is prepared for 60 generated combinations through 264736
keys, with both existing error-bound policies, 128 independent CPU original
QK dots per combination, all probability payloads, native and replayed raw
surfaces, complete candidate identity, untouched cells, guards and inputs.
Complete owner checks observe all five stages, compare QK before overwrite,
reject an undersized slab before submission and use a nonzero output offset.

The original GB10 16384-prefix plus 1024-suffix capture provides 4194304 context
cells. A separate 8192-query extension repeats those original input rows and
provides component coverage only. One warmup and three rotated completed-host
samples compare the original and candidate complete pipelines in one executable;
every sample is checked. Row preparation is common and reported separately.
Copies, resets and validation are outside the attention timing.

Mocked HIP syntax checks accept the actual fixture, pipeline/replay functions
and actual kernel declarations. The first mock used an unused transport
constant under `-Werror`; the corrected mock passes with runtime sources
unchanged. Windows compilation, native numerical comparisons, original-token
model runs, system-memory benefit and retained performance remain pending.
This candidate does not qualify a package or release.

The [preparation record](../benchmarks/correctness/inplace-probability-storage-preparation-20260919.json)
pins source `dc9172f2a2ac6b583b1fd7705d8e2a5da598d6b1`, all 54 compilation
inputs and the guard script, the completed local checks, and both the original
mock failure and corrected syntax result. PowerShell on baiying accepts the
prepared command with zero parse errors at 2026-09-19T15:44:48Z; no native
build or GPU action was executed by that grammar check. Native deadlines are
240 seconds for build, 1500 for generated boundaries, 600 for the original
1024-query capture and 900 for the 8192-query extension. Both dispatch sides
require the active original full256k run's final host/process cleanup record
before these native actions can begin.
