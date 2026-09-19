# Long attention workspace reservation

The long attention provider currently grows its score/probability slab,
decoded Q/K storage, transposed V and compact suffix KV staging as history
increases. Disposable owners drain and release before growth, but repeating
those allocations may still affect the Windows driver. Existing memory
observations do not establish driver retention or a leak.

The default-off `QRT_CK_SM121_WORKSPACE_RESERVE_TOKENS` option reserves a
minimum capacity for those long owners. Zero or an absent value preserves
the existing growth policy. Positive decimal values must fit the existing
264736-token attention capacity. A later request can exceed a smaller hint
and grow normally; the hint never limits or extends the actual input.

Decoded operands and transposed V still refresh only the true key extent.
The long attention slab retains its original per-call stride, query count,
output positions, arithmetic and deadlines. Domain counters follow the
reserved decoded layout. Compact suffix staging reserves KV only and copies
the same four original prefix/suffix spans. Ordinary Q/K/V owners and
single-token staging retain their existing allocation policy. q8192 alone
does not allocate these long owners.

With the maximum hint, each growing owner can serve the full chunk sequence
from its first long call. This increases earlier allocation sizes; it does
not reduce the final live workspace size or prove lower physical-memory use.
The existing independent extended score/key owners still allocate lazily when
the true input crosses 131072 tokens. Reused buffers remain private under the
existing locks until all submitted consumers complete.

Local checks cover all 264736 valid extents with seven hints and both rounding
policies, totaling 3706308 successful cases and 21 invalid-input cases under
ASan/UBSan. The actual provider functions check owner identity across the full
history range, original launch extents, every preparation allocation failure,
retry, submission/completion failures and release after draining. The actual
suffix function checks reserved offsets, unchanged copy sizes, every failed
copy and retry. Existing demand-growth and disposable-workspace regressions
also pass.

Windows build, original GB10 continuations, physical-memory observations and
product performance remain unmeasured for this option. The currently running
full256k case uses earlier CK370 without this option. No package enables it.

The [preparation record](../benchmarks/correctness/attention-workspace-reservation-preparation-20260919.json)
pins source `abbd7b17e7ac9042e6b8be77275e655207c0c1ea`, 65 local compilation
inputs and two build/guard scripts. Only the CK provider and added capacity
policy differ from CK370 among compilation inputs. Ten local attention and
ownership tests, C smoke and public hygiene pass. The bounded build command
passes baiying's PowerShell parser at 2026-09-19T11:46:19Z. That first command
was prepared while the earlier whole334 full256k experiment was active.

The [second preparation](../benchmarks/correctness/attention-workspace-reservation-preparation-r2-20260920.json)
now requires the current wholeb3/CK370 original full256k run's final
host/process cleanup on both dispatch sides. A separate output directory
preserves earlier evidence. The source, bundle, all 67 build/guard inputs,
local checks and 240/390-second native/transport limits are unchanged.
PowerShell accepts `run-native-attention-workspace-reservation-r2.ps1` at
2026-09-19T16:17:37Z with zero parse errors and no script execution. The
eventual build record will include the exact preceding run-record hash.
The candidate remains unbuilt and unrun.
