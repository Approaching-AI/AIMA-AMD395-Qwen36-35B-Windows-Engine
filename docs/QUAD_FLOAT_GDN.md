# Four-lane exact GDN candidate

This isolated candidate distributes each original K16 dot over four adjacent
lanes in W/U, recurrent state and output. Each lane holds four products. The
retained scalar schedule holds sixteen, which contributes to register pressure
in the complete GDN chain. The completed q8192 profile attributes 3327.918 ms
to GDN recurrence; an actual product measurement is required to decide whether
the new schedule helps.

The candidate reuses the existing strong row certificate: every nonzero BF16
operand must have exponent 84 through 174. Both complete operand rows are
checked, including freshly rounded recurrent checkpoints and residuals. K64
and K128 fall within the existing K4096 proof. Product multiplication, unsigned
modulo reduction, integer carry, ordered K16 normalization, BF16 endpoints,
exponential lookup and final FMA retain their original definitions. Excluded
rows use the retained scalar operation in lane zero.

Shared packed operands and nonoverlapping arena lifetimes follow the retained
scalar layout. A quad publishes each result once. W/U captures every input V
column before writing U, including when U and V share storage. Eight complete
state rows stay with their original CTA. Output retains each Q/H result until
all readers cross the barrier that permits reusing the shared arenas.

The local test runs the actual candidate kernel bodies under ASan/UBSan with
threaded CTAs and host-emulated subgroup transport. Its independent wide
integer accumulator checks original tensor values, one-token and 65-token
tails, the first and last heads/column tiles, U=V, all recurrent checkpoints,
untouched output cells and immutable inputs. Both certified and fallback rows
execute. The test passes after correcting a test-helper name collision with
the macOS math library; the initial failed compile is retained in local evidence.
This host check does not execute GPU DPP instructions.

The native fixture compares the unchanged complete chain with quad W/U plus
output, then with quad state added. It checks seven input families and eight
boundary sizes, both U ownership modes, all intermediates and independent CPU
dots. Original GB10 q7169 replay and a separately labelled q8192 capture
extension are prepared. The extension repeats captured rows and is a component
diagnostic; only a subsequent original-token q8192 model run can establish
product correctness or performance.

The candidate now compiles on baiying in8786.385 ms. All336 native generated
configurations pass in7096.39 ms. Original q7169 capture, q8192 capture
extension and both generated throughput families also pass, including every
warmup and measured attempt, intermediate, alias, guard and original boundary.
The q7169 capture checks29364224 original output/W/U/residual cells each,
113 complete checkpoints and524288 final FP32 state cells. The q8192 extension
still repeats1024 rows after7168 original rows and is a component diagnostic.

Completed component medians for the captured production U=V ownership are:

| Query count | Retained scalar ms | Quad W/U + output ms | Quad state added ms |
| --- | ---: | ---: | ---: |
|7169|72.998|104.9623|131.1464|
|8192 extension|84.1881|121.1518|152.1352|

This schedule is not selected for runtime integration. It is substantially
slower in the complete captured component, so it supplies no promising model
performance route. Compiler VGPR counts rise from78 to82 for W/U,77 to98 for
output and113 to122 for state. LDS sizes are unchanged; all seven selected
control/candidate kernels have zero private allocation and spills. These
static counts do not establish measured occupancy or the cause of slowdown.

The [native component evidence](../benchmarks/correctness/quad-float-gdn-native-components-20260919.json)
attaches all five completed runs, original GB10 files, resource extraction,
362 verified build fingerprints and the independently walked23-file local
include closure. The provider dispatch and packaged profile remain unchanged.
No real-model, retained-performance or release acceptance is claimed.

Source `50cc22d`, the local arithmetic and host checks, and the prepared native
commands are recorded in
[`pending-native-candidates-20260919.json`](../benchmarks/correctness/pending-native-candidates-20260919.json).
The command file passes the actual Windows PowerShell parser. The native runs
above follow completed model cleanup and finish before the new full256k
memory-candidate run. The grammar check itself executed no fixture or model.
