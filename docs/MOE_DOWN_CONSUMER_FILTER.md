# Actual MoE down consumer filtering

Source `0cfe413` implements actual candidate omission using the complete
consumer established by the [read-only audit](MOE_DOWN_CONSUMER_AUDIT.md).
Source `d852ff7` fixes a naming conflict in the native test fixture; production
code is identical. The option `QRT_QWEN36_MOE_DOWN_CONSUMER_FILTER` accepts
0/1 and defaults to 0.

## Implementation

For each q8192 token/channel, one GPU pass encloses the original weighted
BF16 contributions from all eight experts. It follows the original VT4
addition tree and the original gated shared BF16 merge. Only a fixed final
combined BF16 value permits removal of the selected contributions as a group.
This fixes the complete unrounded F32 residual, including subsequent RMSNorm
input. Invalid or nonconstant ranges retain all original selected replay.
The range remains conditional on the unchanged empirical projection envelope.

The certificate waits on the same invocation's shared-done event, which the
caller records before routed work. A 16777216-byte bitmap is then consumed
by the original collector. Uncertified candidates keep the original expert
permutation and original K16 replay. Guarded private workspace totals
16778528 bytes. Independent certificate, collector and replay counts are
checked after the original production combine. The owner drains submitted
work before freeing its allocations on both success and failure.

The scope is the qualified q8192 fused merge with weighted routed BF16
endpoints, VT4 addition, BF16 routed sum and split-variance residual. The
read-only down observer and actual filter cannot be active together. Other
logical lengths retain their original dispatch. No external reference or
completed control output selects or supplies computation.

## Verification

Full local checks at `0cfe413` pass 54 Rust tests, 480 Python tests (two
skips), C/q16 ABI, clippy and hygiene. The actual asynchronous owner passes
14 injected transport failures and ten invalid reports under sanitizers,
including shared-event ordering, both allocations' guards, count identities
and drain-before-free. The production-merge interval corner test remains
active in that suite. The subsequent HIP-only fixture naming fix is covered
by the complete native build and numerical sequence at `d852ff7`.

The 84 existing expert-order/routed native reports pass. Sixteen new GPU
cases exercise the actual certificate, collector, original replay and
production merge at 1/3/17/33 tokens with empty, dense, sparse and wide-range
selectors. They compare 442368 final F32 outputs with zero mismatches while
actually skipping 794635 of 2305711 selected replays. Exactly 539395 raw route
values differ from the complete original replay, demonstrating actual
omission rather than merely reporting a hypothetical count. Every wide-range
case retains all selected work. The CPU computes 2464 distinct periodic K16
references and checks 2305711 selected cells against those references.
Input immutability, all guards and counter identities pass.

Two preparation failures are preserved: a PowerShell single-argument array
was serialized as a string and rejected before launch; the first new fixture
compile found a function/local-variable name collision. Explicit arrays and
the fixture rename resolve those issues. Revision 3 completes both native
builds, all numerical reports and the production DLL build with all guards.

## Real model pair

Both processes run `D:\models\Qwen3.6-35B-A3B` on baiying with the same
new MoE DLL, whole `6e4908b` in mode 0, retained CK/FLA/CLI/AOT assets,
dense PPB1000, MoE PPB512, expert ordering and coarse FA OUT. Only the
consumer-filter flag changes.

| Filter | Load ms | TTFT ms | TPOT ms | GB10 IDs / callbacks | First logit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Off | 21379.9922 | 27851.3224 | 100.643059 | 512 / 512 | 10.375 |
| On | 21519.9587 | 27889.4209 | 101.079028 | 512 / 512 | 10.375 |

Both match the original 8192 prompt IDs, all 512 output IDs and actual
streaming callbacks, with zero first-logit error against the 0.125 tolerance.
All 40 intended calls run the filter. Of 158740988 original selected down
replays, **126904922 are actually omitted (79.9446%)** and 31836066 execute.
Each call's selected, omitted and invariant-cell counts exactly reproduce
the prior read-only observer. All 160 other dense and ten coarse FA
candidate/dispatch counts match. These count comparisons are diagnostics;
the GB10 token/logit boundary remains the correctness authority.

The single pair has +38.0985 ms TTFT, providing no net saving to retain and
not establishing a repeatable regression. The actual native certificate has
55 VGPRs, zero LDS/private allocation and wave32 in the built code object;
these are static resources, not measured occupancy or duration.

## Completed MoE profile

Two further fresh processes use the same artifacts and options, adding only
MoE subphase profiling and disabling marker filtering. Both preserve all 512
GB10 output IDs, actual callbacks and first logit 10.375. All 40 ordinals and
all finite, nonnegative timing intervals are present. Enabled filter counts
exactly match the uninstrumented run.

| Completed interval, ms | Off | On |
| --- | ---: | ---: |
| Down correction | 805.620793 | 663.870898 |
| Complete down | 1539.142910 | 1401.854189 |
| Routed | 5099.483336 | 4965.950205 |
| Shared | 3098.619452 | 3100.860420 |
| Complete MoE | 5283.109901 | 5201.154889 |

Down correction includes certification, the shared-event dependency,
collection, permutation and remaining original replay when enabled.
Nested intervals and routed/shared overlap must not be added. The targeted
interval saves 141.749895 ms and complete MoE saves 81.955012 ms in this
instrumented pair. Diagnostic TTFT is 31550.2614 / 31503.1318 ms; profiling
inflates wall time and these values do not qualify retained performance.

## Decision and evidence

Keep the filter default-off and preserve the retained stack. Even eliminating
the original 805.620793 ms targeted interval cannot close the remaining
q8192 gap. Continue a broader exact arithmetic or dataflow replacement;
further narrow filter tuning has insufficient measured scope. Do not combine
it with another candidate on an assumed speedup.
The existing empirical range was checked against every actual selected
replay in the prior same-golden observer; this does not establish a universal
hardware bound or check the omitted values during this run. No retained
performance, long-context, package or release qualification follows.
The 10000 ms gate and 4187.415605 ms retained TTFT target remain unchanged.

- [Native builds, owner checks, failures and complete numerical evidence](../benchmarks/correctness/moe-down-consumer-filter-native-20260917.json), 290496 bytes, SHA256 `552de518ca78c4e3077ebd6298ca02c9efa6418fcfa1a6ac494f4ca50a7ca97a`.
- [Same-DLL model pair, original GB10 boundary and actual removal](../benchmarks/correctness/moe-down-consumer-filter-product-20260917.json), 436685 bytes, SHA256 `404d97198e4d9761754b657c78b9639c9e785b4f7e9ce727ebc7536afce1932e`.
- [Same-DLL completed MoE profile with the GB10 boundary](../benchmarks/correctness/moe-down-consumer-filter-completed-profile-20260917.json), 508056 bytes, SHA256 `64438851c2a7cce61cf209d86a8a90ab72b9532a03c2c751061c15c5f866e0f8`.
