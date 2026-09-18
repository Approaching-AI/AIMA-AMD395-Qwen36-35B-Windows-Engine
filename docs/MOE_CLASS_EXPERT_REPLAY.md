# Classified MoE expert replay experiment

`QRT_QWEN36_MOE_CLASS_EXPERT_ORDER_REPLAY=1` is a default-off q8192 experiment.
It requires the existing compact, four-lane staged-half routed replay and
expert ordering. It preserves the current candidate predicates, norms, BF16
endpoints, ordered K16 carries, and shared/routed stream ownership.

The existing lossless Row36 preparation also reduces a whole-row class in its
256-thread CTA. A row is either eligible for all-nonzero transformed replay,
eligible with possible zeros, or requires the original staged fallback. This
pass supports widths 512, 1024, 2048, and 4096; each row contains whole wave32s
and stays inside one CTA. New class arrays are separate from the registered
weight norm/validation cache and refresh with every prepared operand surface.

The original collected list is then permuted into 768 buckets: arithmetic
class first, expert second. Three replay kernels read disjoint device ranges;
they do not filter three copies of the original list. The fast variants use
the existing exact transformed arithmetic, including its original ordered
carry. The fallback retains the original staged dot. Gate, up, ordinary down,
and the retained compact down owner all use these ranges. Up finalization
still follows completion of every replay class on the routed stream.

At the maximum 4,194,304-candidate window, the ordered workspace is 16,786,436
bytes, 6,144 bytes above expert-only ordering. Separate activation and weight
class arrays add 2,359,296 bytes. No host count read or candidate removal is
introduced. Other prompt lengths and the option's default retain the existing
runtime route.

The local harness checks actual selectors, all three replay phases, sparse
and dense windows, debug outputs, allocation/release faults, and every
submission failure. Native fixtures compare the fused encoding and row class
against original BF16 inputs, each raw K16 carry against an independent
original primitive, all 768 permutation buckets, exact-once range consumption,
unchanged inputs, and buffer guards. These are component checks. The experiment
requires a same-DLL q8192 comparison with all 512 captured GB10 continuation
tokens and the first-token logit before any performance result can be retained.

Native qualification at source `078ef5247de8d762a5b32cbbee2b8799c9943c0b`
passes all nine local MoE tests, 640,416 raw carry comparisons across four
preparation shapes and three replay variants, and 108 GPU permutation cases
covering 56,682,342 selected candidates. Every native build checks 414 compiled
source inputs against the pinned commit. Provider build time is 20,230.842 ms;
the DLL is 1,001,472 bytes, SHA256
`03d1100a3a4c673e8e7706de897cc7a2fb935da5c131ece4569f1be889eef15c`.

Static VGPR counts for gate/up/down are 63/64/65 for generic replay,
43/44/46 for sparse transformed replay and 39/40/42 for all-nonzero replay.
All declare zero private storage. Fused preparation uses 23 VGPRs and 32 LDS
bytes. These declarations do not measure occupancy or prove a performance gain.

[Native evidence](../benchmarks/correctness/moe-class-expert-native-20260918.json):
457,834 bytes, SHA256
`82185184f28843659f4f19b98d9f91e1a12dd4c0a477142e283177d6ec031610`.
Command `run-native-moe-class-expert-r1.ps1` records baiying, all source inputs,
compiler arguments, arithmetic outputs and lifecycle/host checks. No model
inference is claimed by these generated fixtures.

Four fresh processes then run `D:\models\Qwen3.6-35B-A3B` on baiying with
`run-moe-class-expert-product-r1.ps1`. Only the classification option changes
in the same new MoE DLL. Whole `ddacdc9`, CK `df2ea51`, FLA `7b20c90`, CLI
`24c4304`, both terminal liveness flags, AOT assets and original numerical
thresholds remain fixed.

| Order / mode | Load ms | TTFT ms | TPOT ms |
| --- | ---: | ---: | ---: |
| 1 / OFF | 21392.7838 | 23894.5174 | 100.906486 |
| 2 / ON | 21436.4290 | 24041.0808 | 100.753126 |
| 3 / ON | 21388.3600 | 23986.1118 | 101.159627 |
| 4 / OFF | 21313.0722 | 23885.5287 | 101.464866 |

All 2,048 original GB10 output IDs, prompt IDs and actual callbacks match.
All first logits are 10.375 with zero error; host and activation checks pass.
Dense 130, coarse OUT nine, compact down 40 and adaptive linear OUT 30
candidate/count records are unchanged as diagnostics.

OFF/ON TTFT medians are 23,890.02305/24,013.59630 ms. Both ON observations
are slower than both OFF observations, a median regression of 123.57325 ms
(0.5173%). Reject this implementation as a performance route and keep it
unused by the current control. Reduced static register use is insufficient
evidence for retention. The measured result does not isolate how much time
is spent on fused preparation, ordering, extra submissions or replay.

The qualified baseline stays at 23,902.4417 ms with MoE `9235750`; it is not
recalibrated to the new OFF measurements. Code and package defaults remain
off. The 10,000 ms first gate, retained 4,187.415605 ms target, long-context
and release gates remain open. No package or release changes.

[Complete product comparison](../benchmarks/correctness/moe-class-expert-product-20260918.json):
1,550,852 bytes, SHA256
`000060ca4af84d6f565d27cd80060d7b60436d50d73a33023c644b727a9bd75b`.
