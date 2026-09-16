# Exact matrix QK with per-wave fallback queues

Source `8fa15f1` adds an isolated matrix QK kernel and native fixture. It keeps
the existing exact integer certificate and full internal carry for each K16.
Rejected cells enter a bounded LDS queue per wave using ballot prefixes.
Scalar consumers process consecutive live entries; accepted cells retain
their original result. Eligible inputs use the existing scalar FP32 alignment,
while subnormal/wide cases keep the original integer fallback. There are no
queue atomics or cross-wave consumers. No product dispatcher includes it.

The initial native compile exits1 because a diagnostic fprintf still names
an old `groups` variable. `14aab5e` changes only that identifier/label to
`variant`; both records are retained. The rebuilt fixture explicitly uses
WGP/wave32 and passes88 generated cases, including wide normal and subnormal
rows that exercise fully uncertified tiles, zeros, cancellation, late
exceptions, wide carried exponents and partial causal tiles. It checks5632
independent wide CPU dots, original input/encoding bytes and output redzones.

Both complete q7169 and repeated-row q8192 comparisons pass every raw score,
including unused tails and guards after each warmup and rotated sample:
15420096768 score comparisons and1936 additional independent CPU dots.
The q8192 extension repeats the first1023 rows from the immutable layer3
q7169 Q/K capture; it is not a new model prompt or GB10 token run.

| Route | q7169 total ms | q8192 total ms |
| --- | ---: | ---: |
| Retained prepared scalar | 273.5219 | 348.9436 |
| Matrix64, direct scalar fallback | 769.9603 | 1026.5896 |
| Matrix64, per-wave queue | 669.4350 | 880.0118 |
| Matrix32, per-wave queue | 524.5428 | 683.4635 |

Completed host timers include query encoding and the full score route, plus
the relevant one-time input preparation. Matrix64 queuing saves146.5778 ms
against its same-width control at q8192, but the fastest matrix candidate
still takes about1.96 times the retained route. Matrix32 has no direct32
ablation in this fixture. Keep this route isolated from the retained stack.

The three new kernels declare93/95/92 VGPRs,36416/40512/21440 LDS bytes,
and zero private bytes, respectively. These are static declarations and do
not establish occupancy or a measured hardware bottleneck. Local C ABI and
hygiene checks pass; native build, safety and both capture comparisons pass.
No new full Rust/Python suite run is claimed for these two isolated HIP files.

A preceding read-only audit found no duplicate original32-byte K16 blocks
among1835264 Q and229408 K groups, within each head/group position of this
capture. That observation alone does not describe other layers or prompts.

Evidence: [commands, source pins, failed build and complete comparisons](../benchmarks/correctness/matrix-wave-queue-qk-native-components-20260916.json),
SHA256 `a8f501c218a651f2fa70f4edfe0895a6ceefaee0e5ce2da456cb9055c7b3aa75`. Product dispatch, all GB10 tolerances, the10000 ms gate,
4187.415605 ms retained target and release state remain unchanged.
