# Queued adaptive linear OUT replay

Source `6e4908b` adds variant `2` to the default-off
`QRT_QWEN36_Q8192_LINEAR_OUT_VARIANCE_REPLAY` option. It separates original
K16 dots from the per-token interval certificate used by
[the fused experiment](OUT_VARIANCE_REPLAY_EXPERIMENT.md). The eligible shape,
algorithm0 producer, prepared K16 arithmetic, empirical PPB/midpoint envelope,
18-round priority policy and original residual/RMSNorm consumer stay the same.
No golden or completed control values enter the runtime.

Each token compacts selected cells into a shared queue before one global
allocation. Independent four-lane original dots consume the GPU count without
host count reads. The certificate keeps pending masks and maximum contributions
across rounds, packing BF16 interval endpoints into the no-longer-needed raw
matrix storage. Final conversion restores its rounded F32 carrier. Nineteen
count clears, 18 replay launches and 18 certificate launches remain ordered
on the original stream. The final round replays every pending candidate;
observed envelope violations disable early certification for that token.
The empirical producer envelope remains conditional: the implementation does
not prove that an unobserved hardware error cannot exceed it.

Variant 2 owns 170,205,188 workspace bytes, 75,530,244 more than variant 1.
All submitted work drains before release, including failures. Actual-owner
ASAN/UBSAN checks cover both variants with 42 successes, 1638 injected failures
and eight invalid reports. Full local checks pass 54 Rust and 477 Python tests
(two skipped), clippy, C/q16 ABI and hygiene before the final GPU-only shared
compaction edit. Final C/hygiene/owner checks pass again. Native testing covers
that final edit; no second full local suite is claimed.

The same Windows numerical executable tests both variants. Each passes all
63 cases and 903168 independent CPU original dots, every residual and
normalized F32/BF16 result, immutable inputs, queue/state/carrier checks and
all redzones. Generated cases include full and empty queues, cancellation,
subnormals, nonfinite norms and deliberately undercovered envelopes. Per-case
counts match between variants. These component fixtures do not qualify
inference performance or arbitrary producer-envelope accuracy.

The native safety build completes in 116056.291 ms and whole-provider build
in 93918.726 ms, with all host guards passing. The whole DLL is 13395456 bytes,
SHA256 `42c64ba3c621c10536585caa89f29c075b53d1a9b13b17cadef5831d6153c486`.
Initialization/replay/certificate kernels declare respectively 24/53/49 VGPRs
and 11272/0/12308 LDS bytes, all wave32 and zero private bytes. These static
descriptors do not measure occupancy or establish performance.

The fresh same-DLL q8192/out512 pair runs on `baiying` with the real model
`D:\models\Qwen3.6-35B-A3B`. Only the variant flag differs. Both runs preserve
all original prompt IDs, 512 GB10 output IDs, 512 actual callback IDs and
first logit 10.375 with zero error against tolerance 0.125. All 30 intended
linear layers execute variant 2.

| Variant | Load ms | TTFT ms | TPOT ms |
| --- | ---: | ---: | ---: |
| 0, original | 21368.0706 | 27870.9080 | 100.563683 |
| 2, global queue | 21234.5688 | 27193.1037 | 100.731388 |

Actual totals match the fused experiment: 58,489,767 original candidates,
13,455,017 first replays, 15,843,176 additional replays and 29,191,574 skipped
(49.9088567%). All 245,760 token rows certify, with 1,712,887 rounds,
225 complete-replay fallback rows and zero observed boundary failures.
Completed owner clocks total 1938.4483 ms and include production, preparation,
replay, conversion, completion and ownership. Other 130 dense calls and ten
coarse FA OUT calls retain their per-call correction counts. Count identity
is diagnostic; GB10 token/logit/callback boundaries determine correctness.

Keep variant 2 default-off and outside the retained stack. The single
677.8043 ms TTFT reduction does not establish a repeatable gain; TPOT rises
0.167705 ms. The 10000 ms TTFT gate, 4187.415605 ms retained target and
30000 ms load limit remain unchanged. No prefix, long-context, package or
release acceptance follows. Preserve this experiment for combined structural
trials and investigate adaptive strict attention-denominator refinement with
the original fast prepared QK repair and complete output certification.

Evidence:

- [Native source, numerical cases, ownership checks and kernel resources](../benchmarks/correctness/out-variance-queue-native-20260917.json), 124107 bytes, SHA256 `33a5f12f4f0007a549c6fef8a2bcc99b639bc04f5d3b7c8d957af0cf3cf9afdb`.
- [Same-DLL real-model pair, complete GB10 boundary and decision](../benchmarks/correctness/out-variance-queue-product-20260917.json), 373988 bytes, SHA256 `fa44462617c7c62401cde37931298dd52b54d4ed1629f9425e5daba47b6fb1ba`.
