# Real-model performance

## Long deferred-PV components, 2026-09-18

Source `e0d7e4a` passes23652092 native bound comparisons,100 generated
attention configurations through264736 keys and15 short-template regressions.
Complete original scalar PV, conservative candidate supersets and all original
GB10 context cells pass. Across1024/8192 queries after16k history, complete
attention medians improve318.2968 to263.5991 ms and3924.3890 to3418.2800 ms.
Every candidate sample is faster than every control. Repeated rows in the
larger shape have original-arithmetic coverage only.

Provider `ea6faff` now passes same-DLL real16384/out32 OFF/ON/ON/OFF checks:
all128 original GB10 IDs, prompts, logits25.625/error0 and callbacks match.
TTFT medians improve76932.886950 to73840.139151 ms, saving3092.747799 ms
(4.02%), with every ON below every OFF. All loads are below30 seconds;
TPOT shows no gain. A separate q8192/out512 regression passes all512 GB10 IDs
and confirms that both requested long options remain inactive. Its unpaired
23251.8877 ms TTFT does not replace the23353.80795 ms qualified q8192 median.
The new DLL's prefix/larger-context qualification is pending. Defaults,
packages and all retained performance/release gates remain unchanged.
[Derivation, complete timing scope and evidence](LONG_FINAL_PV.md).

## Directional C64 projection components, 2026-09-18

Source `0d20e48` passes all88 native safety configurations and complete
captured OUT/QKV checks, including original GB10 operator references. Its
directional loss accounting reduces selected cells20.40%/16.03%, but the
complete64x32 medians regress85.8026 to96.0616 ms for OUT and138.5076 to
172.4419 ms for QKV. Every candidate sample exceeds every shared control.
The smaller tile removes the64x64 producer's spills while preserving all raw
centers, bounds and candidates; additional bound work still costs more than
the saved replay. Keep both schedules isolated. These repeated-row operator
shapes do not change the23353.80795 ms qualified model TTFT or release status.
[Design, all variants and native evidence](SIGNED_LOSS_BOUND.md).

The subsequent deferred-envelope source `ec96fbd` passes701616 host boundaries
and88 native configurations, reusing prepared masks and finalizing its error
bound once. Complete OUT improves relative to the first directional route,
but remains88.0962 ms versus85.8982 ms for its shared control. QKV is
177.6558/183.6243 ms versus139.7932 ms. Every candidate sample is slower than
every shared control; all complete original/GB10 operator checks pass.
Both schedules remain isolated. Their proof and limits are in the same
[experiment record](SIGNED_LOSS_BOUND.md).

## Persistent deferred GDN components, 2026-09-18

Isolated source `6ef29d3` gives each persistent CTA eight complete recurrent
state columns and their replay history. Original interval propagation, BF16
checkpoints, cached ordered K64 replay and final FP32 state remain unchanged.
The control uses the qualified shared-arena state/output route. WU and output
are identical across arms; candidates use scalar or coarse-certified WH.

All 144 generated safety configurations pass. Every q8192 warmup and rotated
sample matches complete original output, state and intermediate surfaces,
with 172032 independent CPU dots. Observer-only original state64 traces are
contained by all checked candidate intervals. The original state-dot count
remains 38218556 of 67108864, matching the earlier deferred primitive.

| U ownership | Control ms | Persistent / scalar WH ms | Persistent / coarse WH ms |
| --- | ---: | ---: | ---: |
| Separate | 70.0704 | 162.6714 | 161.8188 |
| U = V | 69.9304 | 159.5315 | 160.1558 |

Each median covers three complete q8192 samples in eight 1024-token
segments, including initialization, history, replay, diagnostic records and
finalization. Every candidate sample is slower than every matching control
sample. Keep both candidates isolated and choose a different structural
route. Runtime defaults, the qualified 23353.80795 ms model TTFT and package
remain unchanged. These generated components have no model-token acceptance.

[Native evidence](../benchmarks/correctness/persistent-deferred-state-gdn-native-components-20260918.json):
343059 bytes, SHA256
`75384809220778fddededdbcf38cedd655f3e92ee13d195ff6bbd1f4d83d3489`.

## Exact long-attention provider, 2026-09-18

CK `65a2139` passes four real cold16384/out32 runs in OFF/ON/ON/OFF order:
all 128 original GB10 IDs, prompts, first logits 25.625/error 0 and actual
callbacks match. Same-DLL TTFT medians improve 85460.99605 to 76772.1976 ms,
reducing 8688.79845 ms (10.17%). Both ON samples beat both OFF samples and
all loads remain below 30 seconds. There is no decode improvement.

The separate q8192/out512 regression passes all 512 GB10 IDs, first logit
10.375 and callbacks, with the long route inactive; TTFT is 23351.8109 ms.
The qualified short median remains 23353.80795 ms. The enabled long route
becomes an experimental control, with defaults and package unchanged.
The same stack also passes all 512 initial-retry and 512 timed-hit GB10 IDs
for the original 16384-prefix + 1024-suffix case, including actual streaming,
state restoration and changed-prefix rejection. Timed-hit TTFT is
9047.9052 ms and TPOT 158.64168 ms; prefix performance targets remain open.
The original 32768-prefix + 1024-suffix case also passes both 512-token
continuations, original first logits, callbacks, restoration and changed-prefix
rejection. Independent scratch growth is observed at the owner and suffix
boundaries. Load is 21281.3331 ms, hit TTFT 14602.8872 ms, TPOT 214.793319 ms
and total 124426.6185 ms; the retained performance ceilings remain unmet.
The original 65536-prefix + 1024-suffix case also passes both512-token
continuations, first logit5.9375/error0, streaming, restoration and rejection.
Load is21346.7386 ms; hit TTFT27707.133301 ms, TPOT345.768637 ms and
total204461.080601 ms remain above the unchanged64k ceilings. All eight owner
chunks and checked scratch growth complete within the1800-second native bound.
Cold32k/64k continuations,128k/256k contexts and release remain unqualified.
[Implementation, full timings and evidence](LONG_ATTENTION_PIPELINE.md).

## Long complete-attention components, 2026-09-18

Isolated source `d926858` combines range-aware narrow QK, fused
probability/native PV with the original per-group error bounds, and original
exact PV with register rescaling. The 512-group short-context final-bound
proof is not extended. Native safety passes 100 configurations through
264736 keys and 15 regressions of the unchanged short-template default.
The first revision's rejected long replay launch remains in the evidence.

| Captured queries after 16384 history | Original / 32 | Candidate / 32 | Original / 128 | Candidate / 128 |
| --- | ---: | ---: | ---: | ---: |
| 1024 original queries, ms | 472.5512 | 553.1999 | 414.5195 | 320.1435 |
| 8192 extended queries, ms | 5645.5811 | 6441.6989 | 5121.4272 | 4083.1746 |

Medians include candidate domain preparation; common range preparation and
V transpose are another 4.2805/6.7683 ms respectively. Each route has one
warmup and three rotated complete samples. Every candidate128 sample is
faster than every original32 and original128 sample. Candidate32 regresses.
All compared raw surfaces and completed outputs are bit exact, with identical
PV candidate counts. Each route matches 4194304 original GB10 context cells;
the extended 7168 rows have original-arithmetic coverage only.

These component results support separately gated provider integration and a
real-model comparison. They do not change the qualified q8192 median,
package or release status. [Native commands, failures and results](../benchmarks/correctness/long-attention-pipeline-native-components-20260918.json):
354770 bytes, SHA256
`9d2d0094bf7bc175b99a3484e5972306ff77c704d86fabaf218d95040d24d8e9`.

## Current stack at 16k, 2026-09-18

The same qualified whole `ddacdc9`, CK `1c2770d`, MoE `9235750`,
FLA `7b20c90` and CLI `24c4304` run the real 16384-token cold owner on
baiying. All 32 original GB10 continuation IDs, prompt IDs and actual callbacks
match; first token is 16 and raw logit 25.625, error 0. Load is 21371.6083 ms,
instrumented TTFT 86113.7805 ms and TPOT 128.642813 ms. Only the existing
cold chunk option, five profiling flags and full markers differ from the
q8192 control. This single profile is not an uninstrumented speed comparison.

The two complete 8192-token chunks preserve their original state and KV
owners. Final-query/output liveness correctly stays inactive inside both
transactions. Ten first-chunk CK calls total 6516.5173 ms; ten second-chunk
calls total 36526.6208 ms. The latter contains 11745.5448 ms QK,
3760.1110 ms probabilities, 7301.9270 ms native PV and 13160.9654 ms exact PV.
The complete second chunk takes 57750.3 ms. These CK intervals are nested
inside full attention. Allocation, H2D and free clocks remain milliseconds;
pooling does not address the dominant measured work. Negative GPU event
intervals remain invalid and are excluded from totals.

This validates only the cold 16k owner and 32-token continuation, without a
prefix hit, 512-token continuation or larger-context qualification. The
qualified q8192 median remains 23353.80795 ms and all performance/release
targets remain open. [Complete profile evidence](../benchmarks/correctness/narrow-stack-long-completed-profile-20260918.json):
218711 bytes, SHA256
`2a68f76e32a303defd218a6b29ada8ed3ff435708e0a0d92a19eac6ba0f55dce`.


## Current narrow-QK stack profile, 2026-09-18

A completed real q8192/out512 profile on the qualified narrow-QK and final
query/output stack preserves all512 GB10 IDs, original prompt IDs, actual
callbacks and first logit10.375. All binaries and numerical options match the
retained enabled control; only five profiling flags and marker filtering change.
Load is21344.8137ms and instrumented TTFT is27321.9776ms. This is not a new
performance baseline; the uninstrumented median remains23353.80795ms.

Linear projections/core total4362.125/7165.56ms, including3327.918ms recurrence,
1970.7119ms output projection and740.014ms convolution. Nine full-query CK
calls total5477.5099ms:2317.5885ms QK,1446.7542ms probability/nativePV and
1539.6902ms exactPV. MoE totals4968.8112ms; routed/shared intervals overlap.
Dense correction3170.529ms, coarse OUT1176.7438ms and adaptive linear OUT
1820.2961ms are nested work, not additional totals. Nine negative residual
GPU intervals remain recorded as invalid and are excluded. Original dense
and coarse candidate identities/counts match the uninstrumented control.

The completed GDN stream and narrow-half projection experiments below keep
the qualified runtime unchanged. The profile identifies work for broader
provider scheduling; its instrumentation does not establish a new baseline. [Complete profile evidence](../benchmarks/correctness/narrow-stack-completed-profile-20260918.json):
641793bytes, SHA256
`523b2e7ee26b8b71626635f7f86157d54b9e5b38aca6310645e85cfd696416df`.

## Exact narrow-domain QK product, 2026-09-18

CK `1c2770d` preserves the original ordered QK arithmetic while removing
per-group exceptional checks only for complete tiles whose actual operands
satisfy the proved finite domain. Four same-DLL OFF/ON/ON/OFF model runs
pass all 2048 original GB10 IDs, prompts, first logits 10.375/error 0, actual
callbacks and every owner/host check. Whole `ddacdc9`, MoE `9235750`,
FLA `7b20c90`, CLI `24c4304` and all numerical coefficients remain fixed.

OFF/ON TTFT medians are 24087.2361/23353.80795 ms, with both ON samples below
both OFF samples. OFF ranges 23805.0149–24369.4573 ms; the median difference
is an observation, not a universal savings estimate. ON is the current
experimental control, with 589832 additional workspace bytes. Its load
median is 21299.42725 ms. No decode gain, broader prompt/context qualification
or release acceptance follows. Defaults/package remain unchanged; the
10000 ms and retained 4187.415605 ms targets remain open.
[Implementation, all four timings and complete evidence](NARROW_DOMAIN_QK.md).

## Final-layer output liveness, 2026-09-18

Whole source `ddacdc9` computes only the last 2048 OUT values inside the
qualified final-query scope, using the original Q1 K16/width26 kernel and BF16
endpoint. Complete QKV/KV capture, residual normalization and MoE remain.
Four fresh same-DLL q8192/out512 processes change only
`QRT_QWEN36_FINAL_QUERY_OUTPUT_LIVENESS`; attention liveness stays enabled.
CK `df2ea51`, MoE `9235750`, FLA `7b20c90` and CLI `24c4304` are fixed.

| Order / mode | Load ms | TTFT ms | TPOT ms |
| --- | ---: | ---: | ---: |
| 1 / OFF | 21682.8986 | 24160.1204 | 101.450072 |
| 2 / ON | 21333.5847 | 23890.3918 | 100.741730 |
| 3 / ON | 21228.7442 | 23914.4916 | 101.094719 |
| 4 / OFF | 21263.0032 | 24160.9340 | 100.675371 |

All 2048 original GB10 IDs, prompts, actual callbacks and first logits 10.375
pass, as do every activation and host check. OFF/ON median TTFT improves
24160.5272 to 23902.4417 ms, saving 258.0855 ms (1.0682%). Both ON samples
are below both OFF samples. ON became the preceding experimental control. Load
median is 21281.16445 ms; no decode gain is established. The output change
adds no workspace. Attention liveness still uses 83886080 bytes of staging.

The last coarse OUT call disappears; all 130 dense, first nine coarse and
30 adaptive-linear candidate counts match. Defaults and package remain
unchanged. Other prompts, context and release gates remain open; TTFT still
exceeds 10 seconds and the retained 4187.415605 ms target is unchanged.
[Implementation and limits](FINAL_QUERY_LIVENESS.md).
[Four complete model runs](../benchmarks/correctness/final-query-output-product-20260918.json):
1542773 bytes, SHA256
`26efccafbce32f6d3bb2bb44a05b86341cb40cb46e9b92bb6f0a759f2f5619d3`.

## Final-layer query liveness, 2026-09-18

Whole source `2899246` preserves complete final-layer QKV/KV capture and
computes only the last attention query when the cold q8192 caller explicitly
consumes one last row. The existing terminal-prefill route failed462/512
outputs; this new scoped route passes all2048 original GB10 IDs across fresh
same-DLL OFF/ON/ON/OFF processes, with every prompt, first logit10.375,
actual callback and host/owner check. CKdf2ea51, MoE9235750, FLA7b20c90 and
CLI24c4304 remain fixed.

| Order / mode | Load ms | TTFT ms | TPOT ms |
| --- | ---: | ---: | ---: |
| 1 / OFF | 21563.5124 | 24644.0982 | 101.759253 |
| 2 / ON | 21436.2512 | 24316.0438 | 100.852802 |
| 3 / ON | 21381.3563 | 24258.9025 | 101.484683 |
| 4 / OFF | 21319.1887 | 24594.1967 | 100.208110 |

OFF/ON medians24619.14745/24287.47315ms improve331.6743ms (1.3472%);
both ON samples are below both OFF samples. ON became the preceding experimental
control, with an additional83886080-byte suffix staging allocation. All loads
stay below30seconds. No decode gain is established. Defaults/package remain
unchanged and the10-second, retained-performance and context gates remain open.

The downstream OUT owner selects16775587 candidates on zeroed dead contexts,
versus3034571 in OFF; its first-pair completed time rises108.9004 to297.0241ms.
The130 dense, first9 coarse and30 adaptive-linear counts otherwise match.
The later original-K16 last-row OUT comparison is recorded above.
[Implementation and limitations](FINAL_QUERY_LIVENESS.md).
[Four complete model runs](../benchmarks/correctness/final-query-liveness-product-20260918.json):
1542704 bytes, SHA256
`1cb8d1b7e81e475424ffd6185dcf1e56c8b4961bda175fe1604fd9536becd375`.

## Complete GDN provider pipeline, 2026-09-18

Source `cf86fef` separates complete segment preparation, ordered state and
output using three independent storage slots and HIP events. Thirteen native
probes and the host ownership/seed/tail checks pass. The memory-only revision
fixes the exported query to include all179945472 extra bytes; all32 GPU kernels
retain identical executable sections and metadata across that revision.

Four fresh same-DLL model runs pass all2048 original GB10 IDs, prompts, first
logits10.375/error0, actual callbacks and every owner check. OFF/three-stream/
two-stream/OFF TTFTs are23273.0217/23338.3490/23468.8446/23278.0577ms.
Both enabled samples are slower than both controls. All30 layers activate,
and all dense/coarse/adaptive/MoE candidate identities match. Loads stay below
21391ms. Keep the schedule disabled and preserve the23353.80795ms qualified
control; no decode, other-context, package or release gain is established.

[Ownership, native evidence and complete model runs](PIPELINED_GDN_PROVIDER.md).

## Compact matrix QK with wave fallback, 2026-09-18

Source `7c9628c` retains exact ordered carries with 60-byte lossless rows,
four IU8 matrices per K16 group and either direct or register-shuffle fallback.
Host ASan/UBSan and all320 native configurations pass, including forced
original scheduling. q7169/q8192 preserve every original attention surface,
candidate count, metadata and guard, plus29364224 original GB10 context cells
per variant. The repeated1023 rows have original-arithmetic coverage only.

Complete attention plus route preparation regresses382.2061 to796.6028/
709.925ms at q7169 and540.3715 to1102.133/989.6595ms at q8192. Every candidate
sample exceeds every control. Static VGPR use falls167 to71/73 without spills,
but the complete operation slows. Keep the source isolated; pursue broader
scheduling or arithmetic replacement. The23353.80795ms model control and all
open mission gates remain unchanged.

[Design and complete evidence](COMPACT_MATRIX_QUEUE_QK.md).

## Captured QK integer-domain and submission audit, 2026-09-18

Host-only source `cf1f046` samples65536 deterministic causal QK dots from the
pinned original capture, extending7169 inputs to8192 with1023 repeated rows.
All1048576 original ordered groups are narrow;850223 support lossless signed16
operands. The older upper-alignment certificate admits712488 groups, while
exact paired alignment admits714728 with no product remainder; all admitted
raw carries match. A cheaper minimum-trailing-row test admits only444587.
Only402 complete dots stay in the exact matrix domain throughout. This is
sampled operand/CPU evidence, not native throughput or model acceptance.

The qualified uninstrumented q8192 control contains130 dense owners,290 host
count reads and1141 exact dispatches. Dense allocation/free totals20.132/
2.2305ms; pooling alone cannot explain seconds of TTFT. Correction3314.888ms,
GDN state923.02241ms and WU/output1284.76680ms include GPU work and must not
be treated as additive host overhead. No negative intervals occur in these
selected markers. Existing completion batching remains a prior, not a route
restriction. A matrix replacement needs compact metadata and efficient
mixed-group execution; changing only the admission test offers little gain.

[Complete audit](../benchmarks/correctness/qk-integer-domain-host-audit-20260918.json),6062bytes, SHA256
`8a0898f647ab43bda61185771dd2304180d461abf66ebb2e5b558027553e8475`.
The23353.80795ms qualified model control and open mission gates are unchanged.

## Whole-row narrow half projection, 2026-09-18

Source `0714cea` keeps the retained four-lane lossless-half schedule and
ordered K16 arithmetic, but preclassifies complete operand rows before using
normal FP32 carries without per-group exceptional checks. All28 native
configurations and2645036 carry endpoints pass. Captured QKV/OUT preserve
all67108864/16777216 GB10 BF16 outputs,4331635/8471989 raw selected values,
inactive cells, row flags, prepared words and guards after every attempt.

QKV preparation/classification/replay medians regress42.7773ms to49.2372/
48.1558ms; original midpoint OUT improves157.272ms to143.006/142.877ms.
QKV admits3458934 selected dots; OUT admits8469953. This is not the retained
coarse-OUT owner and no MoE or model gain follows. Keep the component isolated;
the qualified model median remains23353.80795ms.
[Implementation and complete evidence](NARROW_HALF_PROJECTION.md).

## GDN segments on independent streams, 2026-09-18

Isolated source `1bf6755` retains the exact paired-score/WU/shared-arena
state8/output kernels and overlaps independent1024-token segments using
HIP events. State updates remain ordered. All288 safety configurations and
every original q7169/extended q8192 score, output and state boundary pass.
The initial safety run exposed a missing join to the default-stream completion
helper; both the failure and corrected dependency remain recorded.

At q8192, serial/two/three-stream medians are86.7021/78.5150/77.3428ms with
U=V and86.3189/77.4624/78.2368ms with separate U. All candidate samples beat
their corresponding controls. The fixture includes all four stages and event
operations but supplies normalization, gate scans, KKT and inverse inputs.
Whole-invocation intermediate storage differs from the production single-segment
owner. Keep the result as an isolated building block, without promoting a
model TTFT or release result. [Design, scopes and complete evidence](PIPELINED_GDN.md).

## Sparse byte-metadata projection, 2026-09-18

Source `8ae7879` combines original candidate tile reuse with the proved
wide-dot byte arithmetic. All host tests,216 native configurations and
complete original/GB10 operator boundaries pass. Preparation plus replay
regresses42.4972 to115.753/256.317ms on retained matrix4 QKV. The original
midpoint OUT owner also regresses156.823 to272.651/325.829ms, with every tile
taking dense fallback; that comparison does not measure current coarse OUT.
Keep both schedules isolated and preserve the23353.80795ms model control.
[Arithmetic, coverage, clocks and evidence](SPARSE_BYTE_PROJECTION.md).

## Checked narrow QK lookahead, 2026-09-18

Source `774e5e8` prepares two or four K16 groups from exact byte metadata,
using native partials only to predict alignment. Every plan checks the actual
original carry before use. All host tests, 400 native configurations and
complete q8192 original/GB10 boundaries pass, including 178502656 deliberately
rejected scores. However, attention including route preparation regresses
543.8702 to 597.8399/637.3632 ms. The same-run byte 4x4 route takes 519.6112 ms.
Keep both schedules outside dispatch and preserve the 23353.80795 ms model
control. [Implementation, exact scope and evidence](NARROW_QK_LOOKAHEAD.md).

## Byte-exponent exact QK component, 2026-09-18

Source `5f0fcc1` obtains exact K16 alignment scales from packed exponent
deficits, then consumes each product immediately. The original narrow-domain
proof, general fallback, probability/PV consumer and candidate set remain.
All host checks, 240 native configurations and complete q8192 original/GB10
boundaries pass. Attention including route preparation improves 534.6539 to
522.8729/510.6473 ms for 2x4/4x4 layouts. The best 24.0066 ms saving is a
modest component gain; keep it isolated while seeking a broader reduction
in arithmetic. No provider/model run or baseline change follows.
[Implementation, memory cost and evidence](BYTE_EXPONENT_QK.md).

## Query-row exact PV scheduling, 2026-09-18

Source `d2a275a` groups the original selected columns by query/head and shares
probability inputs through K128 shared staging or wave broadcasts. All 288
generated cases, original raw surfaces and complete captured GB10 context
boundaries pass. Complete q8192 PV medians including V transpose regress
268.6239 to 334.8396/404.4658 ms; q7169 also regresses. Candidate collection,
row reset/scatter and complete exact replay are included. Keep both schedules
outside dispatch and preserve the 23353.80795 ms qualified model control.
[Implementation, clocks and evidence](ROW_SHARED_PV.md).

## Exact narrow-domain GDN components, 2026-09-18

Source `a77c043` extends the proved QK finite-domain carry to GDN scores,
WU, recurrent state and output, preserving the retained shared arenas and
both U ownership modes. All 336 generated cases, every captured original
arithmetic comparison and all available original GB10 boundaries pass.
Complete q8192 production U=V medians are 83.8103 ms for the retained
control, 86.0711 ms for narrow matrices and 84.9590 ms with narrow scores
as well. No gain is established; keep both routes outside dispatch. The
qualified model baseline stays 23353.80795 ms. No model timing or release
acceptance follows. [Implementation, all timings and evidence](NARROW_DOMAIN_GDN.md).

## Exact narrow-domain QK components, 2026-09-18

Complete q8192 attention in source `397ddc8` improves from 605.9593 ms to
546.1894 ms including original-input classification. Both revisions pass
288 generated cases, every raw captured score/output comparison and all
29364224 original GB10 context values. The selected 2x4/K64 layout preserves
ordered K16 arithmetic and original fallback. This is a component result;
the later same-DLL model qualification is recorded above. [Implementation, all layouts and evidence](NARROW_DOMAIN_QK.md).

## Inline attention-probability repair, 2026-09-18

Isolated source `878d8c0` combines maximum/probability repair and denominator
interval construction. All 80 generated configurations, independent initial
probability checks and complete captured GB10 boundaries pass. However,
130495 of 131072 rows still need full remaining-score fallback. Final original
score work is 99.7533% of the control and complete q8192 attention regresses
588.7369 to 1406.7957 ms. Keep this route outside dispatch and preserve the
23902.4417 ms qualified model baseline. No model run or release claim follows.
[Dataflow, counts and evidence](INLINE_PROBABILITY_REPAIR.md).

## Probability/PV ownership comparison, 2026-09-18

Isolated source `ab6b9ce` preserves every tested native and captured GB10
boundary but regresses complete q8192 attention from 586.8079 ms to
1267.8185 ms (16 queries/256 threads) or 1091.4214 ms (32 queries/512 threads).
All 120 generated cases pass, including native surfaces before exact replay.
Keep both candidates outside dispatch. This is a component result, with no
model run or change to the 23902.4417 ms qualified product baseline.
[Measurements and complete evidence](REDISTRIBUTED_PROBABILITY_PV.md).

## Producer-local gate replay has no established product benefit, 2026-09-18

Source `932b55b` moves original gate/up selection and K16 correction into the
completed-up matrix tile, reusing existing shared storage. Generated q8192
improves 123.521191 to 112.554649 ms, with every raw FP32 and BF16 comparison
passing. Four same-DLL real-model processes also pass all2048 original GB10
IDs, prompts, logit10.375/error0, actual callbacks and host/owner checks.

OFF/ON TTFT medians are23829.64085/23800.3680ms. The29.27285ms reduction
(0.1228%) has overlapping observed ranges, so no stable or material product
benefit is established. Keep the option off and preserve the23902.4417ms
qualified baseline with MoE9235750. No performance target, package or release
state changes. [Implementation, all timings and evidence](MOE_PRODUCER_GATE_REPLAY.md).

## Classified MoE replay rejected for performance, 2026-09-18

Source `078ef52` fuses Row36 preparation with row classification and orders the
original candidates by arithmetic class and expert. Four same-DLL q8192/out512
OFF/ON/ON/OFF runs pass all 2,048 GB10 IDs, prompt and logit boundaries, actual
callbacks and host checks. OFF/ON median TTFT is 23890.02305/24013.59630 ms;
both ON samples are slower than both OFF samples. The 123.57325 ms regression
rejects this performance route despite lower static VGPR counts. All loads
remain below 30 seconds. Keep the 23902.4417 ms qualified control with original
expert-only ordering and MoE `9235750`; defaults and package remain unchanged.
[Implementation, native checks and four product runs](MOE_CLASS_EXPERT_REPLAY.md).

## Native RZ tree QK rejection, 2026-09-18

CK source `0cc0459` replaces per-product integer alignment with a native FP32
RZ tree only under an explicit experiment flag. Independent host/GPU scalar
addition checks and all native tree, fallback, memory and wave-mode checks
pass. The original arithmetic's score bits are not an acceptance gate.

The actual q8192/out512 control passes all 512 GB10 IDs at 23890.9927 ms TTFT.
Enabling the tree gives 23100.9178 ms but fails 451 output positions, starting
at index 1 (expected 255, actual 244). First token 144/logit 10.375 and actual
callbacks pass in both runs. Reject the candidate timing because its original
GB10 continuation fails; keep the qualified 23902.4417 ms baseline and the
option disabled. Other providers, both terminal liveness flags and all
numerical thresholds are identical between arms.
[Implementation and native checks](RZ_TREE_QK.md).
[Two complete model runs](../benchmarks/correctness/rz-tree-qk-product-rejection-20260918.json):
801237 bytes, SHA256
`1fb39c12224971d03213277e077b07f6e2695fb90a4213b7ae56750bb695663f`.

## Compact integer QK component, 2026-09-18

Source `eac6932` reduces the isolated signed16 dot4 row from 156 to 52 bytes,
with exact BF16 reconstruction and original K16 carries. Every native raw
carry, all 72 generated configurations and 6543114240 captured score
comparisons pass. Complete q8192 control/K64/K128 times are
303.1695/2383.5624/2302.1050 ms. Both candidates remain isolated.

Compiled LDS is 46080/59392 bytes with 8 bytes of private storage, exceeding
explicit row storage by 32768 bytes. The captured host sample takes original
fallback for 12238/65536 groups. A packed fallback alone leaves the extra
storage unchanged. Explicit independent carries at `8310180` remove it:
LDS becomes 13312/26624 bytes, with zero private storage. Every original
numerical check passes again, but full q8192 control/K64/K128 times remain
313.6175/1427.3859/1521.9112 ms. Both candidates stay isolated; no model
baseline or default changes. [Explicit-carry evidence](../benchmarks/correctness/compact-integer-qk-explicit-carries-20260918.json).
[Implementation, compiler failure and scope](COMPACT_INTEGER_QK.md).
[Complete evidence](../benchmarks/correctness/compact-integer-qk-native-components-20260918.json):
161520 bytes, SHA256
`b4c4e2b74a85c8c9b9676463ec98d9f5f6575ff6c405a9b0903f2aea8e9312f0`.

## Paired GDN scores and shared arenas, 2026-09-18

Four fresh same-DLL OFF/ON/ON/OFF runs from FLA source `7b20c90` pass all
2,048 original GB10 IDs, prompts and callbacks, with first logit 10.375 and
zero error. Whole `6e4908b`, CK `df2ea51`, MoE `9235750`, CLI `24c4304` and
all AOT/table assets remain fixed. Every ON run activates 240 segments for
each of scores/state/output; all other owners and host guards pass.

OFF/ON TTFT medians are 24,972.6823/24,709.6139 ms, saving 263.0684 ms
(1.0534%). Both ON samples are below both OFF samples. Mode 1 is the current
experimental control before final-query liveness, with no new device workspace. All loads stay below
30 seconds. TPOT medians are 100.4114965/100.5125505 ms, without evidence of
a decode gain. Code/package defaults remain off; TTFT still exceeds 10 seconds
and all long-context, retained-performance and release gates remain open.
[Implementation, component limits and each product measurement](PAIRED_SCORE_GDN.md).
[Product evidence](../benchmarks/correctness/paired-score-gdn-product-20260918.json):
1269463 bytes, SHA256
`e416abed136f64ed0c9ab8e01d34b3473c9a38695cde36a829f2f884dfe1b8c3`.

## Completed current-stack phases, 2026-09-18

The paired-GDN control passes a fresh q8192/out512 profile with every original
GB10 ID, prompt, actual callback and first logit 10.375. Artifacts and numerical
options match the retained run; only five profiling flags and marker filtering
change. Load is 21283.6076 ms; instrumented TTFT is 28663.5245 ms and TPOT
104.155783 ms. This does not replace the 24709.6139 ms ordinary baseline.

| Surface | Completed time ms |
| --- | ---: |
| Linear core | 7146.189 |
| Preceding linear projections, separate from core | 4355.962 |
| Recurrent work, within linear core | 3314.694 |
| Linear OUT, within linear core | 1963.548 |
| Full attention, outer interval | 9147.449 |
| CK attention, within full attention | 6793.513 |
| QK, within CK | 3198.092 |
| Fused softmax/native PV, within CK | 1663.966 |
| Exact PV, within CK | 1739.224 |
| MoE total; routed and shared overlap | 4953.601 |

Nested and overlapping clocks must not be summed. All 240 linear phases,
ten CK profiles and 40 MoE profiles complete. Every paired-GDN owner and all
host guards pass; dense/coarse candidate counts match the ordinary control.
The CK approximate-PV field remains an empty completion boundary, totaling
0.3710 ms. All ten full-attention residual device intervals are negative;
their original values remain recorded and excluded from totals. No MoE
interval is negative in this run.

Layer 39 full-prefix attention takes 894.791 ms. Later final-layer query/output
liveness comparisons are recorded above with complete KV and original arithmetic.
This older profile does not describe those changes. Broader changes remain necessary
for the unchanged 10-second gate. [Complete same-artifact profile and GB10
boundary](../benchmarks/correctness/paired-stack-completed-profile-20260918.json):
637135 bytes, SHA256
`265781252661a42bb009c5b145423680dca450bc66bfe93777f94487355d029e`.

## Fused probability/native PV, 2026-09-17

Four fresh same-DLL q8192/out512 processes compare the source `df2ea51`
CK provider OFF/ON/ON/OFF. The retained whole, compact MoE, FLA and CLI
assets and every other option are fixed. All 2048 original GB10 output IDs,
prompts and actual callbacks match; every first logit is 10.375 with zero
error. All loads are below 30 seconds and every owner/host guard passes.

OFF/ON TTFT medians are 25376.85035/24835.3024 ms, saving 541.54795 ms
(2.1340%); both ON samples are below both OFF samples. Mode 1 became the
preceding experimental q8192 baseline. It retains independently parallel exact
QK and fuses only softmax/native PV, followed by the original selective exact
replay. It adds no persistent workspace. Two samples per arm do not establish
a decode gain: TPOT medians are 101.276713/101.1726645 ms. TTFT still exceeds
10 seconds. Code/package defaults, long-context and release gates remain
unchanged. [Implementation and all four measurements](STREAMED_EXACT_ATTENTION.md).
[Product evidence](../benchmarks/correctness/fused-probability-pv-product-20260917.json):
965116 bytes, SHA256
`9963b45485de6f944d2f5fadc1c5f588fa64e0c30f128359e019d3551e6a2758`.

## Completed phases before paired GDN integration, 2026-09-17

The same qualified fused-PV1, compact-MoE1 and adaptive-OUT2 artifacts pass
all 512 original GB10 tokens, prompts, callbacks and first logit 10.375.
Only five profiling flags and marker filtering differ from the ordinary
control. Load is 21343.2126 ms; instrumented TTFT is 29051.1805 ms and TPOT
104.150362 ms. These include profiling overhead and do not replace the
24835.3024 ms uninstrumented baseline.

| Surface | Completed time ms |
| --- | ---: |
| Linear core | 7518.854 |
| Preceding linear projections, separate from core | 4349.800 |
| Recurrent work, within linear core | 3571.606 |
| Linear OUT, within linear core | 1992.610 |
| Full attention, outer interval | 9180.172 |
| CK attention, within full attention | 6830.711 |
| QK, within CK | 3199.114 |
| Fused softmax/native PV, within CK | 1703.656 |
| Exact PV, within CK | 1736.810 |
| MoE total; routed and shared overlap | 4941.788 |

Do not sum nested or overlapping intervals. All 240 linear phases, ten
complete CK profiles and 40 MoE profiles are present. The old CK
`approximate_pv_ms` field is now an empty completion boundary (0.4324 ms
across ten calls); native PV is included in `probability_ms`. Its individual
field is not comparable with the older unfused softmax field. Dense/coarse
candidate counts match the ordinary control. Nine full-attention residual
device intervals and one MoE input interval are negative, retained as invalid
and excluded from totals; they do not invalidate the passing token boundary.

This profile precedes the paired GDN replacement. Its attention artifacts
remain in the current control. The subsequent [lossless scaled-half four-score QK
experiment](MICROTILE_HALF_QK.md) preserves all checked arithmetic but reduces
complete q8192 QK only302.3008 to284.7096 ms; it remains isolated.
[Pinned artifacts, options, callbacks and completed phases](../benchmarks/correctness/fused-stack-completed-profile-20260917.json):
489965 bytes, SHA256
`541bb4a3020c3e100db873555685ef37506911b6e8579b2d0ca2e70c8f30e477`.

## Certified K16 grids for scalar QK, 2026-09-17

Isolated source `e34dc05` reuses per-row exponent and binary-unit bounds
inside the current 2x2 scalar QK schedule. A dominating carry gives the
original alignment grid directly; a higher grid is admitted only when every
product and carry remains exact. Other groups retain the original paired
exponent scan, and unsupported endpoints retain complete original-dot replay.
No product dispatch changes.

The sanitized host test checks 2097152 ordered K16 groups against the wide
integer reference. All 48 generated GPU comparisons, 3072 independent CPU
dots and 65696416 score cells pass. The complete q8192 captured geometry
checks all 545259520 score slots on every warmup and measured attempt in
both arms, plus 512 independent CPU dots. Every raw bit, prepared metadata,
immutable input, redzone and unused tail passes. As in preceding component
tests, q8192 repeats 1023 original q7169 rows and is not a model prompt.

| Route | QK and fallback ms | Preparation ms | Total ms |
| --- | ---: | ---: | ---: |
| Retained 2x2 | 297.6979 | 9.4226 | 307.1205 |
| Certified grids | 492.7468 | 10.1107 | 502.8575 |

Keep the candidate isolated. Three rotated samples establish a substantial
regression despite exact numerical agreement. Compiler metadata declares
117/104 VGPRs, 32768/34816 shared bytes and zero private bytes for the two
score kernels; these are allocation metadata, not measured occupancy.
[Pinned source, commands and all numerical boundaries](../benchmarks/correctness/certified-group-qk-native-components-20260917.json):
128957 bytes, SHA256
`21f9908174f0cae3696f5fa621d7c103000995e8cc6f6042fba93716288b7519`.

## Complete convolution-consumer observation, 2026-09-17

The default-off observer in whole source `0e5bf7d` leaves every original
QKV correction in place and measures potential omission across all 30
linear layers. A fresh q8192/out512 run passes all original GB10 prompt and
output IDs, actual callbacks and first logit 10.375 with zero error. Every
observed selected endpoint interval and constant convolution output passes.
It identifies 38931451 provisionally omittable corrections out of 160641307
(24.2350%). The remaining tiny-input guard is preserved.

Load is 21241.5679 ms; TTFT is 26121.1392 ms and TPOT is 101.348289 ms.
These clocks include private copies, certificates, duplicate convolution,
reductions and synchronization. They are not a performance comparison or a
replacement for the then-current uninstrumented 25379.41695 ms baseline. Keep the
observer off and leave filtering unimplemented while prioritizing a broader
attention route. [Implementation, component checks and model evidence](CONV_CONSUMER_INTERVAL.md).

## Fused MoE down consumer collection, 2026-09-17

Four fresh same-DLL q8192/out512 processes run OFF/ON/ON/OFF with adaptive
linear OUT mode 2, combined exact attention and register PV held fixed.
All 2048 GB10 output IDs, original prompts and actual callbacks match; all
first logits are 10.375 and all loads are below 30 seconds. The new MoE
provider source is `9235750`; the other component builds remain pinned.

OFF/ON TTFT medians are 25610.7622/25379.41695 ms, saving 231.34525 ms
(0.9033%). Both ON samples are below both OFF samples. Each of 40 enabled
calls applies the previous complete BF16 consumer certificate while
publishing remaining candidates directly into the original bounded queue.
It removes the omission bitmap and duplicate selector scan; 126904922 of
158740988 down replays are omitted, with unchanged exact replay for the rest.
The per-invocation counter/guard owner is 1056 bytes. Native generated tests
also exercise the actual runtime launcher and all four comparison paths.

Retain only the default-off experimental prefill setting. TPOT medians are
100.942518/102.3135815 ms, so no decode or total-request gain is claimed.
TTFT remains above 10 seconds, and the empirical error envelope, long-context
and release gates are unchanged. [Implementation and full measurements](MOE_DOWN_CONSUMER_COMPACT.md).
[Product evidence](../benchmarks/correctness/moe-down-consumer-compact-product-20260917.json):
884564 bytes, SHA256
`8ae8646fb5c93866c7e7c3588e26290c0fd5223c07bee104d01d208a0f1649cf`.

## Cached weight controls for exact replay, 2026-09-17

Isolated source `110174d` keeps original BF16 weights and caches only the
4-byte scale/nonzero control for each K16 group. Replay reconstructs the
same FP16 payload on demand and calls the unchanged staged accumulator.
The exhaustive sanitized host check covers all 65536 BF16 patterns and
16777216 encoded pairs. Eighteen native cases match 2080134 ordered raw
K16 states, including 622074 original fallback groups. Fourteen invalid
views are rejected; guards, immutable operands and audit parity pass.

Complete generated MoE owners include activation preparation and replay.
The control also includes full weight preparation; the cached variant
excludes its separately measured initial cache construction.

| Geometry | Original ms | Cached controls ms | Per-call controls ms |
| --- | ---: | ---: | ---: |
| Gate/up: 262144 weight rows, K2048, 8192 inputs | 132.0360 | 174.4945 | 177.9072 |
| Down: 524288 weight rows, K512, 65536 routed inputs | 21.2069 | 24.1325 | 26.5251 |

One warmup and three rotated samples verify all selected raw outputs,
all control/prepared words and 512 independent CPU dots across the shapes.
The queues and values are synthetic. Initial cache construction takes
6.9279/2.7965 ms for the two shapes; 40 layers would require 8053063680
control bytes by allocation arithmetic, without a measured model peak-memory
or load result. Keep this route isolated: both complete owners regress even
with an existing cache. No provider, model token, GB10, package or release
qualification follows. [Source, commands and measurements](../benchmarks/correctness/weight-control-projection-components-20260917.json):
123596 bytes, SHA256
`bedad5d042abd29eaf0d4dc7a5128570919042fd2b685d8ae3859af73ff16b65`.

## Queued linear OUT on the combined-attention stack, 2026-09-17

Four fresh same-artifact q8192/out512 processes, ordered OFF/ON/ON/OFF,
retain combined exact attention and register PV throughout. The only option
that differs is `QRT_QWEN36_Q8192_LINEAR_OUT_VARIANCE_REPLAY=0/2`.
All 2048 original GB10 output IDs and actual callback IDs match; every prompt
and first logit 10.375 pass. All loads are below 30 seconds.

| Order / mode | Load ms | TTFT ms | TPOT ms |
| --- | ---: | ---: | ---: |
| 1 / OFF | 21281.5439 | 26248.7864 | 101.008479 |
| 2 / ON | 21265.8075 | 25530.0511 | 101.539822 |
| 3 / ON | 21340.1467 | 25694.8428 | 101.071900 |
| 4 / OFF | 21300.1796 | 26401.5913 | 100.847431 |

OFF/ON TTFT medians are 26325.18885/25612.44695 ms, a 712.7419 ms
reduction (2.7075%). Both ON samples are below both OFF samples. Each ON
process activates all 30 linear layers and skips 29191574 of 58489767
original selected corrections (49.9089%). Remaining dense and coarse
attention counts match. Certification remains conditional on the original
producer envelope; observed violations force original replay, and none are
reported here. GB10 tokens remain the correctness authority.

Retain mode 2 as the next experimental q8192 baseline with whole `6e4908b`,
CK `cf0f889`, and the same MoE/FLA/CLI assets. Keep code and package defaults
unchanged. Two samples per mode provide a limited repeated comparison;
OFF/ON TPOT medians are 100.927955/101.305861 ms, with no decode speedup
claimed. Additional adaptive workspace is 170205188 bytes, separate from
the retained attention table's 82182144 bytes; neither is total peak memory.
TTFT remains above 10 seconds. Prefix, long-context, HTTP package and release
gates remain open. [Commands, artifacts and all four boundaries](../benchmarks/correctness/out-variance-combined-product-20260917.json):
795207 bytes, SHA256
`808b651f4c4651dd0b5c64a5e18e65ea9e94287bc96916d1559c0750b4198dfb`.

## Shared exact GDN gate values, 2026-09-17

Source `e320f47`, with observer correction `31b2e93`, prepares two exact FP32
exponential values per token/head for WU, state8 and output. Original table,
ordered dots, BF16 boundaries, final FMAs and U ownership remain unchanged.
All 96 generated safety comparisons and all q8192 attempts match every raw
output, final state, W/U, checkpoint and residual. Prepared values match CPU
lookups; guards, immutable inputs and seven invalid-call rejections pass.

Three rotated complete-owner samples include the extra preparation. With
production-style `U=V`, original/prepared medians are 80.6180/74.4829 ms;
with separate U they are 77.4745/82.4624 ms. Samples overlap. Keep the route
isolated: the gain is modest and depends on ownership. q8192 inputs are
generated and its full-length state call differs from production 1024-row
segments; no model tokens, GB10 boundary or TTFT are newly qualified.
One invalid device phase series is retained with a null median and excluded
from phase conclusions. Complete host measurements and all numerical checks
finish. The earlier observer failure is preserved in the evidence.
[Commands, checks and samples](../benchmarks/correctness/prepared-gate-gdn-components-20260917.json):
562973 bytes, SHA256
`04de7cf9e0cd865f19e53a6720453dc954beaa52a2f3f2254f91224897208f67`.

## FP32-carry exact PV component, 2026-09-17

Source `55cd4a7` compares current production REGISTER PV with direct BF16 and
predecoded operands carrying each original K16 endpoint in one FP32 register.
Both preserve K32 rescale, reciprocal and original candidate membership.
Rejected rows or arithmetic endpoints restart complete original replay.
All 288 generated comparisons pass, covering 43,794,432 output positions,
empty/all/sparse selections, tails and special values. Operand and arithmetic
fallbacks are exercised. All raw surfaces, CPU recurrence/encoding, guards,
unused tails and immutable inputs pass; 14 invalid calls are rejected.

Complete PV component medians include native production, collection, P
preparation, fast replay, full fallback, V preparation and transpose:

| Tokens | REGISTER control ms | Direct BF16 ms | Predecoded ms |
| --- | ---: | ---: | ---: |
| 7169 | 177.2326 | 161.5373 | 162.2950 |
| 8192 | 277.6303 | 250.8002 | 282.2137 |

One warmup and three rotated samples are checked in full. Both shapes match
all 29,364,224 available GB10 context cells on every attempt and every original
raw output; q8192 repeats the first 1023 captured Q/K/V rows. No model token
loop or TTFT is measured. Keep both routes isolated: the direct route's
26.8301 ms q8192 saving is useful component evidence, while the predecoded
route regresses. This result does not establish a seconds-scale model gain.
The product control at this component measurement was `cf0f889`, median
TTFT 26385.72255 ms; the later combined OUT comparison is recorded above.
Windows build, host guards, local accumulator checks, C smoke and hygiene pass.
[Native commands and evidence](../benchmarks/correctness/f32-carry-pv-replay-components-20260917.json):
330238 bytes, SHA256
`6209538fdb7d45c89aa5cf0a8b970db77d6f06ca7703b12a0157440baffbca56`.

## Fused linear input preparation component, 2026-09-17

Source `deb2fa9` directly produces normalized compact Q/K and V from projected
QKV, preserving BF16 tap products, ordered FP32 sums, exact SiLU/RSQRT tables
and the original normalization tree. All 78 generated configurations pass,
including special values, causal halos and tails. The original convolution
is extracted unchanged from the whole provider; normalization uses the actual
FLA source. Full raw/compact outputs, sampled independent CPU heads, guards,
immutable inputs/tables and production/capture parity pass.

Three rotated complete-operator samples give q7169 medians of 5.4034/3.9577 ms
and q8192 medians of 6.2084/4.4969 ms for original/fused. Both shapes match all
58,728,448 held-out GB10 raw convolution cells; q8192 repeats 1023 captured
projected rows. Clocks exclude common allocation/table loading and contain no
gate or recurrent work. The broader instrumented whole convolution interval
is not interchangeable with this operator clock. Keep the component isolated:
this measured saving does not justify prioritizing provider integration while
TTFT remains 26.386 seconds. Continue work on common exact replay and attention.
No model tokens or product performance are newly qualified. Local contract,
C smoke and hygiene checks, Windows build and host guards pass. Evidence:
`benchmarks/correctness/linear-input-preparation-components-20260917.json`,
188476 bytes, SHA256
`9ad2a77c179ecda8e9eae5ba2b6e6762880c795f49cc99640be58ff677a62f2a`.

## Completed combined-attention stack profile, 2026-09-17

The same qualified `cf0f889` CK DLL and pinned whole/MoE/FLA/CLI assets pass
another original q8192/out512 GB10 boundary with all actual callbacks and
first logit 10.375. Five profiling options plus disabled marker filtering
give a diagnostic TTFT of 30769.9249 ms; synchronization and logging overhead
prevent using that value as retained performance.

Linear core is 8268.664 ms, with preceding projections separately 4384.565 ms.
Within the core, recurrence is 3580.224 ms and OUT is 2930.8333 ms. Full attention
is 9733.701 ms, containing CK 7388.2931 ms: QK 3189.587, probability 985.4753,
native PV 1333.6734 and exact PV 1688.2369 ms. MoE totals 5292.834898 ms;
routed/shared overlap. All 160 dense corrections total 4392.431 ms excluding
their producers; candidate counts match the uninstrumented control.

These nested clocks must not be added. Ten negative residual/postnorm device
intervals remain recorded as invalid and are excluded. Initial EXP creation
is outside the internal CK clock but remains in outer attention and TTFT.
Investigate broader linear projection/recurrent and common exact-replay
changes next. [Complete profile and GB10 boundary](../benchmarks/correctness/exact-attention-completed-profile-20260917.json):
458897 bytes, SHA256
`3c4403e4d10ad4c1b736c24be0f0adf872319ecfa5f03d8752e7400207507c33`.

## Combined exact attention product comparison, 2026-09-17

Source `cf0f889` integrates the 2x2 QK schedule and device-verified native EXP
decoder behind one default-off option. Four fresh same-DLL q8192/out512
processes, ordered OFF/ON/ON/OFF with register PV enabled throughout, match
all 2048 GB10 output IDs and actual callback IDs, original prompts and first
logit 10.375. TTFT medians are 27263.01365/26385.72255 ms OFF/ON, saving
877.2911 ms (3.2179%); both ON samples are below both OFF samples. All loads
remain below 30 seconds. Each ON process builds and verifies the additional
82182144-byte EXP table once, inside actual first-callback TTFT.

Retain ON in the next experimental control, with code/package defaults off.
Native production callbacks also preserve every original q7169 GB10 context
element and raw output file, and the complete local suite passes. The 10000 ms
gate, retained targets, long contexts and release remain open. The completed
profile above directs the next broad investigation.
See [implementation, exact artifacts and four-run evidence](MULTI_SCORE_QK_EXPERIMENT.md).

## Combined exact attention components, 2026-09-17

Source `c3a7c3b` combines2x2 exact QK and the exact native EXP decoder over
the qualified register PV arithmetic. Complete q8192 attention falls from
747.2801 to654.7009ms; including common preparation, table generation and
exhaustive328728576-input validation,751.0101 becomes672.2606ms. All raw
surfaces, original candidate counts and29364224 available GB10 context cells
match. Advance the combined route to a default-off provider experiment.
These captures supply no new model tokens, TTFT or release qualification.
See [four-arm comparison and evidence](MULTI_SCORE_QK_EXPERIMENT.md).

## Shared inputs across exact QK scores, 2026-09-17

Source `cbe357d` preserves all tested original raw scores while its2x2
per-thread schedule reduces captured q8192 QK353.4056 to300.9476ms;
the unchanged deferred control is327.1309ms. All120 generated configurations
and both complete captured geometries pass, including independent CPU dots,
partial tiles, per-cell fallback and guards. Advance it to a combined
attention experiment with exact native EXP correction. No provider change,
model tokens, TTFT or release acceptance follows from these components.
See [scope, clocks and evidence](MULTI_SCORE_QK_EXPERIMENT.md).

## Register PV rescaling product comparison, 2026-09-17

Source `e669940` adds a default-off register-rescale option. Four fresh
same-DLL q8192/out512 processes, ordered OFF/ON/ON/OFF, pass all2048 GB10
output IDs and actual callback IDs, original prompt IDs and first logit10.375.
TTFT medians are27866.6750/27328.8799ms OFF/ON, a537.7951ms reduction;
both ON samples are below both OFF samples. All loads remain below30s.
Retain the enabled option in the next experimental q8192 control, with code
and package defaults unchanged. This limited comparison does not satisfy
the10000ms gate, retained performance or release acceptance. Native integration
also passes336 generated configurations, full captured comparisons and the
repository suite. See [implementation, boundaries and evidence](REGISTER_PV_ARITHMETIC_EXPERIMENT.md).

## Register PV arithmetic components, 2026-09-17

Source `bae2730` replaces volatile accumulator rescaling barriers with an
explicit FP32 register multiply. All 1048576 multiply comparisons, 252
native configurations and both complete captures pass with original raw
outputs, error bounds, candidate membership and available GB10 context.
Complete q8192 PV falls from323.8687 to269.5124ms; q7169 falls from209.0290
to173.2841ms. Shared row reciprocals add only1.3510ms atq8192. Advance the
register-only route to a default-off provider experiment. These captures
measure no model tokens or TTFT and do not change the package or release
status. See [arithmetic, timing and evidence](REGISTER_PV_ARITHMETIC_EXPERIMENT.md).

## Prefix-certified PV replay components, 2026-09-17

Source `3e41f70` preserves all tested BF16 results, original native centers,
error bits and candidate membership. A suffix certificate lets selected
cells stop after an exact prefix; q8192 replay work falls 31.63% / 28.06%
at 512 / 1024 keys. Complete PV remains slower: 329.6917 ms original versus
369.6577 / 336.9665 ms. All 252 generated configurations and both captured
GB10 context comparisons pass. Keep it isolated and investigate common
arithmetic costs. No model tokens or TTFT were measured.
See [certificate, scope and evidence](PREFIX_CERTIFIED_PV_EXPERIMENT.md).

## Globally split probability components, 2026-09-17

Source `7ff6cd1` preserves all tested raw attention surfaces, PV candidate
counts and available GB10 context cells. All 240 native cases and both
complete captures pass. Four global probability stages give q8192 attention
796.8697 ms versus existing staged 786.9070 ms and online 800.9081 ms;
the probability interval itself also fails to improve. Keep the variant
isolated and investigate larger arithmetic/correction costs. No new model
tokens, TTFT or release qualification were measured.
See [scope and evidence](SPLIT_PROBABILITY_EXPERIMENT.md).

## C64 finite-domain recurrence components, 2026-09-17

Source `b99fb00` removes repeated exceptional checks within a proved finite
input domain, preserving the identical original recurrence and all tested
raw centers, intervals and replay work. Host sanitizers, 2097152 direct GPU
states, 95 generated configurations and both complete captured operators
pass. OUT improves from vector control93.8090 to87.3077ms, and C64 QKV from
150.4241 to136.0386ms. Keep it isolated: these component gains do not close
the TTFT gap or qualify a replacement for the retained QKV producer.
See [domain argument and evidence](DOMAIN_COARSE_PROJECTION_EXPERIMENT.md).

## Shared-memory C64 projection components, 2026-09-17

Source `2c020c1` preserves all tested raw centers, error intervals, candidate
masks and original selected outputs. All 75 native safety configurations
and both captured q8192 operators match their independent references.
Shared-memory OUT is at best 96.3518 ms versus vector control 93.6715 ms;
QKV is at best 162.2570 ms versus 151.6550 ms. Keep the three variants
isolated and investigate common arithmetic costs. No model tokens, TTFT or
release qualification were measured. See [scope and evidence](SHARED_COARSE_PROJECTION_EXPERIMENT.md).

## Exact CPU QK partition components, 2026-09-17

Sources `5986f7f` / `6599691` preserve original raw attention and all available
GB10 context cells with exact AVX-512 K16 arithmetic. A paired q8192 layout
comparison reduces 32-worker CPU QK from 4278.4413 to 2043.3010 ms by adding
one cache line between feature rows. Complete attention remains 3606.5788 ms
against the same-executable GPU control's 982.5672 ms. All safety cases and
captured comparisons pass, but no CPU/GPU overlap or model token loop was
measured. Keep it isolated and continue broader GPU arithmetic/dataflow work.
See [scope, preparation costs and evidence](CPU_EXACT_QK_EXPERIMENT.md).

## Exact native EXP correction components, 2026-09-17

Sources `4fca895` / `d89c260` verify all328728576 exp2 inputs twice and all
120 generated probability cases. An input-indexed two-bit correction uses
the actual gfx1151 EXP instruction; unencodable inputs retain the original
table. Complete captured q8192 attention is786.5368 /745.8377ms original/new,
or790.2496 /757.1356ms including preparation. Every original raw output and
all29364224 available GB10 context cells match; q8192 repeats1023 source rows
and supplies no new model token loop. Keep the representation isolated: its
measured scope cannot resolve the TTFT gap. See [implementation and evidence](NATIVE_EXP2_DELTA_EXPERIMENT.md).

## Actual MoE down consumer filter, 2026-09-17

Sources `0cfe413` / `d852ff7` pass the complete same-DLL q8192/out512 GB10
boundary and actually omit 126904922 / 158740988 selected down replays
(79.9446%), exactly reproducing the prior observer in all 40 calls.
TTFT is 27851.3224 / 27889.4209 ms off/on, with no measured net saving.
The same-DLL MoE profile also preserves the full GB10 boundary: down correction
is 805.620793 / 663.870898 ms, while complete MoE saves only 81.955012 ms.
Nested intervals overlap and diagnostic TTFT includes profiling overhead.
Keep the filter default-off and continue a broader arithmetic or dataflow
replacement. The retained stack and all mission/release thresholds remain
unchanged. See [implementation and complete evidence](MOE_DOWN_CONSUMER_FILTER.md).

## MoE down consumer audit, 2026-09-17

Source `abb26a8` preserves all 512 GB10 IDs, actual callbacks and first logit
10.375 in both same-DLL modes. The read-only audit checks all 671088640
production F32 residuals with zero errors and conditionally certifies
126904922 / 158740988 selected down replays (79.9446%) removable at the full
combined consumer. Every original replay still executes. TTFT is
27925.6884 / 28486.2800 ms off/on; observation adds time and demonstrates no
speedup. Keep it off and test actual consumer filtering separately. All
mission and release gates remain unchanged. See [scope and evidence](MOE_DOWN_CONSUMER_AUDIT.md).

## Adaptive strict attention-denominator components, 2026-09-17

Sources `1f13077` / `065b258` preserve every tested original attention output
and all 29364224 GB10 context cells per captured configuration. Adaptive
refinement reduces q8192 score replay to 36.4895%, but remains slower than
the original full-score path. Scalar/shared-tile and 18/6-round comparisons
put the best candidate at 2177.5395 ms versus 445.1504 ms for its original
QK/probability control. Collection plus interval initialization alone exceed
the original control. Keep every variant isolated and continue structural
MoE consumer work. These are captured components, with no new model token
loop, TTFT or release acceptance. See [complete scope and evidence](ADAPTIVE_DENOMINATOR_QK_EXPERIMENT.md).

## Queued adaptive linear OUT replay, 2026-09-17

Source `6e4908b` separates original K16 replay from per-token certification.
Both same-DLL q8192/out512 modes pass all GB10 output IDs, actual callbacks
and the first-logit boundary; all 30 intended layers execute the queue.
TTFT is 27870.9080 / 27193.1037 ms off/on. All 245760 token rows certify,
with zero observed interval failures and the same 49.9088567% candidate
reduction as the fused route. The single 677.8043 ms difference does not
establish a repeatable gain or meet the 10000 ms gate. Keep variant 2
default-off and the retained stack unchanged; continue structural attention
work. See [implementation, limits and complete evidence](OUT_VARIANCE_QUEUE_EXPERIMENT.md).

## Actual adaptive linear OUT replay, 2026-09-17

Corrected source `5c085c9` activates all 30 intended linear layers and passes
all 512 original GB10 output and callback IDs plus first logit 10.375 in both
same-DLL modes. Actual replay skips 29,191,574 of 58,489,767 candidates,
matching the prior observer, with zero observed interval failures. TTFT is
27777.4475 / 27678.0636 ms off/on; the single 99.3839 ms difference does not
establish a repeatable gain. Keep it default-off and preserve the retained
stack. The next experiment separates original K16 replay from per-token
certification to assess the fused execution cost. All mission and release
gates remain unchanged. See [implementation, activation fix and complete evidence](OUT_VARIANCE_REPLAY_EXPERIMENT.md).

## Adaptive OUT variance-budget audit, 2026-09-17

The same-DLL q8192/out512 pair at `c3b4ad5` preserves every GB10 output and
callback ID, first-logit boundary and original correction count. A read-only
observer conditionally certifies 29,191,574 of 58,489,767 linear OUT candidates
removable while preserving the complete RMSNorm consumer. No candidate replay
or saved execution runs; the audit adds 1,375.2279 ms in this single pair.
Advance only the linear shape to an actual default-off experiment. The
retained stack and all mission/release gates remain unchanged.
See [classification, limitations and product evidence](OUT_VARIANCE_BUDGET_AUDIT.md).

## Shared exponent-mask GDN, 2026-09-16

Source `185546a` passes full bitwise component comparisons using the provider's
eight 1024-token q8192 segments. Product U=V totals are 69.4084 ms for the
original route and 71.4949 / 72.1455 ms for the masked 64/32-row output tiles.
The unmasked 32-row ablation is 66.9655 ms. Keep all variants isolated; the
certificate does not improve the complete alias path. No real-model or GB10
acceptance is claimed. See [the complete experiment](EXPONENT_MASK_GDN_EXPERIMENT.md).

## Exact exponent-mask QK, 2026-09-16

The same-DLL q8192/out512 pair at `e63b66f` passes every GB10 output ID,
first-logit boundary and actual callback in both modes. TTFT is
27552.9067 / 27452.0422 ms with the exact mask option off/on; all 10 attention
layers execute it when enabled. This single 100.8645 ms difference does not
establish a repeatable product gain. Keep the option default-off and the
retained stack unchanged. The 10-second gate remains open.
See [component, integration and product evidence](EXPONENT_MASK_QK_EXPERIMENT.md).

## Native GPU execution modes, 2026-09-16

Same-source WGP/CU builds of all four in-tree HIP providers pass the real
q8192/out512 GB10 token, first-logit and callback boundaries. TTFT is
27936.2702 / 27730.1676 ms; the single206.1026 ms reduction accompanies a
higher CU TPOT. Keep the default and retained stack unchanged. Actual kernel
descriptors confirm all637 entries in the requested mode, all wave32.
See [the build scope and evidence](GPU_EXECUTION_MODE_EXPERIMENT.md).

## Coarse linear OUT, 2026-09-16

A same-DLL real q8192/out512 pair at `768d054` passes every GB10 token,
first logit and actual callback with the new linear OUT owner both disabled
and enabled. TTFT increases from27891.0528 to28559.2248 ms. All30 linear
layers execute the experiment; candidate work grows to88762686 outputs.
Keep the option default-off and outside the retained stack. The original
10-second gate and4187.415605 ms retained target remain unmet.
See [the implementation and exact evidence](COARSE_LINEAR_OUT_EXPERIMENT.md).

## Current unreleased measurements, 2026-09-15

Source `5011e8b` audits exact input-row reuse before all170 q8192 dense
corrections. Hashes only propose representatives; a separate GPU pass compares
every original BF16 word before accepting a duplicate. All original producers,
candidates and K16 replays remain active. Sanitized host-owner tests cover
8192 rows, a deliberate hash collision and24 injected transport failures.
Full54 Rust/425 Python(two skips), C/q16 ABI, clippy and hygiene checks pass.
Native build passes in92442.054 ms. Evidence:
`benchmarks/correctness/projection-row-reuse-native-20260915.json`, SHA256
`bf0318d776ea5796b0455c66781d810ba0966033447d933ba3e24f3ee90c5cef`.

The real q8192/out512 audit matches every GB10 output ID, actual callback and
first logit10.375. TTFT is30645.3769 ms, TPOT100.940785 ms and load21488.4797 ms;
these include the audit and are not a paired timing comparison. Only the first
four layer0 projections have duplicate inputs:7935 reusable rows each,257
distinct rows and largest class46. The other166 calls have no identical input
rows. All full-word comparisons and private-buffer guards pass, with zero
hash collisions. All170 original counts and dispatch counts match the prior
disabled control. Even a loose candidate-reuse upper bound is only6366393 /
432889515 (about1.47%), before cache and scatter costs. Do not implement this
whole-row cache for the current bottleneck; retain the audit default off.
No actual reuse or candidate speedup is measured. Evidence:
`benchmarks/correctness/projection-row-reuse-q8192-20260915.json`, SHA256
`685d30c52583bca7f807ac44fefc7f8da9b8d165179a16e1cf36c3989ad4cb03`.

Source `4a5a317` tests an actual default-off OUT residual filter inside the
existing candidate collector. It keeps the original matrix, PPB/Cauchy and
midpoint envelope, mandatory prefixes and retained K16 arithmetic. A selected
cell can skip replay only when its entire projection interval yields one
BF16 residual; unrounded RMSNorm variance is not certified. No extra matrix
or workspace is allocated. Full local checks pass 54 Rust/424 Python (two
skips), C/q16 ABI, clippy and hygiene. Sanitized actual-owner tests exercise
the full16777216-cell geometry, mixed retained/removed candidates, mandatory
prefixes and five injected synchronization failures; interval tests cover
130560 cases. Native build passes in92673.142 ms with all guards passing.
Evidence: `benchmarks/correctness/out-residual-filter-native-20260915.json`,
SHA256 `bc3b14caafb4acbd18edfda7f44c294de35a21a82ad26acc776280626c11b38d`.

The real same-DLL q8192/out512 pair rejects the enabled filter. Disabled
matches all512 GB10 IDs; enabled first diverges at index1 (expected255,
actual248) and matches only42 positions. Both first tokens/logits remain
144/10.375, and all512 callbacks in each run match its own output. Enabled
exits6 at the numerical gate with all host guards passing. Disabled/enabled
TTFT is30307.294601/28301.5993 ms, TPOT101.024893/100.519353 ms and load
21532.9574/21369.1028 ms. Across40 OUT calls the filter retains64785461
candidates; total dense candidates change432889515 to332708494 and exact
dispatches1792 to1414. The observed2005.695301 ms TTFT reduction cannot be
retained because continuation fails. Keep the option off and move to a
structural route preserving the numerical boundary. Evidence:
`benchmarks/correctness/out-residual-filter-q8192-20260915.json`, SHA256
`fa90c8c04e931b592bde47d4fcf0c8398fca15746fad34131acd12d72740f771`.

Source `1d14571` adds a default-off, read-only OUT consumer interval audit.
All original corrections and model inputs remain in use. Host sanitizers
cover 130560 interval edges, 74576 enumerated BF16 values and 24 injected
owner failures; full local checks pass 54 Rust/424 Python (two skips), C/q16
ABI, clippy and hygiene. Native build completes in 92599.893 ms with all
guards passing. Evidence: `benchmarks/correctness/out-consumer-audit-native-20260915.json`,
SHA256 `f85e343aeecc97f802de2444c3665a293e29fba656df72646eda3781d8929dd4`.

The same-DLL real q8192/out512 disabled/audit pair matches every GB10 token,
all actual callbacks and first raw logit 10.375. TTFT is 30294.6291 /
31661.7987 ms, TPOT 101.114694 / 100.394805 ms and load 21595.0482 /
21340.6742 ms. All 170 original correction counts and dispatch counts match.
The 40 audits report no observed producer-envelope undercoverage, residual
or normalization certificate failures, or private-buffer guard failures.
Only 2318631 of 164967882 OUT candidates (1.4055%) are removable when both
residual and normalized output must be certified unchanged. The diagnostic
uses completed original outputs as a stand-in for a hypothetical first
replay pass; no candidate pass or saved work executes. Its extra 1367.1696 ms
TTFT is diagnostic overhead, not a speedup. Do not implement that strict
two-pass route on this evidence. Residual-only invariance covers 100221835
candidates, but does not preserve the unrounded variance used by RMSNorm.
Any such broader selector requires a new actual GB10 token/logit comparison.
Evidence: `benchmarks/correctness/out-consumer-audit-q8192-20260915.json`,
SHA256 `d8635e70e0f35c26fc3ce0af5b896466091a70b518a1b94193745b3cf44cf8dd`.

A separate host arithmetic diagnostic samples 65536 independent logical
cells for each full QKV/OUT geometry. The original CPU OUT implementation
matches all sampled GB10 BF16 endpoints. Carry truncation dominates the
observed error, but a simple output-dependent bias leaves 63 OUT and 25 QKV
BF16 differences in the held-out 32768-cell halves. A stronger fitted proxy
requires the original expensive carry values. No fitted coefficient is
integrated; FP64 sums and these host samples establish neither native matrix
accuracy nor inference performance. Evidence:
`benchmarks/correctness/projection-rounding-profile-host-20260915.json`, SHA256
`727d715136cbae365bcc16465c58f84e40416e2a27d7fb44790aefc63fde84ba`.

Source `ce4d829` adds default-off `QRT_QWEN36_Q8192_OUT_L1_BOUND` for the
original-algorithm OUT shape 2048x8192x4096. It computes an outward-finished
positive magnitude matrix, caps the empirical bound by the original Cauchy
bound, then invokes the original candidate collector and K16 replay. PPB,
midpoint512, tiny-value admission and prefix behavior remain unchanged.
The real owner and finish kernel pass sanitized deferred-stream tests over
16777216 bound cells, all six supported factors and 23 injected transport
failures, including drain-before-release. Full local checks pass: 54 Rust,
423 Python (two skips), C/q16 ABI, clippy and hygiene. The native whole-provider
build completes in 92037.836 ms with all host guards passing. Evidence:
`benchmarks/correctness/out-l1-replay-native-20260915.json`, SHA256
`d1a254c241f2dce51a638c8e4b2e87f16b822b6907bf9f09267075f9c2b919b2`.

Three fresh real-model q8192/out512 runs use the same new DLL, the original
CLI and all other archived providers. Required-marker filtering is enabled
in every arm; only the L1 factor changes. Both candidates fail the full GB10
token boundary despite passing the first-logit tolerance of 0.125:

| L1 factor | TTFT ms | TPOT ms | Load ms | First raw logit | Matching prefix / positional IDs |
|---:|---:|---:|---:|---:|---:|
| 0 (disabled) | 30261.108 | 101.038764 | 21526.532 | 10.375 | 512 / 512 |
| 1 | 29128.796299 | 100.819079 | 21347.1606 | 10.3125 | 115 / 119 |
| 2 | 30119.7973 | 100.944428 | 21290.897 | 10.375 | 115 / 116 |

Every run emits 512 actual callbacks matching its own output. Both candidates
first return196 instead of271 at zero-based index115, exit6 at the numerical
gate, and finish with all host guards passing. Factor1/2 reduce total dense
candidates from432889515 to352498653/399045958, while their40 magnitude owners
add615.8288/614.7283 ms preparation. These timings are diagnostic and do not
qualify performance. Nested owner replay time includes the original correction
and must not be added again. Keep the option default off and pursue a broader
structural route. No candidate correctness, long-context, soak or release
acceptance follows. Evidence:
`benchmarks/correctness/out-l1-replay-q8192-20260915.json`, SHA256
`193edf8815434424f121922505ed11283a8e995d191117d99ebf9a79ae2cc0cf`.

The unpublished portable package `v1.0.2-current-stack.20260915.r1` now
combines whole/server/CLI `24c4304`, CK `42c2f0f`, MoE `4a7f0c4` and FLA
`2ee6215`. Its 268 runtime assets and 281 release files pass hash, size and
archive CRC checks after relocation with the original staging directory
unavailable. Only 33 path values change in the 506-option runtime profile.
The 131997808-byte ZIP has SHA256
`5c8ea7a76a2ac368a306641104c277978a67f4770d62b7002b1e0b6354edc458`.
It remains unpublished and has no retained-performance or release acceptance.

Two real q8192/out32 HTTP requests pass: all 32 nonstream GB10 token IDs,
both prompt-bound first logits (144/raw 10.375), and matching SSE text and
usage. Load is 21009.5133 ms; nonstream/stream TTFT is 30129.7631 /
29492.5287 ms and TPOT 109.277645 / 99.455355 ms. These are out32 HTTP
observations, not an out512 comparison. Evidence:
`benchmarks/correctness/current-portable-http-20260915.json`, SHA256
`9025c0046d90104956b0ea9dfb3e8cf72c5a642423cf8d4eef978ad500ed1bd3`.

The same package passes 17 protocol checks in 30 captured HTTP requests,
including tools, thinking, FIFO, timeout, overflow and graceful shutdown.
The active q8192 request finishes with all 32 GB10 IDs and first logit while
waiting requests are cancelled. Four saved-prefix branches pass 256 raw
output IDs, 12 first logits and four SSE comparisons in 15 requests. All
12 private transactions restore the owner after 31 committed decode inputs;
owner replacement and subsequent cold fallback also pass. Prefix checkpoints
remain opt-in and SSE exposes no raw token IDs. Both processes stop normally
with host guards passing. Evidence:
`benchmarks/correctness/current-portable-protocol-prefix-20260915.json`, SHA256
`aecf32749d6d213bc567c4f85e6ba3514436fb6d97b11820546592ee2d172032`.

Five short cold CLI cases from actual HTTP prompt IDs pass all 640 GB10
output IDs, all actual callbacks and exact first logits. They cover plain
text, chat, tool call, tool continuation and a 512-token thinking sequence.
Each uses the packaged CLI and providers in a new process, with no dispatch
or numerical overrides. Load spans 21298.7011–21394.3392 ms; callback TTFT
2733.1037–3545.0566 ms. Evidence:
`benchmarks/correctness/current-portable-short-matrix-20260915.json`, SHA256
`28e7d27f0bb752187db51df83a9e039bfc96f78052d68285e0bab4a8ff759b34`.
The eight-case full cold matrix now passes all 1216 GB10 output IDs,
all actual callbacks and exact first logits, including q8193. Each case uses
a new process and the packaged CLI/providers without numerical overrides.
The packaged q8192/out512 run measures 33062.3377 ms callback TTFT,
100.315210 ms TPOT and 21300.7426 ms load; all eight loads are below30 s.
Evidence: `benchmarks/correctness/current-portable-cold-matrix-20260915.json`,
SHA256 `16ee876fc4f94d1244670563ac7eec30fcd34fd09ed0e024741a8536003a0e61`.

Server `f1ea668` fixes the soak setup timeout by caching the immutable full
vocabulary size, including added special tokens, at tokenizer load and reading
that bound once per detokenize request. The former loop copied the full
vocabulary for every token. All 47 local Rust tests, clippy and 42 Windows
server tests pass. Native build wall is 51683.174 ms. Failure evidence:
`benchmarks/correctness/current-portable-soak-setup-failure-20260915.json`.

The rebuilt portable package changes only `engine/qrt.exe` among 268 runtime
assets; all CLI/provider assets and the runtime profile match the qualified
short/cold matrices. Relocation and all 281 release-file checks pass. Fresh
q8192/out32 HTTP requests match all raw IDs, both first logits and SSE.
Load is 21400.437 ms; TTFT is 30048.8098 / 29548.0342 ms. Evidence:
`benchmarks/correctness/tokenizer-vocabulary-cache-portable-http-20260915.json`,
SHA256 `b90d6bba3ee83dcf89350ee7c0ce20e8abc0eba8cb5b9654ad8ff9db39945348`.

Fresh protocol regression passes 18 checks in 31 requests. The formerly
failing 512-token detokenize returns in 15.6734 ms HTTP wall / 31.5067 ms
including capture, within the original five-second bound. Saved-prefix
regression again passes 256 raw branch IDs, 12 first logits, four SSE cases
and all 12 owner-state rollbacks. Both services shut down normally. Evidence:
`benchmarks/correctness/tokenizer-vocabulary-cache-protocol-prefix-20260915.json`,
SHA256 `268bee10b952b5cf50571d46e75f2c504b186e35521f8d66e8fe76fbb60299bc`.

The repaired service passes a same-process HTTP soak with a 3625.232714-second
active window: 63 requests in 21 complete cycles verify 13344 raw GB10 output
IDs, 42 prompt-bound q8192 first logits and 21 SSE text/usage/finish cases.
Offline replay verifies the saved response bytes and queue accounting.
The service exits 0 normally without forced cleanup; all host guards pass.
Load is 21087.0874 ms. q8192/out512 nonstream/stream median TTFT is
29579.4646/29573.4978 ms, median TPOT 100.032222/100.154117 ms.
All 722 health/memory samples complete below 19.628 ms. After the first cycle,
private bytes change from 62900523008 to 62976548864; this includes caches and
allocator state and does not prove absence of leaks. Evidence:
`benchmarks/correctness/current-portable-one-hour-soak-20260915.json`, SHA256
`0b0a5904bb377ec223b84ffd43c4c0fca31a392972a5b6480802ce79ee66daec`.
The previous setup/controller failures remain preserved separately. Long
contexts, immutable performance and release remain open.

An actual C-core/Rust Windows probe confirms a separate configuration bug:
Rust's first process-environment assignment is invisible to the existing C
early-prefill flag reader. Further override/removal leave stale CRT values;
the same C runtime's setter updates both views. Seven cases include four
stale reads, with no model or GPU work. This does not yet explain the CLI/HTTP
timing difference. Source `e7d605f` repairs startup propagation through the
same C runtime, then preserves Rust's OS string and empty-value semantics.
All 47 local Rust tests, C smoke, clippy and hygiene pass. Native regression
also verifies the actual core reader, invalid inputs, Unicode and 8192-unit
values. The Windows service build passes all 42 server tests. Evidence:
`benchmarks/correctness/windows-runtime-environment-handoff-native-20260915.json`,
SHA256 `cacd36ff5a44876b04d6636b3cf7fb264c2ff970f1f727466ece896918bb3378`.
Evidence: `benchmarks/correctness/windows-runtime-environment-stale-core-20260915.json`,
SHA256 `9ada0c5c799a3cc8ac73f92015489e606a49f0bdc49f701f3aab788bd3ea0111`.

The R3 portable package changes only `engine/qrt.exe` among 268 assets and
retains identical profile bytes. All 281 release-file hashes and relocation
checks pass. The 132007801-byte archive has SHA256
`cd098cc2b830f6f7d475e342702fdb8656954c7b16b67a7e519d1261cbdf7c56`.
Both real q8192/out32 HTTP requests now exercise the early-prefill callback
and pass the original raw IDs, both first logits and SSE comparison. Load is
21509.6126 ms; nonstream/stream TTFT is 30219.4665/29676.6777 ms and TPOT
108.597532/99.423623 ms. The prior CLI/HTTP timing gap remains unexplained.
Evidence: `benchmarks/correctness/windows-runtime-environment-portable-http-20260915.json`,
SHA256 `2bc375fbc6717de638126795afb4da752fd58fc3f6572372148ca3a0691ae743`.
New protocol regression passes 18 checks in 31 requests; full512 detokenize
finishes in 40.1152 ms including capture, within its five-second bound.
Four saved-prefix branches pass all 256 raw GB10 IDs, 12 first logits, four
SSE cases and 12 private owner-state rollbacks. Owner replacement, cold
fallback and the five short protocol prompts also pass. Evidence:
`benchmarks/correctness/windows-runtime-environment-protocol-prefix-20260915.json`,
SHA256 `a07e4cffe7e463fdcf416e2eec1cbbaf588d1ded0e00d6468e6e9b51e3b0d9ca`.

The same R3 service recovers from TCP resets during native q8192 prefill and
after the first nonempty SSE text. Reset-to-idle time is 30.1569 / 0.2402 s;
prefill cancellation is observed at the first callback, without an immediate
GPU abort. A subsequent cold request matches all 32 GB10 IDs and first logit
144 / 10.375. Its TTFT is 29544.9332 ms, TPOT 98.248458 ms and load
21215.6723 ms. Queue accounting, 142 health samples and normal shutdown pass.
Cancelled generations receive no numerical acceptance. Evidence:
`benchmarks/correctness/windows-runtime-environment-disconnect-20260915.json`,
SHA256 `d9d6e01352c0df09a8ac66858330a0e0c06430f55334d07b7c6b05b2ee1aa67d`.
The R3 one-hour soak completes a 3628.061158-second active window with
63 requests in 21 cycles. Offline replay verifies all 13344 raw GB10 output
IDs, 42 prompt-bound first logits, 21 SSE text/usage/finish cases and the actual
accepted early-prefill callback in every q8192 request. All 723 health samples
respond within 19.7189 ms, queue accounting passes and the service exits
normally. Load is 21134.7647 ms. Nonstream/stream median q8192 TTFT is
29586.2630 / 29582.0702 ms, with TPOT 100.087300 / 100.090114 ms. Evidence:
`benchmarks/correctness/windows-runtime-environment-one-hour-soak-20260915.json`,
SHA256 `fcbc93ed85969ba024f08ee13ba0fef800dea3bae3010d50e6d5c0b6aab014f4`.
This evidence is attached to the e7d605f service and its repaired C configuration;
it does not transfer to a later executable or qualify the open performance gates.

The September15 upstream refresh identifies Linux `.9`, published September14.
The Windows CPU protocol candidate now reopens tool retries after completed
repairs, retains lifetime diagnostics, orders results by their issuing turn
and rejects replayed IDs. All exhausted proposals return HTTP400 or an SSE
error followed by DONE without a successful finish; mixed calls retain their
admitted actions. Five new regressions first fail on the previous implementation.
The imported caller-side DOCX checker additionally fixes Unicode JSON output
on legacy Windows console encodings. Its actual CLI returns0/2 correctly for
real/fake containers, and recovery cannot turn a fake DOCX into a passing file.
The packager now includes the checker and its API/integration guides.
All54 Rust tests,416 Python tests with two conditional skips, C smoke, clippy,
format and hygiene pass. Evidence:
`benchmarks/correctness/tool-recovery-and-document-checker-local-20260915.json`,
SHA256 `3e236824edc98613008ba3f232f933b2199156873dd04c8a30c4dfbc58122ddb`.
The f6018ca Windows service builds with all 49 server tests passing. The first
native caller run then fails one of nine tests: default text output translates
recovered CRLF into CRCRLF and adds blank lines. Evidence:
`benchmarks/correctness/tool-document-windows-newline-failure-20260915.json`,
SHA256 `db15f997f24994e065b1bf47ca323579cfa261768215d0c286c0f76bd10b9f39`.
The helper now disables output newline translation. A Windows text-I/O
regression reproduces the LF/CRLF failure before the fix; all 10 caller tests
then pass locally, together with hygiene and diff checks. Evidence:
`benchmarks/correctness/tool-document-newline-local-20260915.json`, SHA256
`9bf65f1dd03c6e1c0d5057a43f7c8f80d7d8a1252673a1ec0025aaa7a46b9512`.
Source `4fca963` passes the renewed Windows build, all 49 server tests,
10 caller tests and two verifier tests. Native evidence:
`benchmarks/correctness/tool-recovery-and-document-checker-native-20260915.json`,
SHA256 `7dc6b2418d7e095f28e00e991a51d3e37c26a291b79f388249e2d292bf9678f9`.
The R4 archive contains 268 runtime assets and 284 release files, all verified
after relocation while the original staging path is unavailable. The actual
packaged checker passes six cases, preserving CRLF recovery bytes and existing
Markdown. Its JSON/SSE q8192/out32 pair passes the original GB10 IDs, first logits
and both actual early callbacks. TTFT is 30065.2408 / 29509.5863 ms and TPOT
108.963965 / 98.694952 ms. Archive SHA256 is
`54958df734f52e32f419ef65482497fc4c9380e57acdf2a03e0fbdea4e4974cc`
(132023789 bytes). Evidence:
`benchmarks/correctness/tool-recovery-portable-package-http-20260915.json`,
SHA256 `fb85e959d23ef00e5929ba636ab96d9f5a49b194d60073ff4f9cecc88614276c`.
The full native suite then passes 22 checks in 39 captured HTTP requests,
including completed and silent repair windows, retained lifetime diagnostics,
explicit JSON/SSE exhaustion errors, replayed-result rejection, queue timeout/
overflow and graceful shutdown. The final active q8192 request matches all
32 original GB10 IDs and first 144/raw10.375. Offline response replay passes;
18 accepted early callbacks include the final prompt-bound q8192 case. Evidence:
`benchmarks/correctness/tool-recovery-native-http-protocol-20260915.json`, SHA256
`8998fe084db5fc3e6239df3f3ea3aeaac110b6cb2cc0e8989ab694bc313caa64`.
The newline fix changes no runtime math. The earlier e7d605f soak is not
transferred to this server, and performance/long-context release gates remain open.

The new default-off `QRT_QWEN36_FLA_DEVICE_PREPARATION` option removes the
normalized postconv allocation and two unused kernels when raw FLA owns Q/K
normalization. It moves the complete-domain GB10 gate lookup from the CPU to
the current GPU stream, sharing decode's immutable G allocation. Host/trace
captures retain their original surfaces. No dot arithmetic or selection bound
changes. Existing 405 Python tests (two skips), C smoke and hygiene pass.
The Windows whole build and 159 native cases pass, comparing 630724160 raw
output cells with no bit mismatch and unchanged guarded inputs/tables.
Evidence: `benchmarks/correctness/fla-device-preparation-native-20260915.json`,
SHA256 `5096abb69fcdbb4806bf4c4b7315396092adf421c02692298c3eb1fe06e7db3a`.
The same-DLL q8192/out512 OFF/ON pair matches all 512 GB10 IDs and actual
callbacks in each run, first 144/raw 10.375. TTFT is 32940.2100/32518.3415 ms,
TPOT 101.195048/101.251283 ms and load 21260.8878/21325.6390 ms. The observed
421.8685 ms reduction leaves q8192 above 10 seconds; the option stays default
off after this single pair. All 30 preparation markers occur only in ON.
Evidence: `benchmarks/correctness/fla-device-preparation-q8192-20260915.json`,
SHA256 `38f34c0f865ccb20a9746b220447d1aa8ad73172c7d55b95134cd81353aa9dfd`.
Local record: `benchmarks/correctness/fla-device-preparation-local-20260915.json`.

Whole provider and CLI `24c4304`, CK `42c2f0f`, staged MoE `4a7f0c4`
and FLA `2ee6215` pass the current q8192 regression on baiying with
`D:\models\Qwen3.6-35B-A3B`. All 512 original GB10 output IDs and actual
callbacks match, first 144/raw logit 10.375. Callback TTFT is 32880.1059 ms,
TPOT 101.126072 ms and model plus engine load 21331.9965 ms. Long QK and
long V options are inert on this shape; neither long owner nor extended
attention scratch is allocated. The 264736-row RoPE table remains active.
This is a single regression observation; q8192 remains above 10 seconds.
Evidence: `benchmarks/correctness/long-query-range-q8192-20260915.json`, SHA256
`ad8a3ecdd32785953f74522da4634395464ce849f2efdd73e77c72fcd156e009`.

The same DLL pair passes the real 16384-token owner plus 1024-token suffix
and 512-token continuation with long QK off/on. Both initial suffix retries
and warm hits match every GB10 token, first 3709/raw 5.6875. Warm hits match
all actual callbacks; prefix restoration and changed-prefix rejection pass.
The complete owner first token is 16/raw 25.625; its separate 32-token
continuation is outside these runs.

Only `QRT_CK_SM121_LONG_PREPARED_DECODED_QK=0/1` differs, with long V
transpose enabled in both. Warm TTFT is 11112.1268 / 10731.3217 ms,
TPOT 157.262214 / 154.945963 ms and complete warm request 91545.7034 /
89972.8458 ms. Owner time is 96502.4706 / 93530.0121 ms; first retry is
93864.4221 / 89179.6647 ms. Seed plus first retry changes from
190367.0161 to 182709.8002 ms, and process wall from 303653.084 to
294445.252 ms. Load is 21258.3081 / 21283.5870 ms. Select long QK and long V
for continued long-context experiments; retain default-off source settings
and all-cell PV off. This single pair establishes no repeatability or
retained-performance acceptance. Evidence:
`benchmarks/correctness/long-query-range-prefix16k-20260915.json`, SHA256
`004c76c50435db7aae7ce38af572a9f73c25d4e3b0db5d391d31c33b8f99eb8f`.

Range QK prepares only the consumed Q interval and complete K history,
retaining the original BF16 fallback and 32-query long slabs. Its separate
owner grows by 8192 tokens, capped at 264736 keys / 679039232 bytes, with
at most 8192 prepared Q rows. Failed growth preserves the previous owner;
every selected call refreshes metadata under the existing provider mutex.
The real 16k run grows from 168427520 to 185270272 bytes and records 30
long calls plus 10 cold-owner calls. Actual-provider host tests cover strict
options, both PV layouts, all-cell routes, both attention owners, growth
failure/retry, cold/long coexistence, V transpose and failure drain.
Windows CK build, C checks, 401 Python tests with two skips and hygiene pass;
unchanged Rust/Cargo reuse the prior 47 tests and clippy.

The range component `e0b2732` passes 26 native reports across 13 shapes
through position 264735. Both float-alignment and prepared paths match all
418496528 original q7169 score slots and 228 independent CPU dots per path,
with complete device metadata, original inputs and guards checked. Including
preparation, same-executable component clocks are 348.3046 / 276.4122 ms.
These component observations are not model performance. Evidence:
`benchmarks/correctness/prepared-decoded-qk-range-components-20260915.json`, SHA256
`c9bcf414aca5faa7a2dff51d136bcf673d90e20b9f9430fd41c7f508c9381b83`.

Long V reuses the existing exact PV arithmetic through a separate refreshed
transpose owner, capped at 271089664 bytes. Its 18 native cases pass 1609728
output cells, raw accumulators/denominators, candidate sets, input layout and
guards. The earlier CK `7fcf8a1` 16k off/on pair also passes GB10 and observes
warm TTFT 12203.6283 / 11094.3167 ms, with slower decode TPOT when enabled;
no retained speed follows from that pair. Evidence:
`benchmarks/correctness/long-transposed-value-components-20260915.json`, SHA256
`3d6321a0908b744db6ea9bf983b8dbfa0e6a634d813ace61e6171ed1948f8268`, and
`benchmarks/correctness/long-transposed-value-prefix16k-20260915.json`, SHA256
`16e08b534ba68b4150299d8f95c0d86247dcd85e47324fc3958b9028ac4e14cd`.

Current-stack profiling preserves all 512 GB10 outputs and callbacks.
Only four diagnostic flags differ from the same-artifact q8192 control.
Completed attention totals 8897.2974 ms: QK 3716.4445, probabilities
1365.3326, approximate PV 1765.1039 and exact PV 1842.0845 ms; preparation
is 51.2365 ms. Forty dense OUT corrections total 3463.853 ms, and MoE
routed gate/up/down corrections 1195.073878 / 1347.092419 / 1014.207606 ms.
These detail intervals are nested; shared MoE overlaps routed work. Ten
negative residual/postnorm GPU intervals are excluded. Instrumented TTFT
34529.1447 ms, TPOT 104.707169 ms and load 21349.6095 ms are diagnostic.
The subsequent block-layout and shared-PV comparisons below preserve the
original K16 carry but produce no gain. Profile evidence:
`benchmarks/correctness/current-range-stack-profile-20260915.json`, SHA256
`8209dbcfe8d47433f5313229fa26b1a038af577a53adf1d6f7a85dcbaf9d5d88`.

The actual 128k prefix run `range-prefix128k-r1`, with long QK/V enabled,
was explicitly cancelled by Codex after 1042299.768 ms. Six owner chunks
completed, covering 49152 input tokens; the last full chunk took 233104 ms.
No owner token, suffix retry or warm continuation was produced. The native
guard confirms process cleanup, memory reserves and GPU health. This is an
incomplete measurement without a numerical verdict; its 3600-second process
bound did not expire. Real 128k/256k remain unqualified. Evidence:
`benchmarks/correctness/long-query-range-prefix128k-incomplete-20260915.json`, SHA256
`2bab80f037311f79ce231cb34457609f3e7cc08ae1b2d4ceca7ed8ecd3d97142`.

The prepared-weight block component `c303565` passes 12 native reports
covering 1386756 original K16 carry states, including padding, lossless
encodings, unsupported operands and guards. Three same-executable q8192
captured projection routes match all 67108864 external GB10 BF16 cells and
4331311 unrounded correction candidates on every attempt. Including full
operand preparation, original scalar4 / current staged2 / blocked-weight
staged2 medians are 58.8297 / 42.6927 / 55.9998 ms over three rotated samples.
Keep current row-major staging2; the blocked layout is slower and remains a
component. Sanitized layout/launcher checks cover 48 shapes; C smoke, 402
Python tests with two skips and hygiene pass. Product dispatch and retained
performance are unchanged. Evidence:
`benchmarks/correctness/blocked-half-weight-components-20260915.json`, SHA256
`2de35fb86cfca751062e30c6f822df74a7688ba1e1d6e5e16db9607f79353e51`.

Shared scaled-half PV `982acd4` passes 220 short and 20 long native reports,
through query position 264735, preserving original output/accumulator/
denominator bits, candidate ownership, full encodings and guards. Both real
q7169 routes match all 29364224 GB10 BF16 cells and 2198673 selected cells,
with 228 independent CPU recurrences. Including P/V encoding, three-sample
replay medians are 95.0127 ms original and 245.1866 ms scaled-half. Keep the
original compact integer PV; no runtime integration follows. Separate untimed
diagnostics check 637019466 groups: 636913770 transformed and 105696 original
fallback. C smoke, 403 Python tests with two skips and hygiene pass. Evidence:
`benchmarks/correctness/shared-scaled-half-pv-components-20260915.json`, SHA256
`9f48a8e49206199e1087d82022b3df21540c1d44c4c13e3c1adbf86ef54253c7`.

Completed linear/FLA diagnostics at `7fc5b2d` preserve every original q8192
GB10 token and callback, first 144/raw 10.375, with CK `42c2f0f`, MoE
`4a7f0c4` and CLI `24c4304`. Instrumented TTFT is 34852.0179 ms, TPOT
103.959093 ms, load 21283.2918 ms and process wall 109859.358 ms. Whole/FLA
native builds pass in 92117.954 / 40562.391 ms. All arithmetic headers and
AOT sources match the current control; diagnostic provider code alone changes.
Full local checks pass 403 tests with two skips, C smoke and hygiene after
fixing missing parser includes in an existing extracted-code test fixture.

All 30 linear layers have eight ordered completed host phases. Totals in ms:
setup 23.9383, convolution 805.8966, gate 425.8651, recurrent 3783.405,
gated norm 196.4319, OUT projection 2824.3749, residual/postnorm 188.1001
and materialization 47.1473. The final clock/log overhead is 42.2718 ms;
together these account for the 8337.431 ms linear-core scope. The separate
preceding projection scope is 4378.69 ms. These are instrumented scopes.

All 240 FLA segments have eight completed records. Full valid GPU-stage sums
in ms are V/beta copy 42.19459, KKT 157.99462, inverse 51.39964,
WU 420.58732, state 998.69225 and output 1021.75751. One norm interval and
159 gate-cumsum intervals are negative and excluded: their 40.12011 /
1.92992 ms sums are partial, not full-stage times. FLA GPU clocks are nested
inside the recurrent host interval. Ten full-attention residual/postnorm
GPU intervals are also excluded; none rejects the valid GB10 boundary.

Full-attention OUT is 2736.685 ms; the 40 dense OUT corrections nested in
linear/full OUT total 3455.015 ms. The subsequent cooperative tile comparisons
below preserve ascending K16 carries and original fallback, but regress both
QKV and OUT after full operand preparation. Keep current staged2 dispatch;
inspect exact QK computation for a broader reduction in the measured wall.
Retained performance, 128k/256k, remaining package coverage, soak and release remain open.
Evidence: `benchmarks/correctness/completed-linear-pipeline-profile-20260915.json`,
SHA256 `e9fbcf0beb1e8c5c623954b10477bc0eb97f1f6d93eed9e3391779a9706d5404`.

Cooperative prepared-operand tiles `3d24755` pass 72 native reports and
2189499 independent original K16 states, including empty, sparse and full
candidate masks, tail rows/groups, every output bit and production/trace parity.
All four q8192 QKV configurations preserve 67108864 external GB10 BF16 cells
and 4331311 raw correction candidates on every attempt. Current staged2 /
16x16 K128 / 32x16 K128 / 32x16 K256 preparation-plus-bitmap/replay medians
are 42.0426 / 344.640 / 199.670 / 227.107 ms over three rotated samples.
Retain current staged2 QKV; no shared-tile integration follows. Full local
checks pass 404 tests with two skips, C smoke and hygiene. Native build/test
and captured comparison pass with host guards. Evidence:
`benchmarks/correctness/cooperative-half-projection-qkv-components-20260915.json`,
SHA256 `3958cf9bff4e58ba40e1ed1de150275866b3b975c940de4b0160d6779701f0cb`.

The separate, denser OUT shape uses fresh GB10 layer3 OUT references use the same captured real gated input and original
model weight at q7169 and q8192; four evaluations per shape are identical.
Both use `nvjet_sm121_tst_mma_128x208x64_2_32x104x64_tmaAB_bz_TNNN`.
The q7169 output matches its historical capture byte-for-byte and q8192
matches the corresponding repeated-input reference. These are conditional
operator goldens, not whole-model or retained-performance acceptance. Evidence:
`benchmarks/correctness/gb10-dense-out-q8192-reference-20260915.json`, SHA256
`d6085ca9a0be1ecd3c13c602c5c38de2401e08f4a9637cdff50aa435bbdd9103`.

The denser OUT comparison at `21eb841` also retains current staged2. All four
q8192 configurations match all 16777216 fresh GB10 BF16 cells and 8470282
unrounded selected values on every attempt. Matrix algorithm 0, PPB 10000 and
the original midpoint selector are unchanged. Current staged2 / 16x16 K128 /
32x16 K128 / 32x16 K256 medians, including operand encoding, bitmap and replay,
are 156.463 / 255.887 / 271.235 / 342.253 ms over three rotated samples after
one warmup. Full encodings, immutable inputs and guards pass; native build/test
complete in 102064.098 / 7210.125 ms. Local C smoke, 404 Python tests with two
skips and hygiene pass. This shared-tile implementation remains a component;
product dispatch and release qualification do not change. The prior selective
QK route already certified probabilities/alphas, but strict denominator
refinement replayed 88.7196% of scores and its approximate denominator failed
GB10 continuation. Repeating that design does not address the measured cost;
continue with the original exact QK arithmetic as the numerical boundary.
Evidence: `benchmarks/correctness/cooperative-half-out-components-20260915.json`,
SHA256 `070fa589e1f24af3935e9a364cfb3c16c32c5cb857d4786279b19eb2cb4cad62`.

Compact-half operands with FP32 carries at `bab3838` pass all 101 native
reports, including 131072 original K16 states, and full q7169/q8192 score
comparisons. Each of five variants checks 418496528 / 545259520 score slots
on every warmup and timed attempt, plus 228 / 256 independent CPU dots.
All encodings, causal masks, unused tails, guards and immutable inputs pass.
The 500000-group sanitized host check and 405 Python tests with two skips pass;
C smoke and hygiene pass. A controller-only Rust-baseline assertion was repaired
by verifying unchanged Rust/Cargo against the actual `5875f99` full validation;
the original successful Python log and the controller failure are preserved.

Completed encoding-plus-query medians for current prepared / old compact K128 /
new K64 / new K128 / new K256 are 278.8748 / 265.4747 / 311.7844 / 309.8679 /
306.2329 ms at q7169, and 370.0549 / 359.6105 / 419.4118 / 415.8095 /
410.9197 ms at q8192. The q8192 extension repeats the first 1023 captured Q/K
rows; it establishes component cost, not a new model run. Every new variant
is slower; keep current QK. The old compact control's small component gain
also remains outside product dispatch. The current 24c/42c/4a7/2ee
package and HTTP results above renew that functional boundary. Retained performance, long contexts,
soak and release qualification remain open. Evidence:
`benchmarks/correctness/half-f32-qk-components-20260915.json`, SHA256
`33ebfed12e6180c260a19c28fc949bfd691133f707fdc9278cf5dfbb3106bdbd`.



The earlier CK `8aace16` all-cell PV off/on product pair also matched the
same 16k GB10 boundary, but warm TTFT increased from 12212.8045 to
18363.0420 ms and seed plus retry from 198834.0260 to 225885.3303 ms.
Keep all-cell PV off. Evidence:
`benchmarks/correctness/runtime-tail-prefix16k-20260915.json`, SHA256
`1266d08e7416f5a80d5f272fe6c8182552acd102e211ebc5956c094af8f892f6`.

The all-cell PV implementation `8aace16` passes 54 native generated cases,
including query start 264735, against original integer K16 raw accumulators,
denominators and outputs with guards. Both same-executable q7169 capture
routes match all 29364224 GB10 BF16 cells. Completed host time is
662.297 ms off / 1432.540 ms on; PV is 226.316 / 990.149 ms. The 12 long
throughput cases show a benefit only for the high-candidate generated input.
The initial capture audit incorrectly required skipped stages; that failed
run produced no numerical report. `3c6fa88` fixes the capture observer, and
both controls were rebuilt and rerun. Its focused sanitized test passes;
the unchanged full-suite baseline is explicitly reused (399 tests, two skips).
Provider source `84de7c0` also corrects the corresponding stage-order and
completion-count checks. Its actual-provider test covers both PV layouts and
workspace owners, partial slabs and every active-stage completion failure.
C checks, 400 Python tests with two skips and hygiene pass. A native profiled
product run remains pending; the product pair above has profiling disabled. Component evidence:
`benchmarks/correctness/long-all-cell-pv-components-20260915.json`, SHA256
`8ba11178520e9e61e88ff7af80f8e86022ce26d1f713e7cfb6ae22df03c98503`.

Compact wide integer QK `0808052` passes 28 native generated cases and both
132-byte/68-byte layouts match all 418496528 original q7169 score slots,
plus 228 CPU dot samples per layout. All inputs and guards pass. Including
operand preparation, the two layouts take 556.3816 / 550.5903 ms against
304.9548 ms for the selected prepared-decoded QK. The compact experiment
remains outside product dispatch. These are component measurements without
model loading or token generation. Evidence:
`benchmarks/correctness/compact-wide-integer-qk-20260915.json`, SHA256
`2cfd5526846532b8445ec5d858fd59b25cf3cd82f97e20e45be21103b0e656ab`.

Runtime limits now distinguish model metadata 262144, maximum prompt 263168,
request including output 263680, and attention storage 264736. The original
262144-row RoPE prefix is byte-identical to GB10; all 16943104 BF16 cells
of the extended table match the original MRoPE constructor. Actual loaded
row counts remain authoritative, including rejection beyond the older
artifact. Local validation at `24c4304` passes 398 Python tests with two
skips, C syntax/smoke and hygiene; unchanged Rust/Cargo reuse the prior 47
tests and clippy. Native whole/CLI and the product cases above now qualify
this source at their measured shapes. Windows Rust server `24c4304`
also builds and passes 42 native unit tests, including the shared C context
getter. Current-package native HTTP is qualified above; real 128k/256k
continuation remains open. Server and
provider-observer evidence: `benchmarks/correctness/runtime-tail-server-and-provider-phases-20260915.json`,
SHA256 `6695c6b47bbf4e22dabc6b772a392519b0ddf0024ebc2feef05a43040b6eba59`. Bound and table evidence:
`benchmarks/correctness/extended-rope-runtime-bounds-20260914.json`, SHA256
`fa98010efa0ed10ca54574e3ef0631daea24e540ae70f6980601ecd285cc357b`.

The earlier `2fa30fa` 128k prefix baseline was explicitly cancelled by Codex
after 6521194.083 ms and 90112 completed owner tokens. Its latest complete
8192-token chunk took 1231680 ms; no owner first token or suffix output was
produced. Owned-process cleanup passed. It is an incomplete measurement
without a numerical verdict, and is no longer running. Evidence:
`benchmarks/correctness/prefix128k-baseline-incomplete-20260914.json`, SHA256
`45cb8e6d04e7a780c6bdd823a1325992c6dd25fa1d8e77f87b85729bd1fc1d20`.

The immutable retained-performance, 128k/256k context, relocated package,
HTTP, one-hour soak and release gates remain open. The older analytic
address fixtures and the 16k product pair establish no 128k/256k inference
claim. Continue structural work from measured walls and GB10 boundaries.

Compact staged-half routed MoE at `4a7f0c4` reduces complete q8192
TTFT from 33507.2252 to 32939.7689 ms in the same-DLL off/on comparison,
saving 567.4563 ms. Both baiying runs with `D:\models\Qwen3.6-35B-A3B`
preserve all 512 GB10 tokens and actual callbacks, first 144/raw 10.375.
TPOT is 100.802963 / 101.367740 ms and model plus engine load is
21265.1042 / 21328.8211 ms. Only
`QRT_QWEN36_MOE_STAGED_HALF_REPLAY=0/1` differs between the two runs.

That MoE comparison used the option on with
whole `0aa358b` staged dense replay on, QKV4/OUT0, linear OUT PPB1000,
shared prevalidated MoE, CK `9e6e296` prepared QK, FLA `2ee6215` and
CLI `a797b62`. Both source options default off. This pair does not establish
repeatability or retained performance. TTFT is still above 10 seconds;
the immutable speed, context/prefix, package, HTTP, soak and release gates
remain open.

The routed option adds 1283457024 bytes in two lossless operand arenas,
reused from gate/up to activated/down. Both allocation and preparation are
included in product time. Operand preparation occurs even on a cached norm
hit; all 80 original norm-cache hits, 40 shared calls and 110 dense calls
remain. Native shared and routed comparisons, original norm scans, live-view
reconstruction, unsupported-group fallback, guards and both builds pass.
The synthetic 1025-token routed case is slower, demonstrating that the real
product boundary is necessary for the route decision. Host allocation and
submission fault tests, C smoke, 392 Python tests with two skips and hygiene
pass; unchanged Rust/Cargo sources reuse 47 tests and clippy with explicit
hashes and diff. Evidence:
`benchmarks/correctness/staged-half-moe-product-20260914.json`, SHA256
`eb192d32150a1e7e4f238fe28c801e519088001c77ec03d82631342e313ca117`.

The earlier compact staged-half dense replay at `0aa358b` reduces complete q8192 TTFT
from 34160.5916 to 33633.9952 ms in the same-DLL off/on comparison, saving
526.5964 ms. Both runs on baiying with `D:\models\Qwen3.6-35B-A3B` match
all 512 original GB10 output tokens and actual callbacks, first 144/raw
10.375. TPOT is 101.697641 / 100.786339 ms and model plus engine load is
21170.0292 / 21157.2727 ms. The only environment difference is
`QRT_QWEN36_HAWKEYE_STAGED_HALF_REPLAY=0/1`.

That pair used whole `0aa358b` with the option on, QKV4/OUT0, linear OUT
PPB1000, MoE `1958c2c`, prepared-QK CK `9e6e296`, FLA `2ee6215` and
CLI `a797b62`. The source default remains off. This
single product pair does not establish repeatability or the immutable speed,
context/prefix, package, HTTP, soak and release gates. TTFT remains above
10 seconds, so continue broader replacements rather than local parameter tuning.

All 110 applicable dense calls preserve original candidate counts, windows,
exact dispatch counts and host count reads. Preparation of both lossless
operands is included in correction and TTFT; the largest actual view is
94371840 bytes and is freed after each call. Dense correction reductions are
263.792 / 70.187 / 83.752 ms for linear QKV / Z / attention QKV and
85.644 ms for the 40 OUT calls. These are nested diagnostic scopes. The
component improvement is therefore substantially smaller in the whole model.
Native build, actual host-owner fault tests, C smoke, 392 Python tests with
two skips and hygiene pass. Unchanged Rust/Cargo sources reuse the prior
47 tests and clippy with explicit hashes and diff. Evidence:
`benchmarks/correctness/staged-half-dense-product-20260914.json`, SHA256
`d7a82cd60f3f33356dd7645fa276840fc2756de88af0dfff48a6be49683913a8`.

The earlier prepared decoded QK comparison at `9e6e296` reduced complete
q8192 TTFT by 1247.7049 ms.
The same CK DLL with the option off/on records 35434.9430 / 34187.2381 ms,
TPOT 101.231987 / 101.500243 ms and model plus engine load
21227.4292 / 21355.9752 ms. Both runs on baiying with
`D:\models\Qwen3.6-35B-A3B` preserve the original prompt, all 512 GB10 tokens
and actual streaming callbacks, first token 144 and raw logit 10.375.
The only environment difference is `QRT_CK_SM121_PREPARED_DECODED_QK=0/1`;
profiling and shadow audits were disabled. That pair used whole `02f02eb`
QKV4/OUT0, linear OUT PPB1000, shared prevalidated MoE `1958c2c`,
FLA `2ee6215` and CLI `a797b62`.
This pair establishes a correctness-attached gain, without establishing
repeatability or the immutable retained-performance, context, package, HTTP
and soak gates. Release remains unqualified. Evidence:
`benchmarks/correctness/prepared-decoded-qk-product-20260914.json`, SHA256
`bc262fd953a27d67d8a03519de937bf1cbca108ec784d63d33e9d5c1e1f34a1b`.

Scaled-significand fallback at `912e620` passes complete same-DLL q8192
comparisons without a useful TTFT gain. Both flags off/on record
34105.1229 / 34104.6221 ms, a 0.5008 ms difference; TPOT is
101.453644 / 101.873315 ms and load is 21286.1193 / 21274.0603 ms.
Both runs preserve all 512 GB10 tokens and actual callbacks, first 144/raw
10.375. Keep the original fallback and the selected whole/MoE stack above.
The candidate reuses existing row flag storage for two eligibility bits,
preserves fast scalar-float rows, substitutes exact scaled significands only
for other ordinary BF16 rows, and retains special/subnormal fallback.
The original FP64 norm reduction, candidate admission and workspace sizes
are unchanged. Eight retained and 18 new native reports pass, including
24985600 routed raw outputs, 22615040 shared BF16 endpoints and 507677 selected
scaled-fallback cells. Test-only `1709189` repairs missing host mock entries
and checks actual dispatch ownership, shape isolation and failure cleanup;
full local checks pass. Both builds and the complete product boundary are
attached. No repeatability, performance or release acceptance follows.
Evidence: `benchmarks/correctness/scaled-fallback-product-20260914.json`, SHA256
`76b0a207a31e0744236bd63c3ce566cae35b378ffe4e998ad33978ca96da2877`.

A fresh profile of the selected stack requalifies all 512 GB10 tokens and
callbacks with only four diagnostic flags changed. Completed attention phases
sum to 8863.5135 ms: QK 3694.0812, probability 1368.1306, approximate PV
1755.7638 and exact PV 1840.9749 ms. Dense OUT correction contributes
3552.6770 ms across 40 calls and 164967882 selected cells; routed MoE
gate/up/down corrections contribute 1530.198904 / 1962.433089 / 1570.502992 ms.
These scopes are nested within layer totals, and shared MoE overlaps routed
work. All ten negative residual/postnorm event intervals are excluded.
Completed host attention core is 12910.9 ms; allocation/free is negligible.
Instrumented TTFT 35495.3171 ms is diagnostic and does not replace the
uninstrumented retained measurement. This prompted the interleaved-output
comparison below, preserving original K16 recurrence and fallback.
Evidence: `benchmarks/correctness/current-prepared-stack-profile-20260914.json`,
SHA256 `f422fb486d5529c349e0f9732ebef8164d49e42c5bfe74219c8eb3102f2ebb5b`.

Interleaved exact outputs at `8d43545` preserve correctness but are slower.
Complete q8192 QKV preparation-plus-replay medians are 59.0527 ms for the
original, 67.2572 ms for two outputs and 79.4720 ms for four outputs per
four-lane subgroup. All 67108864 GB10-comparable BF16 endpoints and 4331311
raw selected accumulators match, with unchanged candidates, row flags and
workspace. Twenty-four native cases preserve 2773512 ordered K16 states,
shared/independent inputs, partial output groups, unaligned operands, guards
and independent production parity. Full local checks pass. Keep the original
runtime dispatcher; no product or release acceptance is inferred.
Evidence: `benchmarks/correctness/interleaved-projection-comparison-20260914.json`,
SHA256 `7bdd14e9971eab71f15af72f25d010037a4fe980fb0a7f08f99884ab062976da`.

The complete OUT L1 audit at `b851e96` rejects a useful uniform envelope
replacement. All 40 layers and 671088640 BF16 endpoints are compared for
each coefficient while the unchanged control preserves all 512 GB10 tokens
and callbacks, first 144/raw 10.375. Multipliers 1 / 4 omit 80390096 /
5469611 candidates but miss 1110 / 2 corrected endpoints. The two remaining
coefficient-4 misses are in layers 0 and 4; CPU original arithmetic confirms
both control endpoints. Multipliers 8 / 16 / 32 have no omitted endpoint
errors on this case, but omit only 2073917 / 386244 / 7328 of 164967882
candidates. The smallest passing multiplier saves only 1.2572% of replay
candidates before magnitude preparation and multiplication, so it is not
integrated into the runtime selector. Keep the selected stack above.

The audit has zero baseline undercoverage, nonfinite outputs, invalid
magnitude/flag writes or redzone failures. Native full OUT magnitude checks
16777216 independent dyadic outputs and 10240 flags; twelve boundary cases
cover another 110430 cells with 4726 fallbacks. All pass. Full local checks
pass after a test-only pool-harness repair at `623c7a5`; native sources are
unchanged by that repair. Instrumented TTFT 35777.5768 ms includes the audit
and is not a product-performance comparison. No selector, retained
performance or release acceptance follows. Evidence:
`benchmarks/correctness/out-l1-shadow-20260914.json`, SHA256
`bad4a6ea00363bf440f6b3dbb8c56913acb851a094d5a35e284edc62e752a843`.

The routed MoE consumer audit at `9a0ff2e` preserves the full 512-token GB10
boundary while comparing every original gate/up replay across 40 layers.
Of 74389409 gate and 93285356 up candidates, 7197245 / 7976469 have invariant
SiLU-table or activated BF16 outputs over the proposed range: 9.6751% /
8.5506%. All 71583298 / 89963965 valid ranges contain the corrected BF16
endpoint, with zero false invariant certificates, invalid values or guard
failures. The original adjacent endpoints and outward Cauchy error interval
are both covered; at most nine BF16 values are enumerated. No SiLU
monotonicity assumption or correction omission enters inference.

This removal fraction is too small to prioritize above the 10-second product
boundary given the measured gate/up correction scopes. Keep the original
selector and selected stack. Native safety emits 8 shared and 18 routed/shared
reports with exact comparisons; complete local checks pass 386 Python tests
with two skips, 47 Rust tests, C smoke, clippy and hygiene. Instrumented
TTFT 35186.3776 ms is diagnostic, not a performance comparison. First token
144/raw 10.375 and all actual callbacks match GB10. No release qualification
follows. Evidence: `benchmarks/correctness/moe-consumer-interval-audit-20260914.json`,
SHA256 `5f11cf90bf44c8daaa46f05bdff840b496b73d6f17db2da2591c79db26ee6371`.

Matrix QK with cooperative fallback at `f204318` is exact but slower.
All 60 native generated reports preserve 98295696 raw score
slots and 3840 independent CPU dots. Each of three complete q7169
variants preserves 418496528 unique scores across one warmup and three
measured executions: 1673986112 comparisons and 228 CPU dots per variant.
All encoded query/key rows, original operands, guards and unused score tails
pass. The internal integer carry remains intact through overflow cases.

Including preparation, selected prepared QK takes 277.6628 ms; matrix columns
32 / 64 with a block-local queue and four-lane exact fallback take 528.3471 /
734.6199 ms. Query encoding is included per slab and key encoding once.
The original matrix certificate and fallback arithmetic remain unchanged.
Full local checks pass 386 Python tests with two skips, 47 Rust tests,
C smoke, clippy and hygiene. Keep the selected runtime stack; no product,
retained-performance or release acceptance follows. Evidence:
`benchmarks/correctness/matrix-cooperative-qk-20260914.json`, SHA256
`21180a5971127fb7f7b888069d65c38745a39ae0e20228d3ef515875f6dc5fb3`.

Discarded-bit interval certification at `522653f` preserves exact K16
endpoints but makes matrix QK slower. Complete q7169 preparation-plus-query
medians are 271.2800 ms for selected prepared QK, 551.2625 ms for the original
matrix32 certificate, and 876.3462 / 794.8246 / 874.8079 ms for interval
matrix16 / 32 / 64. CPU sampling classifies 2176 old certificates, 636 new
certificates, 64 uncertain and 772 unsupported groups; these are not GPU
counters. Keep this component outside product dispatch.

Signed remainder bounds and directed integer division introduce no numerical
tolerance. All 500000 sanitized host states pass, including 142929 independent
bounds, 129633 certified results and 70360 additional certificates. Native
checks preserve 163826160 generated score slots and 6400 CPU dots. Each of
five complete captured variants preserves 418496528 unique scores across
four executions, 1673986112 comparisons and 228 CPU dots. Encodings, source
immutability, guards and unused score tails pass. Full local checks pass
387 Python tests with two skips, 47 Rust tests, C smoke, clippy and hygiene.
No product, retained-performance or release acceptance follows. Evidence:
`benchmarks/correctness/matrix-interval-qk-20260914.json`, SHA256
`85c25310ef7d123ad1f6e65e1e9e817f93d5ab46d3533cfdeede31a1d15b6802`.

Exact per-K16 PV group metadata at `13f8645` preserves the complete
original q7169 GB10 attention result but slows replay. Completed medians,
including probability metadata and one V metadata preparation, are
96.3601 ms for original integer PV and 140.5985 / 127.2659 / 131.2042 ms
for skip-only / group-float / combined. Keep the product dispatcher unchanged.

All 440 native safety reports preserve 67502080 output positions, raw
accumulators and denominators. Four complete captured variants preserve all
29364224 external BF16 cells, 2198673 original selected cells and raw control
bits across every warmup and measured attempt: 440463360 output comparisons
with no mismatches. The common control has 228 independent CPU dot checks.
Metadata, candidate ownership, guards, tails and immutable inputs pass.
Separate untimed GPU counters record 637019466 K16 groups, with 1099588 zero
skips and 91970 exponent-bound skips, only about 0.18705%. All non-skipped
groups use the exact float helper in the combined capture. K32 rescaling
and K16 rounding boundaries remain original.

The separate exact segment-reuse probe finds no repeated K4 through K2048
segments in four later-layer q7169 captures. The embedding's 257 unique
segments at every position are already covered by layer0 inverse reuse.
Sanitized host arithmetic, C smoke, 388 Python tests with two skips and
hygiene pass; unchanged Rust/Cargo surfaces reuse the prior 47 tests and
clippy with explicit diff and hashes. No product or release acceptance
follows. Evidence: `benchmarks/correctness/pv-group-replay-20260914.json`, SHA256
`93b03b125f746d277e090136e8a271c6934d6ba5687c07f2fc380629197701c1`.

Lossless contiguous WMMA PV operands at `8d7a153` preserve the original
q7169 result. Completed probability-plus-native-PV medians including V
preparation are 201.8398 ms for original, 188.8048 ms for V only, 183.4203 ms
for P and V, and 184.1490 ms for paired-K32 P and V. The best difference is
18.4195 ms in this capture; it does not establish a seconds-scale product
improvement. Keep the selected product dispatcher unchanged.

All 384 native safety reports preserve 70778880 output and 832731648
probability positions. Each of four complete captured variants preserves
418496528 probability cells, all scales, raw approximate output/accumulator
and error-bound bits across every warmup and three measured attempts.
Actual candidate sets match all 2198673 identities. Each variant runs its
own unchanged exact replay and preserves corrected raw output, accumulator
and denominator bits, matching all 29364224 external GB10 BF16 cells. The
control also has 228 independent CPU dots. Full CPU layout reconstruction,
guards, prefixes, tails and immutable sources pass.

Both preparation costs and probability dual writes are counted. Common
QK, exact replay and validation are reported separately. Local C smoke,
388 Python tests with two skips and hygiene pass; unchanged Rust/Cargo
files reuse prior 47 tests and clippy with explicit hashes and diff.
No product or release acceptance follows. Evidence:
`benchmarks/correctness/wmma-pv-operands-20260914.json`, SHA256
`96bfa8923d6388958046633d58ea11e20b2c148a0630f42d150fc9875ac7983d`.

Exact packed low-bit matrix compensation at `cce5c22` passes the complete
q7169 score boundary, but the matrix schedule remains slower than selected
QK. Completed preparation-plus-query medians are 271.4099 ms for selected
prepared QK, 633.8454 / 582.4176 ms for dense / affected-pair packed repair,
623.4462 / 604.7792 ms with the old fast certificate first, and 677.8834 ms
for original sparse-core compensation. Keep the selected product dispatcher.

The arithmetic subtracts exact signed low16 remainders from matrix sums and
repairs original BF16 encoding exceptions individually. It introduces no
tolerance. Four host policies preserve all 500000 raw carried states each;
1048576 arbitrary packed halfword pairs pass. Sampling 65536 original QK
dots covers 1048576 groups, all in range, with zero aligned-sum mismatches.
These CPU counts do not establish GPU savings. Native checks preserve
196591392 generated scores and 7680 CPU dots. Each of six complete captured
variants preserves 418496528 unique scores over four executions, 1673986112
comparisons and 228 CPU dots. All encodings, guards, unused tails and source
immutability pass.

Local C smoke, 389 Python tests with two skips and hygiene pass; unchanged
Rust/Cargo surfaces reuse the prior 47 tests and clippy with explicit diff
and hashes. No product or release acceptance follows. Evidence:
`benchmarks/correctness/matrix-packed-remainder-qk-20260914.json`, SHA256
`b06489fb1b0eb84a1530f75bbcdf256b6992229191a90882e33b6d10088646dc`.

Exact scalar integer dot4 QK at `9d81ce9` preserves the complete captured
q7169 score boundary but is slower than selected prepared QK. Completed
preparation-plus-query medians are 274.4223 ms for selected QK,
731.6920 / 757.9174 ms for K64 sparse / fast-first compensation and
945.8018 / 1037.9965 ms for K128. Every query encoding and one key encoding
are included. Keep the selected product dispatcher.

The installed HIP 7.1 compiler emits ordinary mul/mad for its signed/unsigned
16-bit dot2 interfaces and native dot4 for its byte interfaces. This is a
compiler-only observation; the separate GPU preflight checks all four byte
sign policies on 1048576 random/edge packed pairs against an independent CPU
reference. The failed first compiler-probe bootstrap used a non-git working
directory; its evidence is preserved, and the corrected probe uses the source
checkout. No compiler or GPU work ran in the failed attempt.

The new scalar schedule decomposes signed16 cores into four exact byte dot
families and preserves packed low16 compensation, BF16 encoding exceptions
and original raw carry states. Both host policies preserve 500000 states each.
Native tests preserve 163826160 generated scores and 6400 CPU dots; each of
five complete q7169 variants checks 1673986112 raw scores over four executions
and 228 CPU dots. All encodings, guards, unused tails and immutable inputs
pass. Local C smoke, 390 Python tests with two skips and hygiene pass; unchanged
Rust/Cargo files reuse the prior 47 tests and clippy with explicit diff/hashes.
No product, retained-performance or release acceptance follows. Evidence:
`benchmarks/correctness/scalar-integer-qk-20260914.json`, SHA256
`edd783d3e4b8392c7264dd7f5ec8f3497428f2497eb34a0c96a7c426621b6656`.

Lossless prepared product-pair QK at `921da8a` preserves every captured
q7169 score but remains slower than selected prepared QK. Completed
preparation-plus-query medians are 274.8766 ms for selected QK and
448.2824 / 445.0088 / 442.9941 ms for paired K64 / K128 / K256 windows.
Keep the selected product dispatcher.

Each K16 row occupies 36 bytes with paired exponent/significand halfwords and
a separate sign mask. Packed integer multiply/max preserve original BF16
products, exponent alignment and raw carry normalization. Every BF16 encoding
roundtrips. Host checks cover 1048576 edge/random product pairs, 1048576 packed
maximum pairs, 8000000 group products and 500000 raw carried states. Native
packed arithmetic covers every BF16 encoding against 16 controls and 65536
raw states; generated QK checks cover 131060928 scores and 5120 CPU dots. Each
of four complete captured variants checks 1673986112 raw scores over four
executions and 228 CPU dots. All encodings, guards, unused tails and immutable
inputs pass. Query encoding is timed per slab, key encoding once.

Local C smoke, 391 Python tests with two skips and hygiene pass; unchanged
Rust/Cargo files reuse the prior 47 tests and clippy with explicit diff/hashes.
No product, retained-performance or release acceptance follows. Evidence:
`benchmarks/correctness/prepared-integer-pairs-qk-20260914.json`, SHA256
`bf984c9f4e1096ef4aef81969fcc0d28b83c7af9b70ba06410c36b32169213f5`.

Lossless scaled-half QK at `3f5a21d` preserves every captured q7169 score
and gives a small component improvement. Completed preparation-plus-query
medians are 271.7113 ms for selected QK and 258.7275 / 256.7067 / 257.5802 ms
for K64 / K128 / K256. The best saves 15.0046 ms. It remains outside product
dispatch; no q8192 or retained-performance claim follows.

One exact power-of-two scale maps supported BF16 rows to normal FP16 operands
without losing bits. Scalar mixed FP16/FP32 instructions produce exact FP32
products; original exponent alignment and raw K16 carry normalization remain.
Unsupported rows/arithmetic reconstruct the original BF16 words and use the
wide primitive. All BF16 encodings roundtrip in four host contexts; 500000
raw states pass. A separate native instruction probe validates 2097152 products
and both packed halves. Its first oversized encoded-command launch failed
before compilation; the successful command-file route and failure are both
preserved.

All captured Q/K rows support the view. Sampling 65536 complete original dots
covers 1048576 raw K16 states without fallback or mismatch; these are CPU
diagnostics. Native preflight validates 131072 raw states with actual path
counts. Generated QK checks cover 131060928 scores and 5120 CPU dots; each of
four complete captured variants checks 1673986112 scores and 228 CPU dots.
All encodings, guards, unused tails and immutable inputs pass. Query encoding
is timed per slab, key encoding once. Local C smoke, 392 Python tests with
two skips and hygiene pass; unchanged Rust/Cargo files reuse 47 tests and
clippy with explicit diff/hashes. Evidence:
`benchmarks/correctness/scaled-half-qk-20260914.json`, SHA256
`cd69c6e25e6d40b53291817141425715c19f567a89494c05a431a3156ebdef83`.

Lossless scaled-half projection at `dc6bbd2` passes the complete q8192 QKV
boundary but is slower than selected four-lane replay. Preparation-plus-replay
medians are 60.3421 ms for the control, 139.339 ms for scaled single-lane and
66.1409 ms for scaled four-lane. The single-lane samples vary substantially;
all samples remain attached. Keep the selected product dispatcher.

Direct integer carry alignment removes the earlier floating carry range
restriction. Both host policies preserve 500000 raw states; fallback counts
drop from 188610 to 80512 in the adversarial host distribution. Native tests
cover six widths and 693378 raw states per lane configuration, including
486020 transformed and 207358 original groups. All signs, exponents,
encodings, guards, unaligned operands and production/audit parity pass.

All 1048576 captured weight groups and 1048576 input groups support the new
view, including the 112 weight rows outside the control floating predicate.
The view occupies 75497472 bytes. Every warmup and timed attempt checks all
67108864 GB10-comparable BF16 outputs and all 4331311 unrounded candidates;
producer, selector and candidate identities remain unchanged. Preparation
is included and variant order rotates. Local C smoke, 392 Python tests with
two skips and hygiene pass; unchanged Rust/Cargo files reuse 47 tests and
clippy with explicit diff/hashes. No product or release qualification follows.
Evidence: `benchmarks/correctness/scaled-half-projection-20260914.json`, SHA256
`6fcde1d0bfc968657b83c92d3486e8676ad7f4e01fc9abb2342f045336405433`.

Compact staged-half projection at `5413a9e` improves the complete q8192
QKV component. Including complete operand preparation, original four-lane
replay takes 59.1891 ms; compact staging 1 / 2 / 4 / 8 takes
47.0711 / 43.5192 / 44.4076 / 44.1382 ms. Two groups save 15.6699 ms,
about 26.47%. Three samples per variant and rotated order are attached.
This warrants provider integration and full-model measurement; it does not
establish product performance or release qualification.

Each lane stores six operand dwords per group before the original ordered
integer carries. All five variants preserve every GB10-comparable BF16
output and all 4331311 unrounded candidates after every warmup and timed
attempt: 1342177280 output and 86626220 raw candidate comparisons pass.
All captured groups support the 75497472-byte lossless operand view.
Twenty-four native tests preserve 2773512 raw carry states across six widths,
including staging tails, unsupported groups and production/audit parity.
All encodings, guards and immutable inputs pass. Initial compilation rejected
two host-only memcpy calls; the successful revision uses device builtins,
and the failure is retained. Local checks and explicit unchanged-code reuse
are attached. The selected product stack remains unchanged by this component.
Evidence: `benchmarks/correctness/staged-half-projection-20260914.json`, SHA256
`a6807458b081d078c9e06914c6eeb22684b1bbf367c3582e7af9e3b0b804e7ed`.

Bounded same-stream submission at `22504b3` passes complete same-DLL
q8192/out512 comparisons at 1 / 8 / 64 slabs per completion. All original
GB10 tokens and actual callbacks match, first 144/raw 10.375. TTFT is
34254.9603 / 34145.3290 / 34005.3841 ms; the two candidates save only
109.6313 / 249.5762 ms in this comparison. Keep one slab per completion
while investigating larger costs. No stable gain or performance acceptance
is inferred. Prepared QK remains enabled; all kernels, selectors, providers
and workspace sizes are common. Host submission and completed-work deadlines,
failure drains, tail completion and other call shapes pass tests. Native
build and full local checks pass. Evidence:
`benchmarks/correctness/queued-attention-product-20260914.json`, SHA256
`3cbddbaf9d7010cb56cc5da7eecdf0f2e8df77bf6116aeab0a7b02c3efb13ad1`.

Decoded projection operands at `80d0529` are exact but produce no gain.
Original / decoded-input / decoded-both preparation-plus-replay medians are
60.6041 / 61.1360 / 80.2389 ms on complete q8192 QKV. All three preserve
67108864 GB10-comparable BF16 endpoints and 4331311 original raw selected
accumulators. Candidate storage adds 64 / 128 MiB; all preparation is timed.
Every packed word, row flag, candidate, source and guard passes verification.
Twelve native cases preserve 1386756 ordered K16 states and independent
production parity. Full local checks pass. Keep the current projection
dispatcher; no product promotion or release qualification follows. Evidence:
`benchmarks/correctness/decoded-projection-comparison-20260914.json`, SHA256
`2f8c7346c670ea58bb254bd8612e00fdb289da68a1d419d3cbf12ce355fefd3c`.

The same-DLL OUT admission comparison at `02f02eb` recovers correctness but
rejects the performance tradeoff. QKV4/OUT0 with linear OUT PPB1000 and
QKV4/OUT4 with uniform linear OUT PPB2000 both preserve all 512 GB10 outputs
and actual callbacks, first 144/raw 10.375. Full-attention OUT stays at
PPB10000 and both runs disable the shadow audit. TTFT is 35160.3559 /
35532.2183 ms: the stronger candidate is 371.8624 ms slower. Load is
21274.1108 / 21303.5852 ms; TPOT is 101.766568 / 100.994616 ms.
Keep QKV4/OUT0 and linear OUT PPB1000. Do not infer repeatability or retained
performance from this pair. Algorithm4 OUT with the old linear bound remains
incorrect on this case. Evidence: `benchmarks/correctness/out-matrix-bound-product-20260914.json`,
SHA256 `720451ec5387b3936b101ca9f9b3c3ad7c45200e8f9f6029942ce1dd50589363`.

The complete spatial-replay comparison at `f5e2a4c` produces no gain.
Original order and tiles 64x32 / 128x64 / 256x128 preserve all 67108864
GB10-comparable QKV BF16 endpoints and all 4331311 raw selected accumulators.
Preparation, bitmap construction, reordering, count readback and replay take
58.8072 / 60.1234 / 59.3797 / 68.1507 ms, respectively, as medians of three
completed samples after warmup. Every original candidate, K16 operation and
admission bound is retained. Full permutation, CPU row flags, immutable
sources and redzones pass. The 7169 original input rows plus 1023 repeated
rows retain independent GB10 comparison; references never enter computation.
No alternative enters runtime dispatch. Native build/test and full local
checks pass: 381 Python tests (two skips), 47 Rust tests, C ABI smoke,
clippy and hygiene. These are component results, not product acceptance.
Evidence: `benchmarks/correctness/spatial-projection-comparison-20260914.json`,
SHA256 `edaab7b28e021560ac9f6f26e78b42b17dea7db22dc95058516cba690dbf320d`.
Broader reuse of exact arithmetic or provider replacement remains open.

Packed products and certified K16 batches at `809a44e` preserve complete
QKV correctness but are slower. Original / packed-only / four-group /
eight-group preparation-plus-replay medians are 56.8519 / 66.4194 / 71.4928 /
71.0080 ms. All four variants preserve 67108864 GB10-comparable BF16 endpoints,
4331311 candidate identities and raw selected accumulators, metadata, guards
and immutable inputs. No variant enters product dispatch.

Host checks cover 12720708 packed products and 1318560 ordered carries;
15 native cases compare 1881219 ordered states exactly. A test-only follow-up
at `3e8ba49` resets output before independent production/diagnostic parity;
the unchanged arithmetic passes again. Full local checks pass 382 Python
tests (two skips), 47 Rust tests, C smoke, clippy and hygiene. This is
component evidence, not product performance or release acceptance.
Evidence: `benchmarks/correctness/packed-float-tiles-comparison-20260914.json`,
SHA256 `4021f7a331d847fd2e1cf567f4488d363f7b76fd91fa2e9b8fb5f607b461ec84`.
Complete-row staged probabilities at `dd058ba` preserve the original ordered
FP32 denominator recurrence. All 240 native cases pass, comparing 433191552
probability cells and 13696512 scale cells, all denominator bits, guards and
immutable inputs. Original versus staged q7169 attention preserves all
29364224 external GB10 BF16 endpoints, 2198673 correction candidates and four
complete output artifacts bit for bit. Completed preparation-plus-query wall
is 670.7340 / 660.9703 ms; query-only reduction is 9.7614 ms. This component
gain does not justify product promotion while q8192 remains above 10 seconds.
The provider default stays unchanged. A missing host mock for the new kernel
was corrected, and full local checks pass 382 Python tests (two skips),
47 Rust tests, C smoke, clippy and hygiene. Evidence:
`benchmarks/correctness/staged-probability-comparison-20260914.json`, SHA256
`846d6153601d9515aa888864f32b6477694b69bd6219522d0362aa9e49eb7bf4`.
Two-lane ownership at `a8e4f41` is exact but slower. Six native shapes retain
all 693378 ordered K16 states, original eligibility and fallback decisions,
unaligned operands, partial blocks, guards and independent production parity.
Complete QKV preserves all 67108864 GB10-comparable BF16 endpoints and
4331311 raw selected accumulators. Preparation-plus-replay medians are
59.4497 ms for four lanes and 64.3386 ms for two, a 4.8889 ms regression.
The runtime retains four lanes. Full local checks pass. Evidence:
`benchmarks/correctness/dual-lane-projection-comparison-20260914.json`, SHA256
`e43041d47fa3554be6e38be1bad7ea29cc6f14c828d29cc68d1f07414f713bbe`.
Prepared packed QK at `a63d809` preserves all 418496528 original q7169 score
cells and 228 independent CPU dots in each of eight variants. Final native
safety passes 160 cases; every encoded word, row flag, input and guard passes.
Complete preparation-plus-QK medians are 357.6721 ms for the current scalar
control, 323.9206 ms for the earlier faster decoded geometry and 285.3475 ms
for prepared K128 with 16 query rows, 16 keys and direct FP32 carry. The latter
includes 2.2129 ms for all Q/K preparation and saves 72.3246 ms (20.22 percent)
against the current component control. Original K16 arithmetic and complete
fallback are unchanged. The failed r2 capture path was corrected before GPU
computation; r3 uses the identical executable and original capture hashes.
Evidence: `benchmarks/correctness/prepared-decoded-qk-20260914.json`, SHA256
`1ef1bf444c5de61da77f0c18f77cf2a4b1955e930dbf2e93841e30e575b70b0c`.

An opt-in provider integration uses `QRT_CK_SM121_PREPARED_DECODED_QK=1`
for cold prefill through q8192 with the existing scalar-float and global PV
options. It refreshes a 151584768-byte arena every layer/call under the existing
workspace lock and drains failed submissions. Decode, suffix and larger calls
keep their established route. The default is off; the selected experiment
explicitly enables it after the complete same-DLL q8192 qualification above.
The initial native build exposed ambiguous imported attention constants;
explicit namespace qualification repairs compilation without changing
arithmetic. Both build records, final DLL identity and full local validation
are attached to the product evidence.

The scoped matrix replacement at `570dc90` retains a correctness-attached
2090.0644 ms TTFT reduction. On baiying with `D:\models\Qwen3.6-35B-A3B`,
the same DLL's control and QKV/Z-only algorithm 4 both preserve all 512
original GB10 outputs and actual callbacks, first token 144 and raw logit
10.375. TTFT is 37486.0783 / 35396.0139 ms; QKV/Z-only TPOT is
101.25411 ms and model plus engine load is 21289.6057 ms.

Use `QRT_QWEN36_Q8192_MATRIX_PRODUCER_ALGORITHM=4` with
`QRT_QWEN36_Q8192_MATRIX_PRODUCER_SCOPE=qkv` for subsequent experiments,
retaining shared prevalidated replay. The run records 70 algorithm-4 QKV/Z
calls, 40 algorithm-0 OUT calls and load-time creation of the three selected
plans. MoE `1958c2c`, CK `4a5a5b0`, FLA `2ee6215` state 8 and CLI `a797b62`
remain common. Norms, selectors, exact arithmetic and all bounds are unchanged.

OUT-only algorithm 4 fails at index 4, actual 220 versus expected 79, with
37/512 matches. Its full output sequence exactly matches the prior all-shape
failure. OUT therefore keeps algorithm 0. The subsequent shadow diagnostic below localizes four linear OUT
admission misses. No broad algorithm-4 acceptance is implied. Full local checks pass 380 Python tests (two skips), 47 Rust tests,
C ABI smoke, clippy and hygiene. This single product comparison does not
establish repeatability, retained performance or release readiness.
Command: `run-native-matrix-scope-product-r1.ps1` with control/qkv/out arguments.
Evidence: `benchmarks/correctness/matrix-scope-product-20260914.json`,
SHA256 `59a10e7201ac33512a40e5eea30d668f842e924cd48b7e5bf68ddea826c553d9`.

The standalone hipBLASLt diagnostic at `95a33b3` reproduces all four raw
FP32 mismatch counts without engine code and with alpha/beta live until GPU
completion. All 167772160 generated BF16 endpoints, finite outputs, source
immutability and memory guards pass. The loaded ROCm 7.1 DLL is 6012312 bytes,
SHA256 `9f97d95c3b7257efcab992914b4e1c1b3b71b0693a6ad5d61a8f3cbf08a96dee`.
Official logs confirm the requested datatypes, T/N layout, alpha 1, beta 0,
zero workspace, and solutions 5661/5651 for heuristics 0/4. Native build/test
complete in 2659.578/1185.241 ms with host guards passing. These are diagnostic
process times, not inference performance. This excludes engine code and the
wrapper's scalar lifetime as necessary causes of the observed raw difference;
it does not establish a hardware defect or explain OUT continuation failure.
Evidence: `benchmarks/correctness/hipblaslt-dyadic-standalone-20260914.json`,
SHA256 `66337f9be1e6799a286b57e1300d1cf615de16e8dcc11b06aeedcb9b1fd44370`.

The shadow diagnostic at `02f02eb` localizes four OUT admission misses.
QKV/Z 4 and OUT 0 drive inference and preserve all 512 original GB10 outputs
and callbacks, first 144/raw 10.375. Separate guarded algorithm-4 outputs use
the same real inputs, norms and correction. All 40 OUT projections compare
671088640 BF16 endpoints. Linear-attention layers 10/16/22 differ in 1/1/2
cells; the other 27 linear layers and all ten full-attention layers match.
Original integer wave16 arithmetic validates all four control endpoints and
rejects all four shadow endpoints, each missed by the original selector.
The first differing cell in each affected layer also matches CPU raw bits.
Every diagnostic redzone passes; shadow outputs never drive inference.

These observations localize selector undercoverage with algorithm 4 at
linear OUT PPB1000/radius512. They do not establish a universal bound or a
hardware defect. The next same-DLL product comparison uniformly tests linear
OUT PPB2000 with algorithm 4; full-attention OUT retains PPB10000 and all
other settings are common. The control keeps QKV4/OUT0. Both runs disable
shadow instrumentation and require the full GB10 boundary before any timing
comparison. The shadow run's TTFT39488.1618 ms is diagnostic only.
Native build completes in 91685.814 ms and full local checks pass.
Evidence: `benchmarks/correctness/out-matrix-shadow-20260914.json`,
SHA256 `752eb62c5cd7921c10fadbe45bd1a32120a3770f5dc8799b50c4f1050d6a559c`.

The same-DLL product trial at `c16219f` rejects matrix algorithm 4. On baiying
with `D:\models\Qwen3.6-35B-A3B`, algorithm 0 preserves all 512 original GB10
outputs and actual callbacks, with TTFT 37413.9499 ms, TPOT 101.896379 ms and
load 21304.8867 ms. Algorithm 4 takes 34435.7089 ms but only 37/512 tokens
match; the first difference is index 4, actual 220 versus expected 79.
Both first tokens are 144 and raw logits are 10.375. First-token success
and a faster incorrect continuation do not establish performance acceptance.
The scoped trial above supersedes the initial decision to retain algorithm 0 everywhere.

The subsequent exact-replay comparison at `0f80fe6` also produces no gain.
Current validated replay, prior range normalization and new strong-row replay
all preserve 58728448 original GB10 QKV BF16 outputs and 3791742 unrounded
selected accumulators. Preparation plus replay medians are 49.6012, 49.5987
and 50.9699 ms from three completed samples after warmup. Both certificates
cover the same 3028041 selected cells; all flags and memory checks pass.
Neither alternative enters product dispatch. Their earlier 30 native cases
and 3762438 raw K16 comparisons remain arithmetic evidence.
Evidence: `benchmarks/correctness/strong-projection-comparison-20260914.json`,
SHA256 `0f48444a387cd2edea36759aebe394051d8046b58f458ddf4f54d50f1dff0b7c`.
The subsequent full-model scope diagnosis above separates QKV/Z and OUT
replacement while keeping all original bounds and the GB10 boundary.

Both native captured-QKV variants preserve all 67108864 BF16 cells, all
original 7169 input rows plus 1023 independently comparable repeated rows,
guards and immutable operands. Their full-model runs verify all 110 matrix
calls and, when enabled, load-time prewarming of four algorithm-4 plans.
That initial run did not localize its internal divergence; the later shadow
diagnostic resolves this gap. Full local checks pass 379 Python tests (two skips), 47 Rust tests, C smoke, clippy
and hygiene. The current stack still requires performance, prefix/context,
package, HTTP and soak qualification. All release gates remain open.
Command: `run-native-matrix-algorithm-product-r1.ps1`, source `c16219f`.
Evidence: `benchmarks/correctness/matrix-algorithm-product-20260914.json`,
SHA256 `dac900103ec9e2dcb02ab1ff9bf84ea6e821c2966d8c38174b45f207a7a47a36`.

The full-q8192 matrix comparison at `d49e6a9` completes 128 choices on
baiying: 72 available cases pass every generated BF16 endpoint and memory
check, and 56 are unavailable before submission. Native FP32 algorithm 4
costs 11.9971 / 5.3367 / 12.9838 / 5.176 ms at rows/K
8192/2048, 4096/2048, 9216/2048 and 2048/4096, versus
49.44 / 25.888 / 55.9235 / 24.6237 ms for algorithm 0.
These are three-sample completed host medians after warmup; the inferred
2947.131 ms aggregate reduction is a component estimate, not product timing.

The first two revisions incorrectly required raw FP32 equality from the
approximate producer. Their failed records are preserved. Revision 3 restores
the existing BF16 endpoint check and retains raw FP32 differences; it does
not weaken the GB10 model boundary. The underlying raw arithmetic difference
has not been attributed to a hardware cause.

`QRT_QWEN36_Q8192_MATRIX_PRODUCER_ALGORITHM=4` is now an opt-in for these
four FP32-output shapes. Load-time prewarming covers the same plans; default 0,
explicit-index calls, BF16 outputs and other token counts keep their policy.
The existing exact correction, PPB bounds and norms remain in place. The
captured-QKV comparison passes, but the complete product trial above fails
continuation correctness. OUT algorithm 4 remains an unqualified opt-in pending stronger admission.
Evidence: `benchmarks/correctness/matrix-producer-choices-20260914.json`,
SHA256 `3a546ebf5a09d7fae2e1152defc42f42a85b963566ddf67bc0057a77c68a8700`.

Shared-expert prevalidated exact replay at `1958c2c` passes both complete
q8192/out512 runs on baiying with `D:\models\Qwen3.6-35B-A3B`. The same MoE
DLL OFF/ON preserves all 512 GB10 outputs and actual callbacks, first token
144 and raw logit 10.375. Whole `a50f15d`, CK `4a5a5b0`, FLA `2ee6215`
state 8, CLI `a797b62` and all ten AOT files are common.

Actual callback TTFT is 37867.3011 / 37527.5887 ms OFF/ON, a 339.7124 ms
reduction in this single pair. Enabled TPOT is 101.594291 ms and model plus
engine load is 21368.4516 ms. Use the enabled route for the next experiment;
this does not establish repeatability or retained performance. The immutable
4187.415605 ms TTFT and 35.502151 ms TPOT targets remain open.

Eight native shared comparisons preserve 22658560 BF16 cells per variant,
original norm bits and eligibility flags, including integer fallback. Six
routed regressions preserve raw output bits and norms. Dedicated shared
metadata uses 77824 bytes; the q8192-only option remains off by default.
Selectors, arithmetic, K16 order, PPB bounds and short-shape routes are
unchanged. Full local checks pass. No package, prefix, HTTP or release gate
is inferred from these runs. The synchronized enabled profile also passes
all 512 GB10 outputs and callbacks. Its complete MoE total is 7120.002749 ms,
down 354.568236 ms from the previous OFF profile. Shared-stream intervals
overlap routed work and are not independent kernel times.

Completed linear projection/core totals remain 6571.398 / 7797.212 ms;
the full-attention host bucket is 14275 ms. Ten negative residual/postnorm
GPU event intervals are excluded. Existing q7169 inputs show no repeated
rows in four later-layer captures, so broad row deduplication lacks evidence.
The previous full q8192 matrix-producer profile measures 3925.7338 ms of
completed hipBLASLt work with implementation 0. The subsequent full-shape
comparison above motivates an opt-in algorithm replacement. All product
correction bounds remain fixed.
Profile evidence: `benchmarks/correctness/shared-prevalidated-profile-20260914.json`,
SHA256 `f82dcdb99e9d1f527f8b257b5d9eb163cce364d4d369e8bc4aae589c19537057`.
Command: `run-native-shared-prevalidated-product-r1.ps1` with the recorded
OFF/ON arguments. Evidence: `benchmarks/correctness/shared-prevalidated-product-20260914.json`,
SHA256 `42ba61141f4852a9cb01d4cc0c7e825b29c1348c20dda6b32ce6c84e26358d46`.

The complete QKV interval-filter experiment at `be477fa` preserves all
58728448 GB10 BF16 cells and the original 3791742 selected identities.
It certifies only 56913 candidates (1.50 percent), leaving 3734829 for exact
replay. All replayed FP32 values match the control by identity, all 37 native
safety cases pass, and source/output/partition guards remain intact.

Total preparation plus filtering, compaction, count read and replay costs
458.8400 ms versus 51.3805 ms for unchanged exact replay. This 407.4595 ms
regression rules out product promotion on current evidence. The earlier
92.2 percent unweighted sample did not predict reduction in actual candidates.
Command: `run-native-projection-interval-filter-r1.ps1 -Kind filter|real -Action build|test`.
Evidence: `benchmarks/correctness/projection-interval-filter-20260914.json`, SHA256
`3baa42e6bad722e55c6dda32950babeb596728b5f5f20d4b8dc2f2e14ff8ecf2`.

The existing same-run q8192 GB10-qualified profile has 40 complete nonnegative
MoE detail reports. Routed gate / up / down correction totals are
1714.231794 / 2052.170724 / 1571.298102 ms. Shared-stream intervals total
4898.652916 ms and overlap routed work; do not add these scopes together.
Source inspection finds shared projections still on the original sixteen-lane
integer replay, while routed projections use validated four-lane float replay.
An opt-in implementation now extends that exact path to q8192 shared
projections with 77824 bytes of dedicated stream metadata and unchanged norms,
selectors, K16 order and short-shape dispatch. Full local checks pass 378 Python
tests (two skips), 47 Rust tests, C smoke, clippy and hygiene. Allocation-failure,
drain-before-free and shared/routed pointer ownership tests also pass.
Native safety and the same-artifact product pair are now complete as recorded
above; the option remains off by default.

Prior interval/carry probes remain in `projection-interval-20260914.json`,
`wmma-scalar-carry-envelope-20260914.json` and
`wmma-projection-envelope-20260914.json`. All narrower-coefficient
counterexamples remain attached; no product or release gate is relaxed.

The hybrid exact-integer/scalar-float QK experiment at `824a07f` passes
128 native generated cases, 257735040 score comparisons and 8192 independent
CPU dots. Eight original q7169 variants each preserve 418496528 raw scores
and 228 CPU dots; all memory guards and immutable operands pass. Full local
checks pass 377 Python tests with two skips, 47 Rust tests, C smoke, clippy
and hygiene. The independent sanitized arithmetic test covers 500000 groups
and all three exact-integer, scalar-float and original-integer branches.

The same-run scalar-float control takes 325.1009 ms plus 6.5245 ms transpose.
The best hybrid takes 538.6084 ms plus 0.4851 ms key encoding, including
query encoding. It improves its matching nonhybrid matrix control at
582.9409 ms but remains slower than scalar float. Keep it outside product
dispatch. CPU samples predict 2176 exact-integer and 1472 scalar-float K16
groups; these are diagnostic CPU classifications, not GPU counters.

A separate compiler resource check reports no scratch or register spills.
Scalar QK uses 65 VGPRs; hybrid layouts use 83–94. Their compiler occupancy
is 16 waves/SIMD, which is not a measured occupancy result. The evidence
does not support register spilling as the cause of this performance gap.
Investigate the seconds-scale dense projection replay surface using the
completed whole-stack attribution below. Product arithmetic, admission
bounds and all mission targets remain unchanged.
Command: `run-native-matrix-float-qk-r1.ps1 -Kind qk -Action build|test|capture`.
Evidence: `benchmarks/correctness/matrix-float-qk-20260914.json`, SHA256
`bf667a8b18fd696450a8469f93569d4ce819d5937bea4b8876d4f4cf9132c659`.

Earlier exact matrix-consumer layouts (`395c9ee`) and decoded QK windows
(`7d25cfd`) remain component evidence only. Their full comparisons are in
`matrix-consumer-qk-20260914.json` and `decoded-window-qk-20260914.json`.
No token-loop or product performance qualification follows from these runs.

Dense and MoE candidate grouping at source `a50f15d` passes both complete
original q8192/out512 product runs on baiying with
`D:\models\Qwen3.6-35B-A3B`. Both OFF and ON preserve all 512 GB10 outputs
and actual callbacks, first 144 and exact raw logit 10.375. The same dense
and MoE DLLs, four lanes, one MoE staged K16 group, ten AOT artifacts, CK
`4a5a5b0`, FLA `2ee6215` state 8 and CLI `a797b62` are common.

Callback TTFT is 37810.5655 / 38176.3825 ms OFF/ON. Grouping is 365.8170 ms
slower in this single pair, despite its component gains. Keep both grouping
options at 0 for subsequent work. The qualified OFF control has TPOT
101.637313 ms and model plus engine load 21296.3550 ms. This does not
establish repeatability or retained performance; the immutable 4187.415605 ms
TTFT and 35.502151 ms TPOT targets remain open.

All 48 native permutation/capacity cases, six MoE controls, four actual
dense correction shapes and both original real QKV runs pass. Full local
checks and extended address/undefined-behavior dispatch checks pass. Both
product runs select identical dense candidate counts; completed dense
correction totals are 7310.218 / 7346.269 ms. Those inclusive correction
clocks do not fully attribute TTFT. The completed synchronized OFF profile
below now directs the next structural investigation. Original arithmetic,
FP64 norms and admission bounds remain unchanged; range normalization
remains a component experiment.

The same-artifact synchronized OFF profile preserves all 512 GB10 outputs
and callbacks, first 144 and raw logit 10.375. Completed host full-attention
core time is 14284.6 ms; its nested CK event total is 9812.025 ms. The thirty
completed linear projection/core totals are 6571.388 / 7800.228 ms, with
output projection included in linear core. Forty MoE event totals sum to
7533.152 ms. These intervals overlap by scope and must not be added together.
Nine residual/postnorm event intervals are negative and excluded.

CK `9bbd5d0` now passes native HIP build and the complete q8192 GB10
boundary with `QRT_CK_SM121_PROFILE_COMPLETED_STAGES=1`. All ten attention
calls, 640 query batches and 3200 completed checkpoints pass phase coverage
and disjoint interval accounting. QK is 4666.2065 ms, probability generation
1449.7858 ms, approximate PV 1767.9649 ms, collection 110.8707 ms and exact
PV 1887.8618 ms. Entry wait, preparation and dispatch remainder account for
the rest of the 9949.2487 ms completed attention total. No interval is negative.

The diagnostic defaults to 0 and leaves Q1 synchronization unchanged.
Instrumented callback TTFT is 39229.4652 ms, TPOT 104.926239 ms and load
21328.6543 ms. These do not replace the uninstrumented control. Prioritize
cooperative QK operand decoding in bounded K windows, with the original
K16 arithmetic and fallback. Complete raw QK and external BF16 comparisons
must precede a product decision. Command:
`run-native-ck-completed-profile-q8192-r1.ps1 -CkCompletedProfile 1 -ProfileStages 1`.
Evidence: `benchmarks/correctness/completed-attention-phases-20260914.json`,
SHA256 `8a12277736733c4f94d8d8bdcc1abdd5428d7e1872773dac4caa26aa548bb460`.

The recorded instrumented callback TTFT of 39026.1248 ms is not comparable
to uninstrumented performance. Command:
`run-native-partition-product-q8192-r1.ps1 -PartitionReplay 0 -ProfileStages 1`.
Evidence: `benchmarks/correctness/current-off-stack-profile-20260914.json`,
SHA256 `da00667bec7c1b0276ab83da135d40520514bab6a26d54d1dd3a99bc222d2bdf`.

The qualified 64k stack retains its earlier sixteen-lane MoE identity.
No new prefix, Windows 128k/256k, package, HTTP, soak or release acceptance
follows. Command: `run-native-partition-product-q8192-r1.ps1 -PartitionReplay 0|1`.
Evidence: `benchmarks/correctness/dense-moe-candidate-partition-product-20260914.json`,
SHA256 `3999da7941991baa25bc488981d1589b7e5887346f1d3ae250bdb284f68f9378`.

The preceding MoE work-ownership experiment at source `cb0f266` passes a complete same-source
q8192/out512 pair on baiying with `D:\models\Qwen3.6-35B-A3B`. Both sixteen
and four lanes preserve all 512 GB10 outputs and actual callbacks, first
144 and exact raw logit 10.375. Whole `27cfc32`, CK `4a5a5b0`, FLA `2ee6215`
state 8 and CLI `a797b62` are common. Both native safety suites pass, and
all ten AOT artifacts are identical. This revisits the older integer-era
experiment under current prevalidated-float replay and registered metadata.

Callback TTFT is 39379.8088 / 38091.7431 ms for sixteen lanes with four
staged K16 groups / four lanes with one group, a 1288.0657 ms reduction in
this single pair. Four-lane TPOT is 101.798454 ms and load is 21296.2683 ms.
Use four lanes and one group for subsequent q8192 candidates; source build
defaults remain unchanged. Numerical bounds remain dense/MoE PPB 1000,
full-attention OUT PPB 10000 and routed radius 512. This does not establish
repeatability or retained performance. The qualified 64k stack below still
uses the prior sixteen-lane MoE. Broader contexts, current package/HTTP,
soak and release remain open. Command:
`run-native-moe-prevalidated-ownership-product-r1.ps1 -MoeLanes 16|4`.
Evidence: `benchmarks/correctness/moe-prevalidated-ownership-product-20260914.json`,
SHA256 `c478df7f4139e5fe855c12bed7b61bd77627aa65818ca3d1aef93cb7cbc557f1`.

Exact candidate partition at public source `156badd` passes all 24 native
permutation cases and all original QKV checks: 58728448 GB10 BF16 cells,
3791742 unrounded FP32 values, candidate identities, class order and memory
guards. Counts are 3028041 floating and 763701 integer. Full local checks pass.
With unchanged four-lane arithmetic, completed preparation plus replay is
50.8645 / 43.2025 ms for original / grouped ordering. The range-arithmetic
variants take 48.3263 / 39.7011 ms. GPU partition and raw capture writes are
included; these are single component observations.

The component result motivated the dense/MoE product comparison above;
its complete GB10-valid product regression now determines the option choice.
Range normalization stays outside product dispatch during that comparison.
No new product performance or release acceptance follows. Command:
`run-native-replay-partition-r1.ps1 -Kind partition|real -Action build|test`.
Evidence: `benchmarks/correctness/replay-candidate-partition-20260914.json`,
SHA256 `d550d9627c3426916e4fa44a8c97f70742f97ebbf811686c3b793248495be65e`.

Range-certified projection at `8b0462e` preserves all 28 native dot controls,
all 58728448 original GB10 QKV BF16 cells and 3791742 unrounded candidate
values. The finite-range proof also passes 28835840 independent host
normalization comparisons; full local checks pass. Four-lane preparation
plus replay is 52.7459 / 47.9062 ms for original / certified normalization,
with 3028041 certified candidates. This single component gain does not
establish a seconds-scale product benefit; retain the original dispatcher.

Read-only inspection finds that the 112 old-ineligible weight rows contain
229376 tiny normal BF16 values, with biased exponents 4 through 51. All
8192 weight rows differ bytewise; the first-layer input has 257 unique rows.
Do not generalize that embedding-only repetition to later layers. This motivated the candidate grouping experiment recorded above. Source bounds, original dot order and PPB 1000 remain unchanged.
Command: `run-native-range-projection-r1.ps1 -Kind projection|real -Action build|test`.
No new token-loop or release acceptance follows. Evidence:
`benchmarks/correctness/range-certified-projection-20260914.json`, SHA256
`f5db34aee02363f5bd6432743e49a2cfd5c6b2be7de29092ecd6f076c5e2391b`.

The component-only FP32 K16 carry experiment at `cb0f266` preserves all
tested raw bits, but does not justify product promotion. Native checks cover
2097152 ordered K16 groups, 24 generated projection cases, 42 generated QK
cases and 1255489584 original q7169 QK score comparisons. All six QKV
variants preserve 58728448 external GB10 BF16 cells and 3791742 unrounded
candidate FP32 values, with unchanged PPB 1000 and passing memory guards.

Completed QK time is 341.5882 / 315.7412 / 314.1448 ms for the qualified
scalar-float carry / direct FP32 bits / explicit round-toward-zero variants.
QKV preparation plus replay is 52.8446 / 52.8659 / 52.5744 ms with four
lanes and one staged group; the corresponding sixteen-lane, four-group
variants take 106.928 / 109.552 / 108.020 ms. These single component clocks
include raw QKV capture writes and exclude allocation, uploads and CPU
verification. Keep the new carry outside product dispatch and investigate
current MoE work ownership separately. All local checks and native builds
pass; no new token-loop or release acceptance follows. Command file:
`run-native-f32-carry-r1.ps1 -Kind core|projection|qk|real -Action build|test|capture`.
Evidence: `benchmarks/correctness/f32-carry-components-20260914.json`, SHA256
`c8d57159d6bdb18e55b4d63da6bb03b492b4cfbdeb4a3ada48b1d6d68b14e2cd`.

The current stack passes the original 65536-prefix plus 1024-suffix GB10
boundary on baiying with `D:\models\Qwen3.6-35B-A3B`: all 512 outputs and
actual callbacks match, first token 3709 and exact raw logit 5.9375. All eight
owner chunks complete with first token 16/logit 24.25. State restoration and
changed-prefix rejection pass. Whole `27cfc32`, CK `4a5a5b0` with long direct
PV operands enabled, MoE `9f00db5`, FLA `2ee6215` state 8 and CLI `a797b62`
are the qualified components for this case. Command:
`run-native-long-direct-pv-product-r1.ps1 -LongDirectPvOperands 1`.

Warm callback TTFT is 52683.7323 ms, 6604.4922 ms below the 59288.2245 ms
qualified CK `d705a68` baseline in this single pair. TPOT is 355.022114 ms;
load is 21209.1856 ms, within 30000 ms. The CLI seed phase, including the
initial continuation, is 2576915.3558 ms; native wall is 2832783.961 ms and
host guards pass. Only CK identity and the long direct operand option differ
from the baseline. Long calls retain the original per-group envelope and
add no allocation. Use the enabled option for subsequent long-context
candidates; its source default remains 0.

The unchanged 64k targets are 5432.415542 ms TTFT and 46.658882 ms TPOT.
Performance, saved checkpoints, Windows 128k/256k, current package/HTTP,
one-hour soak and release remain open. Evidence:
`benchmarks/correctness/long-direct-pv-prefix64k-20260914.json`, SHA256
`6c3595aa3ea69bfc37cf553021ba4f0f10496279bbc30fe00207aaf45fbf3cfe`.
The complete preceding 64k baseline remains in
`benchmarks/correctness/prefix64k-current-stack-20260914.json`, SHA256
`a113d05400d2469ecc181c32d4244273af4eddcbfe8c94a886fe91a779cfd3ad`.

The opt-in long-history direct PV provider at `4a5a5b0` passes 42 native
kernel cases and the original 16k+1024 layer-3 component. All 4194304 BF16
cells and same-DLL raw FP32 outputs agree between OFF/ON, including repeat,
same-address refresh and negative checks. Four-call process walls are
2966.147 / 2253.658 ms; these are component diagnostics. Long calls retain
the original per-group error envelope, and the new option defaults to 0.

Its original q8192 control matches all 512 outputs/callbacks and exact first
144/logit 10.375. Callback TTFT is 39203.4717 ms, TPOT 101.690611 ms and load
21282.1297 ms. The long option is inert at q8192; no speed change is inferred
from this single control. The complete real 64k result is recorded above.
Evidence: `benchmarks/correctness/long-direct-pv-controls-20260914.json`,
SHA256 `0d2deb5fba42aa21c2fba9225fd6309b4596324fac402605d9be21c345d147fb`.

CK `d705a68` corrects a long-history application timeout without changing
attention arithmetic. The first 64k attempt completed 24576 original inputs
but stopped in layer 15 of the next chunk: 8128 of 8192 queries had drained
successfully at 20.081793 seconds, exceeding the old flat 20-second budget.
No output or callback was emitted; that run has no inference acceptance.
Each query window now receives a bounded budget based on its query count
and ending key-history extent. q8192 keeps its original 20-second execution
bound, and the long process retains a separate 3600-second hard timeout.
Performance targets are unchanged.

The rebuilt CK passes the original q8192/512 control with whole `27cfc32`,
MoE `9f00db5`, FLA `2ee6215` state 8 and CLI `a797b62`. All 512 outputs and
callbacks match GB10, first 144 and exact logit 10.375. Callback TTFT is
39324.1539 ms, TPOT 102.038706 ms and load 21327.8670 ms. Mathematical kernel
and external CK source hashes are unchanged. All local checks pass, including
injected-clock progress, failure-drain and overflow cases. The real 64k
rerun is recorded above; current package, HTTP, soak and retained-performance gates
remain open. The refreshed Linux release is still `v1.5.1-native-vl.7`, with
unchanged notes. Command: `run-native-attention-history-deadline-control-r1.ps1`.
Evidence: `benchmarks/correctness/attention-history-deadline-20260914.json`,
SHA256 `6ad4a9f91986bb0d3d3cd2fa3ab362dbf3748b2fb66593418451c8dda7818ab3`.

Whole and CK source `27cfc32` extend exact attention capacity from 65536 to
131072, covering a 65536-token prefix, 1024 suffix inputs and the resident
decode tail. The three fixed attention allocations grow by 306184192 bytes
(292 MiB). Model position limits and numerical dispatch parameters stay
unchanged. Local checks pass 371 Python tests with 2 skipped, 47 Rust tests,
C smoke, clippy and hygiene; both native Windows builds pass.

The new q8192 control on baiying with `D:\models\Qwen3.6-35B-A3B` preserves
all 512 original GB10 outputs and callbacks, first 144 and exact logit 10.375.
Callback TTFT is 39498.9559 ms, TPOT 101.624535 ms and model plus engine load
21503.0635 ms. MoE `9f00db5`, FLA `2ee6215` state 8, CLI `a797b62` and all
qualified numerical flags remain common with the 39620.1231 ms reference.
This single capacity control does not establish a speed improvement. The
separate real 65536+1024 prefix result is recorded above; 128k/256k plus suffix need
larger capacity and their own qualification. Performance and release gates
remain open. Command: `run-native-prefix64k-capacity-control-r1.ps1`.
Evidence: `benchmarks/correctness/prefix64k-capacity-q8192-control-20260914.json`,
SHA256 `0981bd909d6495e5a9e305767d7adc8912d289376b4f211537a31b186a1eb160`.

Scalar float GDN state at FLA source `2ee6215` passes same-DLL q8192
OFF/8-column controls on baiying with `D:\models\Qwen3.6-35B-A3B`. Both
match all 512 original GB10 output tokens and actual callbacks, first 144
and exact raw logit 10.375. Callback TTFT is 40531.9227 / 39620.1231 ms,
a 911.7996 ms reduction in this single pair. Enabled TPOT is 100.946664 ms
and model plus engine load is 21304.6899 ms. Use
`QRT_FLA_GDN_SCALAR_FLOAT_STATE=8` in subsequent candidates; source default
remains 0 pending broader qualification. Scalar WU/output 1, tiled KKT 2,
whole/MoE `9f00db5` registered metadata 1 and both prevalidated replay
options 1, CK `6c0c54c` scalar QK 1 and CLI `a797b62` remain common.
Command file: `run-native-fla-scalar-state-product-r1.ps1 -StateColumns 0|8`.

Each state CTA retains complete columns, reuses losslessly packed BF16
operands and row eligibility, and computes the original decay once per
token/chunk. Ordered K16 carry, integer fallback, BF16 residual/checkpoint
boundaries, final FMA and FP32 state layout are unchanged. Both 4-column
and 8-column versions preserve every output and final-state bit in q64,
q65 and the original GB10 q7169 GDN capture, including sync/async parity.
Full `make check` passes 364 Python tests with 2 skipped, 47 Rust tests,
C smoke, clippy and public hygiene. WU/output lane markers now report the
actual scalar lane count 1. State dispatch has its own explicit marker;
checkpoint-capture calls retain their original kernel.

Completed state event diagnostics fall from 1952.21936 to 1009.60123 ms
across 240 segments. These measurements are inside the linear-core wall;
actual callback TTFT remains the product clock. Continue structural work
on the attention and projection walls. q8192 still exceeds 10 seconds and
the immutable 4187.415605 ms target. New long-context, checkpoint, package,
HTTP, soak and release acceptance remain open.
Evidence: `benchmarks/correctness/fla-scalar-state-product-20260913.json`,
SHA256 `758caabf0fbba86e772691902bd052ad2c98f06d1267d36d8fe062593a28d9c1`.

The prevalidated float PV component at source `08326fd` passes all 348
native comparisons (51904512 output positions), including original raw
output/accumulator/denominator equality, CPU metadata checks, candidate
ownership and memory guards. Original, 1-thread and 4-thread routes each
match all 29364224 GB10 q7169 attention cells and all four qualified replay
files, retaining 2198673 candidate cells. Completed query plus V preparation
is 645.9263 / 689.9965 / 638.8247 ms in these single observations. The
1-thread route is slower; the 4-thread difference is only 7.1016 ms and
does not establish a seconds-scale product gain. Keep the product PV
dispatcher unchanged and investigate broader dense projection correction,
which takes 7359.256 ms in the current qualified state-8 product run.

This experiment reuses the consumed score slab for row eligibility,
preserving separate probabilities, scales and candidate storage. It retains
K16 rounding, K32 rescaling and the original fallback. Full local checks
pass; it has no product environment dispatch and no new token-loop,
long-context, package, HTTP, soak or release acceptance. Command file:
`run-native-float-pv-replay-r1.ps1 -FloatPvLanes 0|1|4`.
Evidence: `benchmarks/correctness/prevalidated-float-pv-20260913.json`,
SHA256 `3334d2061034d39818dabf8635ae724c0a999180da95868c7604c0cbd24a27d0`.


The tiled dense projection component at source `6eb2eba` passes all 56
native CPU/GPU comparisons and all 58728448 original GB10 q7169 QKV cells.
Both 64-row and 128-row layouts preserve the 3791742 candidate identities,
raw output bits, CPU eligibility, bitmap ownership and memory guards.
Completed eligibility plus bitmap/replay is 49.4104 / 93.5343 / 142.6540 ms
for original prevalidated4 / 64-row / 128-row, respectively. Each candidate
adds a 7341056-byte bitmap; both are substantially slower in these single
observations and remain outside the product dispatcher. Full local checks
pass 365 Python tests with 2 skipped, 47 Rust tests, C smoke, clippy and
public hygiene. No new product or release qualification is claimed.

The experiment reuses original BF16 K64 slabs across a two-dimensional
candidate tile and falls back to the original dot for dense tiles. Continue
with structural changes to K16 arithmetic and reusable row metadata.
Command file: `run-native-tiled-projection-whole-r1.ps1 -Action reference-test`.
Evidence: `benchmarks/correctness/tiled-dense-projection-20260913.json`,
SHA256 `b596dcc248a3125402b3f6680f36ba89784f489487504729c73e6fde87f8eb50`.


Reusable K16 row bounds at source `0fae245` also preserve all 58728448
GB10 QKV cells and all 3791742 candidate identities. CPU checks cover all
65536 BF16 encodings and 2097152 ordered groups; 20 native comparisons
check 902732 complete carry states, metadata, guards and immutable inputs.
Full local checks pass 367 Python tests with 2 skipped, 47 Rust tests,
C smoke, clippy and public hygiene. The candidate is slower:
52.7489 / 61.7995 ms including preparation and replay. It remains outside
the product dispatcher, with no new token-loop or release acceptance.

A separate untimed accounting pass records 97753728 ineligible fallback
groups out of 485342976. Original input words all satisfy the old range;
229376 QKV weight words do not, including 216676 with biased exponent 4.
Continue investigating exact scalar alignment across this wider range and
fallback control flow. Command file:
`run-native-bounded-projection-whole-r1.ps1 -Action reference-test`.
Evidence: `benchmarks/correctness/float-row-bounds-projection-20260913.json`,
SHA256 `7b65cbe49c208b2c1361058357bd5b59784c0c87cb06ed17ea6b9c5c401c8591`.


The exponent-aligned scalar component at source `d13d36a` supports all
normal BF16 source exponents while preserving original K16 arithmetic. It
passes 2228224 exhaustive aligned-term combinations, 2097152 CPU carry
steps, and 30 native cases with 1354098 complete carry comparisons. Both
component variants preserve all 58728448 GB10 QKV cells, all raw output
bits and all 3791742 candidate identities. The new route uses scalar
arithmetic for every one of the 485342976 replayed groups, including the
original tiny weights, with the same 61444-byte row-flag workspace.

Completed preparation plus replay is 49.3512 / 46.9501 ms for original/new
in one observation. The 2.4011 ms component gain does not establish a
seconds-scale product improvement, so this helper remains outside the
product dispatcher. Full checks pass 369 Python tests with 2 skipped,
47 Rust tests, C smoke, clippy and public hygiene. Continue broader
full-model replay experiments with the original GB10 token/logit boundary.
Command file: `run-native-scaled-projection-whole-r1.ps1 -Action reference-test`.
Evidence: `benchmarks/correctness/scaled-significand-projection-20260914.json`,
SHA256 `ecc9711752b25a52ee4353be7d34549dd68d69219215230f228cfe1a337363a8`.

Four same-DLL q8192/512 admission experiments fail the original GB10
continuation boundary. Combined quarter, dense-only quarter, MoE-only
quarter and combined half PPB settings match 114 / 115 / 50 / 58 of 512
positions; their first mismatches are at zero-based indices 2 / 2 / 1 / 8.
All finish with passing host guards, 512 actual callbacks, first token 144
and first-logit error at most 0.0625. This first-token agreement does not
qualify the failed continuations. Observed callback walls
31976.7090 / 35359.5935 / 35793.3683 / 34521.9848 ms are diagnostic only.

Complete environment comparisons verify only the declared selector PPBs
changed. Keep the qualified dense/MoE PPB 1000 and full-attention OUT PPB
10000, with the original GB10 tolerance 0.125 and all 512 expected tokens.
The product dispatcher and defaults are unchanged. Continue broader exact
arithmetic replacements; these observations do not constrain materially
different designs. Command files: `run-native-replay-admission-product-r1.ps1`,
`run-native-replay-admission-isolation-r1.ps1` and
`run-native-replay-admission-half-r1.ps1`. No new release qualification.
Evidence: `benchmarks/correctness/replay-admission-product-failures-20260914.json`,
SHA256 `01d73e7965f15ffc95d63f8bd0152222cd8b57362fcd05b1312438443541e0aa`.

Complete-row exponent maxima at source `dfd681c` preserve all 58728448
original GB10 QKV cells, raw output bits and 3791742 selected identities.
Forty native cases also preserve 11592 outputs and 1805464 complete K16
carry states, with independent CPU metadata, production/diagnostic parity,
unaligned inputs and memory guards. Full checks pass 371 Python tests with
2 skipped, 47 Rust tests, C smoke, clippy and public hygiene.

The candidate retains the original 61444-byte row metadata allocation and
certifies 181901255 of 485342976 real K16 groups without a product maximum
reduction. Another 205687993 groups retain that reduction and 97753728 use
the original ineligible-row fallback. Completed preparation plus replay is
55.0315 / 56.6621 ms for original/candidate, a 1.6306 ms increase in this
single observation. Keep it outside the product dispatcher and pursue a
broader compute replacement. No token-loop or release qualification follows.
Command file: `run-native-row-max-projection-whole-r1.ps1 -Action reference-test`.
Evidence: `benchmarks/correctness/row-maximum-projection-20260914.json`,
SHA256 `86ab77ef43c721c7d847595a73281227c0d825684323fd83bafac5c0962a4b61`.

The earlier scalar float GDN WU/output matrices at FLA source `186da01` pass same-DLL
q8192 OFF/ON controls on baiying with `D:\models\Qwen3.6-35B-A3B`. Both
match all 512 original GB10 output tokens and actual callbacks, first 144
and exact raw logit 10.375. Callback TTFT is 41738.0364 / 40707.8780 ms,
a 1030.1584 ms reduction in this single pair. Enabled TPOT is 101.156079 ms
and model plus engine load is 21207.7142 ms. Use
`QRT_FLA_GDN_SCALAR_FLOAT_MATRICES=1` in subsequent candidates; source
default remains 0 pending broader qualification. Tiled KKT 2, whole/MoE
`9f00db5` registered metadata 1 and both prevalidated float options 1,
CK `6c0c54c` scalar QK 1 and CLI `a797b62` remain common. Command file:
`run-native-fla-scalar-matrices-product-r1.ps1 -ScalarMatrices 0|1`.

The new kernels reuse lossless packed BF16 rows and eligibility in shared
memory, assigning one thread to each result. They preserve ordered K16
carry, original fallback, BF16 boundaries and output FMA order. A WU CTA
captures all V rows in its eight columns before writing U, retaining U=V
ownership. q64, q65 and original GB10 q7169 comparisons preserve every
output and final-state bit, with synchronous/asynchronous parity. Full
`make check` passes 363 Python tests with 2 skipped, 47 Rust tests, C smoke,
clippy and public hygiene. The enabled product records 240 WU and 240 output
scalar-matrix launches. Its legacy `cooperative_lanes=4` field describes
the outer selected route; separate scalar-matrix markers identify the
actual WU/output dispatch.

Completed existing event diagnostics show WU 827.83775 / 445.71661 ms and
output 1778.16570 / 1131.34616 ms across 240 segments. Its enabled persistent
state took 1967.73891 ms before the state replacement recorded above. These scopes
are inside the linear-core wall and do not replace callback TTFT. q8192
remains above 10 seconds and the immutable 4187.415605 ms target; no new
long-context, checkpoint, package, HTTP, soak or release acceptance follows.
Evidence: `benchmarks/correctness/fla-scalar-matrices-product-20260913.json`,
SHA256 `9cff463f8495e0da995516f12f0812b3101a65a5233eaa08a48cf65377b508b7`.

Tiled scalar KKT at FLA source `62d5fd4` passes complete same-DLL q8192
OFF/ON controls on baiying, `D:\models\Qwen3.6-35B-A3B`. All 512 original
GB10 output tokens and actual callbacks match, with first 144 and exact raw
logit 10.375. Callback TTFT is 42180.7213 / 41931.5687 ms, a 249.1526 ms
reduction in this single pair. Enabled TPOT is 100.911523 ms and load is
21291.7237 ms. Use `QRT_FLA_GDN_TILED_KKT=2` in subsequent candidates;
source default stays 0 pending broader qualification. Whole/MoE `9f00db5`
registered metadata 1 and both prevalidated float options 1, CK `6c0c54c`
scalar QK 1, and CLI `a797b62` remain common. Command file:
`run-native-tiled-kkt-product-r1.ps1 -TiledKkt 0|2`.

Each 8-query by 32-column tile reuses the original rounded beta*K rows and
K columns. One thread owns each output; the ordered K16 carry, fallback,
upper triangle and later gate kernel are preserved. All 18 native controls
match 39845888 raw outputs and 4608 independent CPU dots. Generated q8192
original / tiled integer / tiled float takes 19.1582 / 8.0214 / 5.1945 ms;
only the complete token runs establish product gain. Full FLA q64, q65 and
original GB10 q7169 comparisons preserve every output and final-state bit,
with synchronous/asynchronous parity. Full `make check` passes 362 Python
tests with 2 skipped, 47 Rust tests, C smoke, clippy and public hygiene.

Existing completed stage events report KKT 561.36287 / 171.05978 ms across
240 segments; enabled persistent-state and output stages still take
2014.55534 and 1767.07512 ms. These diagnostics sit inside the linear-core
scope and do not replace callback TTFT. Continue broader GDN matrix work.
q8192 remains above 10 seconds and the immutable 4187.415605 ms target;
long-context, prefix checkpoint, package, HTTP, soak and release remain open.
Evidence: `benchmarks/correctness/tiled-kkt-product-20260913.json`,
SHA256 `e0010a7c206f72178d32c52ebb5569721705c1b17b8762dcba2523c6cca0e9ff`.

Registered immutable MoE weight metadata at whole/MoE source `9f00db5`
passes same-DLL q8192 OFF/ON controls on baiying with the real model
`D:\models\Qwen3.6-35B-A3B`. Both match all 512 original GB10 output tokens
and actual callbacks, first token 144 and exact raw logit 10.375. Callback
TTFT is 43272.7319 / 42224.5708 ms, a 1048.1611 ms reduction in this single
pair. Model plus engine load is 20024.5016 / 21329.8927 ms, both within
30000 ms; enabled TPOT is 101.249185 ms. Use
`QRT_QWEN36_MOE_REGISTERED_WEIGHT_METADATA=1` in subsequent candidates;
source default remains 0 pending broader qualification. Both prevalidated
float replay options remain 1, CK `6c0c54c` scalar QK 1, FLA `8f436db`, and
CLI `a797b62` are common. The command file is
`run-native-registered-metadata-product-r1.ps1 -RegisteredWeightMetadata 0|1`.

Registration runs the original FP64 weight norm and eligibility scan during
loading and owns 240 MiB for 40 layer pairs. The enabled run records one
complete table before request entry, 80 matching metadata copies into the
original scratch, and explicit invalidation after the final decode token.
Invalidation clears every source identity before raw weights are released;
new registration drains old readers before rebuilding. Partial failures
never publish a table and retain ownership for drained cleanup. Full
`make check` passes 361 Python tests with 2 skipped, 47 Rust tests, C smoke,
clippy and public hygiene; sanitized lifecycle checks cover failures,
address reuse and deferred frees. Source `9f00db5` also fixes the legacy
prepared-operands marker to report encoded operands only. No new long-context,
checkpoint, package, HTTP, soak or release qualification follows. q8192 is
still above 10 seconds and the immutable 4187.415605 ms target.
Evidence: `benchmarks/correctness/registered-weight-metadata-product-20260913.json`,
SHA256 `06d156a6156fa29b462eb9a29a3a88daede41bbb10f92b03c67df4cd3c280bb6`.

Source `d2f283f`'s synchronized q8192 profile still matches all 512 GB10
tokens/callbacks and exact first logit. Completed host walls are 6607.706 ms
for all 30 linear input projections, 10841.934 ms for their linear cores
including output projection, 10502.727 ms for all 40 MoE calls, and
14292.800 ms for the full attention pipeline. These scopes are nested and
must not be summed. Some GPU subintervals remain negative, so their sums do
not establish a reliable additive breakdown. Instrumented callback TTFT
45053.0266 ms does not replace the uninstrumented 43324.5935 ms result.
Evidence: `benchmarks/correctness/prevalidated-float-wall-profile-20260913.json`,
SHA256 `46cc00d8851eaf96bcb872cf6521d969db239c3566877a9d594df75767350b15`.

The cooperative scalar-float QK component at source `ae2256d` is correct
but slower. All 171 generated cases match 292498128 raw score slots; each
of the nine variants also matches 418496528 original real q7169 score slots
and 228 independent CPU dots. Guards, input immutability and fallback pass.
The same executable measures original integer QK at 453.4480 ms, current
scalar float at 344.6427 ms, and the fastest new cooperative variant at
491.0003 ms. Keep the product's scalar QK setting; these eight cooperative
layouts remain component diagnostics. The next investigation targets broader
linear-attention matrix work. No new product or release acceptance follows.
Evidence: `benchmarks/correctness/float-subgroup-qk-20260913.json`,
SHA256 `8f93aed84d81cf4c51574f922e72be405d2ce489b622450338f382ef0420d04a`.

Row-prevalidated cooperative float replay at whole/MoE source `d2f283f`
passes complete same-DLL q8192 controls. Baseline / both enabled / dense only
match all 512 original GB10 output tokens and actual callbacks, with first
144 and exact raw logit 10.375. Callback TTFT is
44254.7018 / 43324.5935 / 43506.9730 ms; dense correction host wall is
8104.932 / 7328.709 / 7322.577 ms. Dense only saves 747.7288 ms and adding
MoE saves another 182.3795 ms in this single comparison. Retain
`QRT_QWEN36_HAWKEYE_PREVALIDATED_FLOAT_REPLAY=1` and
`QRT_QWEN36_MOE_PREVALIDATED_FLOAT_REPLAY=1` for subsequent candidate runs;
source defaults remain 0 pending broader qualification. Original per-pair
float replay and prepared MoE stay 0. CK `6c0c54c` scalar QK 1, FLA `8f436db`
and CLI `a797b62` remain common. Both-enabled TPOT is 101.887785 ms and
load is 20043.4378 ms. q8192 remains above 10 seconds and the immutable
4187.415605 ms target; no long-context, package, HTTP, soak or release
qualification follows.

Dense replay keeps only row flags; MoE produces its flags in the original
FP64 norm scan with 2359296 additional bytes. Native checks pass 24 core
cases, two bit-identical norm scans, four routed gate/up/down controls,
four dense correction cases and all 58728448 real captured GB10 QKV BF16
endpoints. Guards and immutable inputs pass. The legacy prepared-operands
marker also follows flags-only storage in this source; the analyzer checks
its exact byte count and requires separate prevalidated-float markers.
Original raw evidence remains unchanged. The test-only follow-up `29fe4cb`
keeps native/build sources identical and passes full `make check`: 47 Rust
tests, 360 Python tests with 2 skipped, C smoke, clippy and public hygiene.
The completed phase profile of this token case is recorded above.
Evidence: `benchmarks/correctness/prevalidated-float-product-20260913.json`,
SHA256 `f9e7b6c1fc38765b9ef1a55984375a283d220755dd3a98936aa428fd795b8cdd`.

The component comparison at source `75d5e56` retains provenance in
`benchmarks/correctness/scalar-projection-20260913.json`: all 12 generated
cases and all captured QKV endpoints match. Original prepared4 / scalar
row-major / scalar transposed / prevalidated cooperative4 preparation plus
replay is 54.6064 / 54.7826 / 304.4780 / 50.3057 ms. Those timings exclude
allocation, upload and candidate collection; only the complete product
controls above establish a q8192 gain.

Cooperative scalar float replay at whole/MoE source `9112d0b` preserves
correctness but is slower. Three real q8192 controls match all 512 GB10 output
tokens and callbacks, with first token 144 and exact raw logit 10.375.
Baseline / both enabled / dense only callback TTFT is
44229.9633 / 45788.2217 / 44865.9197 ms. Dense only adds 635.9564 ms;
adding MoE to dense adds 922.3020 ms. Dense correction host wall is
8097.165 / 8699.474 / 8706.942 ms. Keep both
`QRT_QWEN36_HAWKEYE_FLOAT_REPLAY` and `QRT_QWEN36_MOE_FLOAT_REPLAY`
at 0. All three settings use the same whole/MoE DLLs, CK `6c0c54c` scalar
QK 1, FLA `8f436db`, and CLI `a797b62`; load remains about 20 seconds.
This is one observation per setting, without release qualification.

Native checks also pass all 3563520 intermediate K16 endpoints, generated
dense correction and routed gate/up/down cases, and every one of 58728448
real captured GB10 QKV BF16 cells. Guards, immutable inputs and host checks
pass. The derived analyzer's duplicated prepared-marker assertion was fixed
to allow preparation to be skipped when float replay is active; original
native evidence is preserved. The next investigation targets the amount of
exact replay required by error admission and attention PV, while preserving
the GB10 boundary. Evidence:
`benchmarks/correctness/float-replay-product-20260913.json` and
`benchmarks/correctness/float-subgroup-20260913.json`.

Exact scalar float QK at CK source `6c0c54c` passes complete same-DLL
q8192 OFF/ON validation: all 512 GB10 outputs and callbacks match, with exact
first token 144 and raw logit 10.375. Callback TTFT is 46064.2854/44535.0777 ms,
TPOT 101.936076/101.197382 ms, and load 20034.3533/20021.0454 ms. Only
`QRT_CK_SM121_FLOAT_ALIGNMENT_QK=0|1` differs. The single pair saves
1529.2077 ms without extra workspace; use 1 in subsequent candidate runs.
The source default remains 0 pending wider validation. Whole `4fec3ea`,
FLA-MoE `8f436db`, CLI `a797b62`, selective QK 0 and direct PV 1 remain common.
All host guards pass. q8192 is still above 10 seconds and the immutable
4187.415605 ms target; long-context, package, HTTP, soak and release gates
remain open. The cooperative dense/MoE follow-up and its negative performance
result are recorded above.
Evidence: `benchmarks/correctness/float-alignment-product-20260913.json`.

Scalar FP32 products with canonical integer alignment at source `f15ea53`
pass all 1048576 intermediate K16 endpoints and 65536 dots on gfx1151,
including original fallback. Fourteen generated QK cases match 32182304
score slots and 896 CPU dots; real q7169 Q/K match 418496528 slots and
228 CPU dots. Guards and inputs pass. Query time is 338.3151 ms against
447.4125 ms original, with the same 5.1814 ms key transpose. This is a single
component observation, not retained product performance. The product implementation and complete same-DLL q8192 validation are
recorded above; no release qualification follows yet. Evidence:
`benchmarks/correctness/float-alignment-qk-20260913.json`.

The compact 18-bit core at source `609f468` is exact but remains slower.
Its native primitive enumerates all 262144 core encodings and matches
1048576 independent integer dots and all three raw matrix partials. Fourteen
QK cases match 32182304 score slots and 896 CPU dots; original real q7169
Q/K match 418496528 slots and 228 CPU dots. Guards and inputs pass. The
132-byte row reduces generated replay work, but completed QK plus key
preparation takes 554.0810 ms against original QK plus transpose 446.3313 ms.
Keep this component outside product dispatch. No model load or token run is
claimed. The scalar FP32 follow-up and its component measurements are recorded above.
Evidence: `benchmarks/correctness/wide-integer-qk-20260913.json`.

Nonnegative three-part integer reconstruction is exact but slower. The
primitive source91b069b passes1048576 independent CPU int64 dots with zero
raw partial differences. Prepared QK source31d2b91 then passes28 generated
cases/64364608 score slots and1792 CPU dots; original q7169 Q/K match
836993056 score slots and456 CPU dots. All buffers/immutable inputs pass.
Prepared positive QK plus key encoding takes631.0199ms, versus original
scalar QK plus transpose440.4953ms. Four-IU8 on the same148-byte extended
row layout takes597.7987ms. This component stays outside product dispatch.
No model is loaded and no product tokens or performance are accepted.
The wider-core follow-up and its measured decision are recorded above.
Evidence: `benchmarks/correctness/positive-integer-qk-20260913.json`.

The signed three-BF16 integer-core candidate is rejected at source1738c5f.
Native1048576 cells contain566483 raw partial differences and263108
reconstruction differences; four-IU8 reference partials, guards and inputs
pass. Source dca442c isolates the effect: all196608 core conversions are
exact, but mixed-sign BF16 WMMA, FP16 WMMA and packed BF16 DOT2 each differ
on67345 of262144 cells, with maximum observed error0.015625. Scalar FP32
and the five other tested sign/pattern families are exact. Observed nearest-
integer agreement is diagnostic; no snapping or general error bound is
accepted. The current runtime remains unchanged. Next investigate a
nonnegative third partial with exact integer centering correction.

The probe also repairs a real runner issue at source182492c: omit an empty
Start-Process ArgumentList while preserving quoting and job ownership.
Four Windows argument cases pass, and the unchanged no-argument GPU test
then starts and records its numerical failure correctly. No product timing
is accepted from these component diagnostics. Evidence:
`benchmarks/correctness/integer-matrix-contract-20260913.json`.

Routed MoE prepared replay at source7af2c3c passes both complete q8192
controls: all 512 GB10 tokens/callbacks and exact first144/logit10.375. Only
`QRT_QWEN36_MOE_PREPARED_REPLAY=0|1` changes. Disabled/enabled callback TTFT
is45905.0608/46197.6074ms, TPOT 101.531248/100.843159ms, and load
20036.5836/20084.2981ms. Enabled preparation adds1143209984 bytes and
292.5466ms in this single pair. Keep it disabled; no performance gain or new
release qualification follows. Native fused scans preserve original FP64 norm
bits across663040 encoded cells; four routed cases through1025 tokens preserve
gate/up/down bits, including unsupported-row fallback and compaction windows.
The next investigation targets the repeated exact K16 matrix core. The prior
floating mantissa counterexample remains evidence; native arithmetic must be
verified before any product dispatch. Evidence:
`benchmarks/correctness/moe-prepared-replay-20260913.json`.

Completed workspace and matrix attribution at wholea9f0d0f/5feb7e7 keeps
CKe9673f8 selective QK0/direct PV1, FLA-MoE8f and CLIa797. Both real q8192
runs pass all 512 GB10 outputs/callbacks and exact first144/logit10.375.
The170 dense workspace allocations/frees total47.7403/48.6635ms; reuse of
this measured surface cannot recover seconds. The apparent8001.004ms wait
before linear QKV includes previous queued work. Separate completed intervals
find6361.2797ms predecessor wait and 1590.4194ms for its30 matrix calls.
All110 profiled matrix calls total3925.7338ms, with2.7024ms plan setup.
The extra profile fences can alter stream overlap; these are diagnostic
measurements, with no retained-performance acceptance. Callback TTFT is
46105.0172/46045.1689ms, still far above the immutable target. Next evaluate
reusable routed-MoE replay operands produced during required norm scans,
with the completed negative result recorded above.
Evidence: `benchmarks/correctness/dense-producer-wall-20260913.json`.

Output-margin denominator refinement at sourcea25b1c3 passes all35 generated
cases /6307840 outputs and all29364224 originalq7169 GB10 BF16 cells.
The output margin chooses work only; strict interval admission and complete
row fallback remain authoritative. Replay drops from98.0597% to88.7196%
(364827389 of411213840 scores), but repair/probability still takes2408.1079ms,
against original QK/probability431.5555/87.0005ms. This strict component stays
outside product dispatch. Runtime helper math is unchanged and its previously
failed approximate denominator remains disabled. The next structural check
measures repeated prefill workspace allocation/free before choosing reuse.
No model or product tokens are submitted by this component. Evidence:
`benchmarks/correctness/denominator-output-budget-20260913.json`.

The approximate-denominator product trial at sourcee9673f8 fails the GB10
continuation boundary. On baiying with the real model, the enabled path returns
244 instead of255 at output index1 and matches118/512 positions. First144 and
raw logit10.3125 satisfy the first-token tolerance, but cannot qualify this run.
All512 actual callbacks complete and host checks pass; native exit6 records
the token-contract failure. Keep `QRT_CK_SM121_SELECTIVE_QK_PROBABILITY=0`.
The disabled same-DLL control matches all 512 outputs/callbacks and exact
first144/logit10.375, with callback TTFT46169.357ms. The enabled elapsed time
is diagnostic only. Component checks establish exact probabilities/alpha and
agreement with canonical PV at the same approximate denominator;1726 captured
BF16 context cells differ from GB10. Strict denominator correctness remains
necessary for the next route. Evidence:
`benchmarks/correctness/selective-qk-probability-20260913.json`.

Selective QK source90609e4 now constructs its candidate from original Q/K,
without reading baseline scores. All35 generated cases /6307840 output cells
and all29364224 originalq7169 GB10 BF16 cells pass. Probability and alpha
endpoints agree, all score/denominator intervals hold, and guards pass. The
strict denominator repair nevertheless recomputes403235056 of411213840
scores (98.0597%). Native bounded QK takes99.5748ms and the repair/probability
pipeline2512.3431ms; original QK/probability take431.8602/86.9274ms.
Candidate PV is canonical for every cell in this component. This route remains
outside product dispatch. The subsequent approximate-denominator product
trial is recorded above. Evidence:
`benchmarks/correctness/selective-qk-20260913.json`.

The QK numerical decomposition at sourcee373d47 keeps the product unchanged.
On originalq7169 layer3 Q/K/V, canonical attention matches all29364224 GB10
BF16 cells. Native QK changes45648 outputs. Exact prefix maxima and maximum
anchors reduce this to24345; isolated probability and denominator changes
affect22722 and1853 outputs respectively. All six variants are finite and
all whole-allocation guards and immutable inputs pass. Exact QK constructs
the diagnostic hybrids; these counts do not demonstrate avoided score work
or product performance. The two earlier invalid-domain diagnostic attempts
remain attached as failures. The resulting selective score repair and
denominator/output fallback are recorded above. Evidence:
`benchmarks/correctness/qk-probability-decomposition-20260913.json`.

The subsequent native HIP wait diagnostic finds no millisecond fixed completion
penalty in its one-stream component workload. Source548385b observes flags1 in
the untouched process; default, explicit Spin and explicit Auto have shortest
kernel/event medians around32us. Blocking waits are slightly slower. All60
cases pass final output samples and redzones across1920 completed submissions.
Host intervals include submission. No model is loaded, and this does not
establish product timing or all-stream API equivalence. Runtime flags and
completion semantics remain unchanged. Evidence:
`benchmarks/correctness/hip-completion-wait-20260913.json`.

The current corrected runtime has not recovered the retained performance
targets. The following runs use the actual model on baiying, with startup
excluded from the first streamed callback clock. Each listed case generates
all 512 GB10 tokens and an exact first-token raw logit. Both prefix runs also
verify their fallback transaction and complete owner-state restoration.

| Build / route | Actual prompt shape | Callback TTFT ms | TPOT ms | Load ms |
|---|---|---:|---:|---:|
| whole 730a855 / CK 5816925, bound 10000 | 32768 prefix + 1024 suffix | 25597.6857 | 229.372641 | 20055.0067 |
| whole/CK 9871ef2, chunking enabled | cold 17408, chunks 8192+8192+1024 | 170613.1127 | 162.553605 | 20066.476999 |
| same new binaries, q8192 control | cold 8192, ordinary path | 62478.683599 | 110.904104 | 20046.251001 |
| same new binaries, chunked owner | 16384 prefix + 1024 suffix | 17733.562599 | 161.906225 | 20033.961899 |
| MoE 6278fe0, original 1024-block window | cold 8192, ordinary path | 62231.9179 | 112.049735 | 20085.2361 |
| same binary, 16384-block window | cold 8192, ordinary path | 59348.223499 | 111.265862 | 20039.383501 |
| MoE 25c3693, DPP integer reductions | cold 8192, ordinary path | 58243.9345 | 112.070807 | 20050.7749 |
| whole/CK/FLA 930955a, same DPP MoE | cold 8192, ordinary path | 57617.9907 | 107.136582 | 20077.7027 |
| CK d028182, globally compacted PV replay | cold 8192, ordinary path | 56604.0518 | 106.860717 | 20074.636 |
| MoE 8a7a8dc, staged K64 operands | cold 8192, ordinary path | 55453.937 | 105.957922 | 20116.758099 |
| whole/CK/FLA/MoE 8f436db, compact normalization | cold 8192, ordinary path | 53818.8428 | 102.263044 | 20068.490699 |
| MoE a012e01, exact dot tile certificate (not retained) | cold 8192, ordinary path | 56050.9289 | 101.508477 | 20076.2551 |
| whole 57b3306, original dense dispatch | cold 8192, ordinary path | 53953.8059 | 100.925442 | 20031.092701 |
| same binary, device-count dense replay | cold 8192, ordinary path | 53534.7813 | 100.702816 | 20018.5822 |
| whole 584588a, original matrix producer | cold 8192, ordinary path | 53982.562999 | 100.835152 | 20032.911 |
| same binary, hipBLASLt matrix producer | cold 8192, ordinary path | 52110.692099 | 101.46877 | 20017.0856 |
| CK 42448fe, same binary / 32 queries | cold 8192, ordinary path | 52167.1323 | 101.064893 | 20041.6208 |
| same binary / 128 queries | cold 8192, ordinary path | 50764.6481 | 101.346222 | 20096.24 |
| CK 9a7eaf4, 128 queries / V transpose disabled | cold 8192, ordinary path | 50888.006 | 101.364914 | 20059.2105 |
| same binary / V transpose enabled | cold 8192, ordinary path | 49691.0931 | 101.291451 | 20017.5312 |
| same 9a7eaf4 source, lossless compact exp2 table | cold 8192, ordinary path | 49519.006 | 101.043432 | 20072.6491 |
| whole 8cc97d6, prepared operands disabled | cold 8192, ordinary path | 49840.7445 | 101.842764 | 20028.7697 |
| same binary, prepared operands enabled | cold 8192, ordinary path | 48202.0546 | 102.12639 | 20006.0793 |
| whole 1a7f621, absolute bound disabled / prepared enabled | cold 8192, ordinary path | 48111.7024 | 101.159201 | 20065.8768 |
| whole 2e49049, absolute bound disabled / prepared enabled | cold 8192, ordinary path | 48017.509 | 101.468183 | 20041.4914 |
| whole 2e49049 / CK 6520982, final PV envelope disabled | cold 8192, ordinary path | 48223.2078 | 101.959194 | 20074.458 |
| same binaries, final PV envelope enabled | cold 8192, ordinary path | 47785.5537 | 101.329395 | 20064.6161 |
| whole 4fec3ea / CK 126cdcb, shared PV operand loading | cold 8192, ordinary path | 47770.6281 | 101.6759 | 20008.4729 |
| same binaries, direct PV operand loading | cold 8192, ordinary path | 46055.3727 | 101.615746 | 20023.7468 |

Direct PV operands remove per-K16 shared publication/retirement barriers while
keeping the original BF16 inputs, matrix calls, ordered carries, final error
envelope and exact replay. Both complete product runs match all 512 GB10 outputs
and actual callbacks with exact first144/logit10.375. Only
`QRT_CK_SM121_DIRECT_PV_OPERANDS=0|1` changes in the resolved environment;
whole prepared operands stay enabled with the original layout. This pair
observes1715.2554ms lower callback TTFT. Keep the option enabled for subsequent
qualified-stack experiments, with default0. No retained target is recovered.

Native24 cases /4423680 cells compare four shared/direct and original/final
envelope variants. Original output/carry/denominator bits agree, direct bounds
and candidate sets equal their shared controls, and all17694720 canonical BF16
comparisons pass. Originalq7169 attention matches every29364224 GB10 BF16 cell
and all four qualified replay files in both modes. Completed query plus V
preparation is773.9421/913.7159ms; this component comparison excludes common
upload, allocation, key transpose and safety validation. Neither pair is a
sustained-throughput claim, and other contexts/package/HTTP/soak remain open.
Evidence: `benchmarks/correctness/direct-pv-operands-20260913.json`.

Direct integer QK does not share that gain. Source6142e45 passes28 native
cases /64364608 raw-score comparisons and1792 CPU sampled dots. Both direct
and shared integer schedules match836993056 originalq7169 raw scores and456
CPU dots. With all operand preparation included, direct/shared completion is
687.154/542.3215ms, against original scalar plus key transpose446.0553ms.
Keep this component experiment outside product dispatch. The scalar QK and
integer-core arithmetic are unchanged; no model or token result is submitted.
Evidence: `benchmarks/correctness/direct-integer-qk-20260913.json`.

The optional final PV envelope preserves native matrix arithmetic and enlarges
the existing error envelope. Both q8192 runs above match all 512 GB10 outputs
and actual callbacks with exact first token 144 / raw logit 10.375. The sole
resolved environment change is `QRT_CK_SM121_FINAL_PV_BOUND=0|1`; dense
absolute-product admission remains disabled. This one pair observes a
437.6541 ms callback reduction. It does not establish a sustained speedup or
recover the retained targets, and does not qualify other contexts or release.

On the original q7169 attention capture, enabled/disabled completed query
time plus V preparation is 886.8647 / 927.6509 ms, with all 29,364,224 GB10
BF16 cells matching. Candidates increase from 2,181,900 to 2,198,673.
The first implementation was slower at 1096.6938 / 903.2955 ms because its
volatile metadata generated extra private-memory traffic. Explicit FP32
register operations remove that overhead while preserving all captured output
file hashes. The revised native checks cover 4,210,688 envelope comparisons
and 4,423,680 kernel cells without shrinking bounds or candidate sets.
Keep the route optional and move to changes capable of removing seconds from
dominant attention/dense work. Evidence:
`benchmarks/correctness/final-pv-bound-20260913.json`.

The resident hipBLASLt follow-up also leaves absolute-product admission
disabled in the selected stack. It passes110430 native double-reference
cells,62930951 integration endpoints and all58728448 original QKV cells.
The QKV bound matrix costs52.629ms and total correction117.598ms. However,
the actualq8192/out512 candidate, source2e49049, fails the GB10 continuation:
first mismatch is zero-based index115, expected271 versus actual196;
only130 of512 positions match. Its first144/logit10.3125 passes the0.125
first-logit tolerance, which does not qualify the failed continuation.

The same-DLL bound-disabled control matches all 512 GB10 outputs and actual
callbacks with exact144/logit10.375 at48017.509ms. No candidate performance
result is retained. A conservative absolute-product sum on the tested cells
does not establish that the existing empirical projection-error coefficients
remain valid with tighter admission. The subsequent admission audit localizes
the first BF16 difference to layer0 linear OUT: the absolute-product magnitude
bound is conservative, but its unchanged1000PPB multiplier misses the
producer/canonical difference. Retain the Cauchy configuration; the completed
phase profile is below. Evidence:
`benchmarks/correctness/absolute-product-hipblaslt-20260913.json`.

The refreshed synchronized profile uses this restored2e49049 control and
passes all 512 GB10 outputs/callbacks with exact first144/logit10.375.
Completed host observations are18334.4ms for the full attention pipeline,
10690.656ms for MoE,6890.107ms for linear projections and10894.705ms for
the linear core including its output projection. Coarse buckets and nested
observers are not additive. GPU subphase intervals still include negatives;
their raw values are retained without clamping or a validated additive
attribution. The instrumented49635.3398ms callback is diagnostic and does
not replace the48017.509ms uninstrumented observation. The final PV envelope
above addresses part of this work; attention and dense execution still need
broader changes while retaining their GB10 boundaries. Evidence:
`benchmarks/correctness/prepared-stack-wall-profile-20260913.json`.

The bounded absolute-product experiment has not passed its product request.
Source1a7f621 uses WMMA to compute tighter per-cell admission metadata while
retaining midpoint radii, PPB and exact arithmetic. Twelve native windows
cover110430 cells with no underestimate against independent double sums;
62930951 integration outputs and all58728448 original QKV BF16 cells pass.
On the QKV capture, candidates fall3792033→2908868, but bound generation costs
96.820ms and completed correction wall rises to158.23ms.

On baiying, the actualq8192 candidate stops in layer3 before any token or
callback: a9216-row QKV bound window completes in228.204ms, beyond the100ms
dispatch limit. The process returns5 and releases its allocation pool.
There is no candidate TTFT or product correctness result. The same-DLL
bound-disabled control passes all 512 GB10 outputs and callbacks with exact
first144/logit10.375 at48111.7024ms. Only the bound flag differs. Keep the
bound disabled in the selected stack; the resident backend follow-up above
also fails product correctness. Evidence:
`benchmarks/correctness/absolute-product-admission-20260913.json`.

Dense projection replay now optionally prepares a lossless16-bit operand view
once per correction call. A row containing an excluded BF16 value uses its
original operands and exact dot. K16 order, all admission bounds and endpoint
rounding remain unchanged. The same-DLL q8192 pair differs only in
`QRT_QWEN36_HAWKEYE_PREPARED_OPERANDS=0|1`; both pass all 512 original GB10
tokens and actual callbacks with first144/logit10.375 exactly. The observed
callback reduction is1638.6899ms. Completed dense-correction host wall,
including preparation, falls from9581.693 to8157.930ms; allocation and cleanup
remain included in the actual callback clock. This is one run per setting,
without a statistical repeatability claim.

The native component checks cover6987 raw dots against independent CPU and
original GPU arithmetic,12883520 prepared cells,62930951 integration outputs,
mixed row fallback, immutable inputs and redzones. The originalq7169 QKV
capture also matches all58728448 external BF16 cells. The optional allocation
is bounded by the admitted shape and freed before return. Enable prepared
operands in subsequent experiments with whole8cc97d6 and the existing compact
CK9a7eaf4/FLA-MoE8f436db stack. Default remains disabled; this does not qualify
other contexts, a new package or retained performance. Evidence:
`benchmarks/correctness/prepared-projection-operands-20260913.json`.

A separate QK experiment lets one thread reuse a key operand across two
independent query rows. Fourteen native cases check32182304 raw scores and896
independent CPU dots with no differences. Same-executable originalq7169
attention runs match all29364224 external BF16 cells: query plus V-preparation
host wall is920.561ms for one row and893.9971ms for two. The single-pair
26.5639ms component difference is not a product performance result. Keep the
existing provider and prioritize reducing the amount of exact replay work;
the new shared API defaults to one row and has no provider integration.
See `benchmarks/correctness/paired-query-qk-20260913.json`.

The same-binary window pair changes only
`QRT_QWEN36_MOE_COMPACTION_WINDOW_BLOCKS`. The wider collection and bounded
persistent replay save 2883.694401 ms in this measured pair, using 15 MiB more
scratch; both complete GB10 boundaries pass. This is one paired measurement,
with no statistical repeatability claim. The default remains 1024 blocks;
16384 is retained for further experiments. Its separate synchronized profile
also passes all 512 outputs and callbacks, with 13997.512 ms across 40 MoE host
calls. Raw GPU intervals include invalid negative values and do not support
an additive kernel breakdown. See
`benchmarks/correctness/moe-wide-compaction-20260913.json`.

The DPP experiment changes integer lane transport, preserving the exact
arithmetic and every correction bound. Both new q8192 runs pass all 512 GB10
tokens and callbacks with exact first token/logit. Native masked reductions
check 5,505,276 lane results, and 36,891 BF16 dots match the original scalar
algorithm bitwise. Disassembly confirms eight shuffle exchanges become eight
DPP operations. The full configuration is retained for further experiments;
the observed 1730.232799 ms difference is a comparison of single runs, without
a statistical repeatability claim. Build defaults remain disabled. Evidence:
`benchmarks/correctness/dpp-exact-reductions-20260913.json`.

The compacted PV experiment collects uncertain attention outputs across each
32-query batch, then replays their original ordered K16 arithmetic with four
lanes per output. All 512 GB10 tokens and callbacks pass with exact first
token 144 and logit 10.375. Its 56604.0518 ms callback is 1013.9389 ms below
the preceding DPP run; each configuration has one measurement. The original
q7169 component matches all 29,364,224 external BF16 cells, selecting
2,181,900 cells for correction. Completed host operator time is 1058.14 ms,
versus 1200.62 ms for the ordinary path and 1521.55 ms for the same-build
per-query replay control. Twenty native safety cases preserve raw outputs,
inputs and redzones. This becomes the next experimental baseline through
`QRT_CK_SM121_COMPACT_PV_REPLAY=1`; its default remains disabled. Evidence:
`benchmarks/correctness/attention-compact-pv-20260913.json`.

Staging four K16 operand groups before routed correction preserves the original
16-lane arithmetic and admission rules. The complete q8192/out512 boundary
passes with exact first token/logit at 55453.937 ms callback TTFT, an observed
1150.1148 ms reduction from the compact-PV baseline. All 73,782 native dot
comparisons, including partial tiles, match independent CPU arithmetic.
Emitted code confirms K64 operand loads without scratch spills. One negative
microtiming remains invalid; the product comparison contains one run each.
Use four staged groups in subsequent experiments; the build default remains
one. See `benchmarks/correctness/moe-staged-dot-20260913.json`.

The shared integer normalization now uses one magnitude alignment and a
combined truncating shift, preserving the original internal 26-bit grid.
All 4,194,304 wide CPU reference cases and 73,782 native dots pass. The full
q8192/out512 run matches every GB10 token, callback and exact first logit at
53818.8428 ms callback TTFT, 1635.0942 ms below the staged-MoE baseline.
This single-run comparison supports enabling compact normalization in further
experiments; its build default remains disabled. Broader contexts and release
acceptance remain open. See `benchmarks/correctness/canonical-normalize-20260913.json`.

An exact tile certificate skips repeated normalization only when every K16
intermediate retains the original sign and exponent. Its 86,079 native dots
match independent CPU arithmetic, exercising 161,050 admitted tiles. The full
q8192 run passes all 512 GB10 tokens and callbacks with an exact first logit,
but callback TTFT rises to 56050.9289 ms, an observed increase of 2232.0861 ms
over compact normalization. Keep `CertifiedDotTiles=0` and the 8f436db MoE.
These single runs establish no repeatability or real-model admission density;
one negative native microtiming is invalid. Evidence:
`benchmarks/correctness/certified-dot-20260913.json`.

Dense correction can now collect and round in one pass, then read its candidate
count on the GPU and replay bounded four-million-cell windows. Both same-binary
q8192 runs pass every GB10 token, callback and exact first logit. The observed
419.0246 ms difference accompanies correction host wall of 9512.093 versus
9103.645 ms and eliminates 330 host count reads. Native tests compare 8,874,156
output cells, full candidate windows, short split plans and all redzones.
The larger completed window has a 250 ms deadline, exercised at 101.385 ms for
maximum K and dense selection. This remains opt-in; subsequent producer
experiments use compact normalization with device replay disabled. One paired
measurement does not establish repeatability or qualify immutable performance.
See `benchmarks/correctness/dense-device-replay-20260913.json`.

The existing hipBLASLt BF16-input/F32-output plan now optionally produces the
70 QKV/Z matrices before their unchanged norm bounds and exact correction.
Both same-binary q8192 runs pass all 512 GB10 outputs and callbacks with exact
first token 144/logit 10.375. Callback TTFT changes from 53982.562999 to
52110.692099 ms, an observed 1871.8709 ms reduction in one paired measurement.
The original q7169 capture matches all 58,728,448 corrected BF16 QKV values;
eight native shapes check 49,158 BF16 endpoints and preserve buffers.
Use whole 584588a with `QRT_QWEN36_PREFILL_HIPBLASLT_PRODUCER=1`, qualified
8f436db CK/FLA/MoE and device replay disabled for further experiments. The
shipped producer default remains disabled. This adds no dependency and does
not qualify immutable performance or the pending broader release matrix.
Evidence: `benchmarks/correctness/hipblaslt-producer-20260913.json`.

CK `42448fe` widens the independent query slab from32 to128 while preserving
all QK, online probability and selective PV arithmetic. The existing scratch
already accommodates this q8192 shape, so allocated memory does not grow.
Twenty-nine native cases compare4325376 output cells with no raw-bit mismatch,
including candidate ownership, redzones and unchanged inputs. Both original
q7169 replays match all29364224 external BF16 cells; completed query-dispatch
host wall improves1071.21 to994.928ms, with identical output, accumulator and
denominator files.

The same CK binary at q8192/out512 gives callback52167.1323ms with32 queries
and50764.6481ms with128, a1402.4842ms improvement in this single paired run.
Both pass all 512 original GB10 outputs and actual callbacks, with exact first
token144/logit10.375. Candidate TPOT is101.346222ms and load20096.24ms.
128 queries becomes the next experimental configuration; the shipped default
remains32. No immutable performance or broader release gate is qualified.
Evidence: `benchmarks/correctness/attention-wide-queries-20260913.json`.

An optional completion observer then isolates originalq7169 layer3 attention
with128 queries per slab. Completed host phases are QK437.1211ms,
probability135.1112ms, approximate PV244.2869ms, collection6.5589ms and exact
PV179.7545ms. All57 samples per phase complete; all29364224 external BF16
cells and the prior diagnostic artifact manifests match. The observer adds
synchronizations, so its1006.2578ms outer component wall is diagnostic and does
not replace the q8192 performance result. See
`benchmarks/correctness/attention-completed-phases-20260913.json`.

CK `9a7eaf4` adds an optional 8 MiB transposed V view for compacted exact PV.
The arithmetic and admission bounds remain unchanged; the view refreshes
under the existing workspace lease for every layer/call and is released with
the provider. Fifty-eight native cases compare 8,650,752 output cells with
zero raw-bit differences and preserve inputs, redzones and candidate ownership.
Both original q7169 modes match all 29,364,224 external BF16 cells. Completed
query wall changes from 993.2327 to 965.8722 ms, including 0.1259 ms for the
new transpose but excluding allocation, upload, safety checks and common K
transpose. An instrumented run shows less exact-PV time and more probability
time; the cause of the latter is not established.

Both same-DLL q8192 runs pass all 512 original GB10 outputs and actual
callbacks with exact first token 144/logit 10.375. Callback TTFT changes from
50888.006 to 49691.0931 ms, an observed 1196.9129 ms reduction in one paired
measurement. Candidate TPOT is 101.291451 ms and load is 20017.5312 ms.
Enable `QRT_CK_SM121_COMPACT_PV_TRANSPOSE_VALUE=1` for subsequent experiments
with 128 queries; the shipped default remains disabled. Decode and key
extents beyond 8192 retain the original layout. No repeatability or release
qualification is claimed. Evidence:
`benchmarks/correctness/pv-transposed-view-20260913.json`.

The existing lossless exp2 encoding now passes the full q8192/out512 gate
with that stack: all 512 GB10 tokens and actual callbacks match, with exact
first token 144/logit 10.375. Callback TTFT is 49519.006 ms. The 172.0871 ms
difference from the comparable preceding run does not establish repeatable
acceleration. Retain the format for its exact table storage reduction from
183,174,448 to 38,909,480 bytes. The same decoder was checked over all
328,728,576 covered values and 100,000 domain inputs; the current q7169 replay
again matches all 29,364,224 external BF16 cells. Only CK binary/table paths
change in the resolved product environment. This remains an experimental
build option; package savings, broader contexts and release gates are pending.
See `benchmarks/correctness/compact-exp2-stack-20260913.json`.

Four-lane prepared QK (`f2b21b1`) preserves all 32,160,016 native FP32 scores
and 640 independent CPU dot comparisons, including partial/causal tiles,
raw fallback and input/redzone checks. Both same-executable q7169 modes
match all 29,364,224 external BF16 cells. However, completed query wall with
V preparation increases from 906.5995 to 1090.3751 ms. Keep one lane; this
experiment is not integrated into the provider or selected for product use.
No q8192 result is attributed to it. See
`benchmarks/correctness/subgroup-tiled-qk-20260913.json`.

An instrumented q8192 run with whole584588a and CK/FLA/MoE8f436db, before
wider query slabs and the V transpose, passes the same full GB10 boundary.
Completed host clocks show 21470.5 ms for the attention pipeline,
11298.44 ms for linear core including output projection, 10792.946 ms for MoE
and 7340.268 ms for linear projection. Its 53787.850699 ms callback includes
profiling and does not replace the uninstrumented result.

The profile also corrects earlier MoE timing labels. The prior 16546.228 and
13785.066 ms values came from synchronized GPU events; the corresponding host
clocks are 16764.390 and 13997.512 ms. Original callback comparisons and GB10
acceptance remain valid. Canonical historical records are preserved, with the
field correction and current signed event anomalies documented in
`benchmarks/correctness/hipblaslt-producer-wall-profile-20260913.json`.

Parallelizing probability generation across eight K32 waves preserves the
original sequential denominator recurrence and all 216 native numerical
cases. Both full q7169 comparisons match every external BF16 cell, but completed
host operator time changes only 1069.18 to 1063.79 ms. No speedup is retained
and no full-model run selects this mode. Subsequent experiments retain compact
PV mode 1. See `benchmarks/correctness/attention-parallel-probability-20260913.json`.

Removing the three fixed routed midpoint bands fails 384 of the 512 original
continuation tokens, despite an exact first token and logit. The absolute-only
selector experiment is rejected and its timing is excluded from performance
acceptance. Further experiments retain all three fixed radii at 512, alongside
the original norm-scaled bound. Full failed tokens, callbacks and command
provenance are in `benchmarks/correctness/moe-absolute-selector-20260913.json`.

The chunked route bounds activation carriers but shows no speedup and remains
opt-in. The 10000ppb correction bound is an empirically qualified admission
setting for the declared cases, with unchanged exact-dot arithmetic and
GB10 tolerance. The targets remain q8192 TTFT <= 4187.415605 ms,
TPOT <= 35.502151 ms and model+engine load <= 30000 ms. The new builds still need
broader contexts and renewed package, HTTP and soak qualification. Commands,
component hashes, raw outputs, callback timing and external oracle bindings
are in `benchmarks/correctness/prefix32k-admission-product-20260913.json` and
`benchmarks/correctness/cold-prefill-chunks-20260913.json`.

## Historical published measurements

The following are historical native Windows results for Qwen3.6-35B-A3B BF16 on Ryzen AI
Max+ 395 (`gfx1151`), batch size 1. Startup is measured separately from TTFT.
Every retained product row was accepted only with its matching external BF16
correctness boundary. These recorded profiles and boundaries do not qualify
the current corrected runtime above.

## Product matrix

| Mode | Prompt shape | Total tokens | Load ms | TTFT ms | Prefill tok/s | TPOT ms | Decode tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| cold | q8192 | 8,192 | 19,940.245 | 3,852.909 | 2,126.186 | 32.732 | 30.551 |
| cold | q16384 | 16,384 | 19,948.525 | 8,027.368 | 2,041.018 | 32.457 | 30.810 |
| cold | q32768 | 32,768 | 19,909.139 | 17,274.313 | 1,896.921 | 33.935 | 29.468 |
| cold | q65536 | 65,536 | 19,924.188 | 41,381.599 | 1,583.699 | 39.778 | 25.139 |
| cold control | q130560 | 130,560 | 19,938.847 | 124,594.960 | 1,047.875 | 47.173 | 21.198 |
| cold | q131072 | 131,072 | 20,051.963 | 108,563.050 | 1,207.335 | 42.762 | 23.385 |
| prefix | 16,384 + 1,024 | 17,408 | 19,919.603 | 1,529.093 | 11,384.526 | 36.488 | 27.406 |
| prefix | 32,768 + 1,024 | 33,792 | 19,920.588 | 1,930.620 | 17,503.185 | 38.509 | 25.968 |
| prefix | 65,536 + 1,024 | 66,560 | 19,933.010 | 3,065.768 | 21,710.708 | 36.922 | 27.084 |
| prefix | 129,536 + 1,024 | 130,560 | 20,035.182 | 7,899.540 | 16,527.544 | 42.745 | 23.394 |
| prefix | 131,072 + 1,024 | 132,096 | 20,023.448 | 8,592.158 | 15,374.019 | 42.890 | 23.315 |
| prefix | 262,144 + 1,024 | 263,168 | 20,045.888 | 9,278.300 | 28,363.817 | 55.555 | 18.000 |

The last historical prefix row used a 263,680-token qualification service
limit. The public HTTP profile defaults to a stricter 262,144-token total
context; operators must not infer that the larger row changes the released
default contract.

## q8192 retained target

The confirmed retained target is 1,506.407 prefill tok/s and 4,187.416 ms TTFT.
The accepted row reached 2,126.186 tok/s and 3,852.909 ms. Model plus engine
load remained below the 30-second product bound.

## q8192-neighbor continuity gate

The v1.0.1 repair replaces fixed-q8192-only CK-FMHA and fused-GDN calls with
dynamic q8191/q8193 entries while leaving the retained q8192 entry points
unchanged. The isolated gfx1151 provider smoke produced:

| Provider | q8191 | q8192 fixed | q8193 | Neighbor/fixed ratio | Numerical gate |
|---|---:|---:|---:|---:|---|
| CK-FMHA | 19.888 ms | 19.983 ms | 20.067 ms | 0.995 / 1.004 | no value above `1e-5`; no nonfinite |
| fused GDN, decay | 8.546 ms | 8.773 ms | 8.709 ms | 0.974 / 0.993 | max output error `7.63e-6`; async exact |
| fused GDN, log-g | 8.731 ms | 8.727 ms | 8.703 ms | 1.000 / 0.997 | max output error `7.63e-6`; async exact |

The q8191 CK comparison had 267 bit-level differences at the final compared
token, with maximum absolute error `2.98e-8`; none exceeded the declared
`1e-5` component tolerance. q8193's q8192 prefix was bitwise exact. The packed,
fixed q8192, and q262144 tile-regression checks were also exact. These numbers
remain synthetic component evidence rather than product inference evidence.

The native Windows real-model publication gate passed on `baiying` from a
clean build at source commit `09bd96fd2d85a0715f1501d16fb6391ce199d0f1`.
It used the model at
`D:\models\Qwen3.6-35B-A3B`, the public GB10 oracle, `max_tokens=1` and `2`,
and three cold-prefix repetitions per shape. All 18 returned token sequences
matched GB10 exactly.

| Prompt | max_tokens=1 median TTFT | max_tokens=2 median TTFT | All-six range |
|---|---:|---:|---:|
| q8191 | 3,948.904 ms | 4,027.895 ms | 3,911.132–4,046.155 ms |
| q8192 | 3,878.767 ms | 3,885.607 ms | 3,858.763–3,903.565 ms |
| q8193 | 3,904.950 ms | 3,892.132 ms | 3,846.939–3,905.346 ms |

The worst neighbor/q8192 median ratio is now `1.036619`; the worst positive
residual is `142.288 ms`. Both pass the tightened `1.10x` and `500 ms`
limits, and every q8192 sample beats the retained `4,187.416 ms` TTFT target.
The former q8193 5.5-second fallback is no longer present.

## Wide prompt-length continuity gate

The neighbor check above is supplemented by a 72-request cold-prefill sweep
covering eight independent length boundaries from q4096 through q16384. Each
boundary tests `q-1`, `q`, and `q+1` three times with a unique leading token
to prevent prefix-cache reuse. Every AMD-generated token ID matched the GB10
BF16 authority.

| Center | q-1 median TTFT | q median TTFT | q+1 median TTFT | Local max/min |
|---:|---:|---:|---:|---:|
| 4,096 | 2,265.420 ms | 2,193.201 ms | 2,187.632 ms | 1.035558 |
| 6,144 | 3,310.968 ms | 3,254.327 ms | 3,384.861 ms | 1.040111 |
| 8,192 | 3,956.523 ms | 3,889.966 ms | 3,890.921 ms | 1.017110 |
| 9,216 | 4,614.217 ms | 4,601.238 ms | 4,620.542 ms | 1.004196 |
| 10,240 | 5,134.645 ms | 5,120.694 ms | 5,143.162 ms | 1.004388 |
| 12,288 | 6,037.402 ms | 6,023.924 ms | 6,028.683 ms | 1.002237 |
| 14,336 | 7,504.772 ms | 7,396.619 ms | 7,675.859 ms | 1.037752 |
| 16,384 | 8,456.237 ms | 8,047.645 ms | 8,089.251 ms | 1.050772 |

All eight local gates remain below `1.10x` and `500 ms`; the worst observed
values are `1.050772x` and `408.592 ms`. Median throughput ranges from
`1,807.612` to `2,105.931 tok/s` across all 24 cohorts, a global max/min ratio
of `1.165035`. The split-tail path now overlaps its two independent
transactions; this raises the q6144 neighborhood without changing the retained
direct-q8192 route. The result distinguishes normal shape-efficiency variation
from a length-rounding cliff: no tested exact boundary or adjacent non-boundary
length takes a separate multi-second fallback.

The machine-readable q8192 product record, wide continuity record, provider
smoke, verifiers, and GB10 oracle are under `benchmarks/` and `scripts/`.

## Correctness attachment

At q8192 the external BF16 authority and native runtime selected token 16. The
authority first-token logit was 24.750 and native was 24.875, an absolute
difference of 0.125 (the declared tolerance boundary).

Long continuation checks generated 512/512 matching tokens:

| Case | Input digest | Output digest | Match |
|---|---|---|---:|
| cold q131072 | `ffef18d340fb4fe8` | `4813300b562d057b` | 512 / 512 |
| prefix q131072 + 1024 | `9a681295dde809d4` | `fd69ca7327c912bf` | 512 / 512 |
| prefix q262144 + 1024 | `6aa7d9579bb8427c` | `7a8c5d8ec398878b` | 512 / 512 |

Digests are compact publication identifiers, not standalone correctness
authority. Acceptance was token-for-token against the external BF16 service.

## Interpretation

TTFT excludes model/engine startup but includes real prompt prefill and the
first generated token. Prefix throughput divides the full effective prompt by
the measured reused-prefix request time and should not be compared as cold
compute throughput. Results are specific to the stated model, hardware,
toolchain, runtime profile, and batch size.

Machine-readable rows and hashes are under `benchmarks/performance/`.


The checked-signed K16 map audit at `faf18b2` adds an independent wide-division
and modulo64 crosscheck, including all1048576 boundaries in a deterministic
65536-dot sample of original layer3 q7169 Q/K. It supplies host arithmetic
coverage. The existing native dyadic scan remains slower; no repeated GPU
implementation is adopted. See [the audit](CHECKED_QUANTIZED_MAP.md) and
`benchmarks/correctness/checked-quantized-map-host-20260918.json`.

The complete tiled GDN experiment at `ad6265b` shares operands across two/four
independent ordered dots and moves exceptional full-dot replay outside fast
product lifetimes. Native144 safety configurations and all complete q8192
comparisons pass with the production1024-token segment boundary and both U
ownership modes. Revised1x2 medians are70.3813/72.3855ms versus75.5091/76.3576ms
for control;2x2 is85.6010/83.5985ms. Samples overlap. Keep both isolated because
the limited generated gain does not establish a seconds-scale model change.
All state/intermediate bits, original fallback, guards and CPU checks pass;
this is no new GB10 or product-performance boundary. See
[TILED_SCALAR_GDN.md](TILED_SCALAR_GDN.md) and
`benchmarks/correctness/tiled-scalar-gdn-native-components-20260918.json`,
SHA256`9160a1fb37b209b032b2b80996cf352318b191e7202667ad33438978482d0a9c`.
Current qualified experimental TTFT stays24835.3024ms; package and release
remain unqualified. Investigate a smaller exact attention EXP representation,
with complete-domain verification before any component adoption.
