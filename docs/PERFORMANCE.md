# Real-model performance

## Current unreleased measurements, 2026-09-13

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
Both pass all512 original GB10 outputs and actual callbacks, with exact first
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
