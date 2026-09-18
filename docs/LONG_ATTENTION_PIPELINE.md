# Exact long-attention pipeline

Provider `65a2139` reduces measured cold 16384-token TTFT from 85460.99605
to 76772.1976 ms in a same-DLL OFF/ON/ON/OFF comparison. All 128 original
GB10 output IDs, prompt identities, first logits and actual callbacks pass.
This is the experimental long-context control. The option remains default
off and no runtime package changes.

## Implementation and ownership

`QRT_CK_SM121_LONG_ATTENTION_PIPELINE=1` selects the separate long launcher
for 2–8192 queries ending beyond position 8192. It requires prepared range
QK, transposed V, direct V operands and compact replay mode 1. Conflicting
matrix, selective or all-cell replay modes are rejected. Cold q8192 and
single-query calls retain their established routes.

The launcher combines range-aware narrow QK, fused probability/native PV,
and register-rescale exact PV. Original ordered K16 carry, softmax recurrence,
per-group error bounds, candidate predicate and exact selected replay remain.
The short-context 512-group final-bound proof is not extended. Unsupported
QK tiles and dots use the original general arithmetic and complete fallback.

Each slab validates query/output ranges and its complete contiguous storage
before submitting work. The provider refreshes range and domain metadata
under its existing mutex. Existing space is reused when sufficient; larger
histories grow an independent checked scratch owner. Failed allocation
preserves prior owners. Completion, failure drains, application deadlines
and release remain explicit. At this 16k shape the original 412092420-byte
scratch is sufficient; domain classification adds 655368 bytes, without an
additional score slab. Larger histories can require additional scratch.

## Native checks

The source inventory verifies 293 files against immutable Git blobs. Six
local tests cover policies, more than 1.5 million layout extents, and actual
provider growth, reuse, failed allocation, submission drains and release.
The original workspace mock compilation failure and its correction remain
in the evidence.

Windows native checks pass 100 arithmetic configurations through 264736
keys, 15 regressions of the short-template default, and 50 contiguous-owner
cases. Owner checks use output offset 3, reject one-cell-short storage before
submission, compare QK/P/scales/errors/count/full output, preserve padding
and guards, and observe all five stages. Eight captured 128-query slabs
exercise that owner against original GB10 16k+1024 layer3 context. These
checks do not qualify every long model context.

## Real-model comparison

All runs use baiying and `D:\models\Qwen3.6-35B-A3B`, with whole `ddacdc9`,
CK `65a2139`, MoE `9235750`, FLA `7b20c90` and CLI `24c4304`. Each cold
16384-token request produces 32 tokens. Only the long option differs between
arms; completed phase profiling is disabled and the two complete 8192-token
chunks are observed. First token is 16 with raw logit 25.625, error 0.

| Order / mode | Load ms | TTFT ms | TPOT ms |
| --- | ---: | ---: | ---: |
| 1 / OFF | 21442.3999 | 85250.4014 | 172.135168 |
| 2 / ON | 21318.8396 | 76672.4167 | 163.191310 |
| 3 / ON | 21320.0297 | 76871.9785 | 165.484994 |
| 4 / OFF | 21397.8657 | 85671.5907 | 150.647258 |

Both ON samples beat both OFF samples. The median TTFT reduction is
8688.79845 ms (10.166975%). Second-chunk elapsed time changes from
57551.7/57783.7 to 48808.7/48959.2 ms. Load remains below 30 seconds.
Decode does not show an improvement: OFF/ON TPOT medians are
161.391213/164.338152 ms.

A separate q8192/out512 run with the long option requested passes all 512
GB10 IDs and callbacks, first logit 10.375/error 0. Load is 21333.3627 ms,
TTFT 23351.8109 ms and TPOT 100.617079 ms. The long route is inactive;
nine full-query and ten auxiliary single-query completion records are
validated separately. The obsolete nine-record-only observer assumption
and its offline correction are preserved, without repeating native work.

The same component stack and environment also pass the original
16384-prefix + 1024-suffix case. All 512 initial retry IDs and 512 timed-hit
IDs match GB10, including first token 3709, raw logit 5.6875/error 0 and
512 actual timed callbacks. Both suffix requests restore prefix state; the
changed-prefix guard rejects without invoking the provider. The cold owner
first token/logit also match. Ten owner and twenty suffix calls select the
long route.

This single functional prefix run loads in 21377.9701 ms, seeds the owner
and initial retry in 164928.7351 ms, and completes the timed hit in
90175.8202 ms. Timed-hit TTFT is 9047.9052 ms and TPOT is 158.64168 ms.
There is no paired prefix speedup claim. The retained prefix ceilings remain
2977.539631 ms TTFT, 37.718887 ms TPOT and 22251.890998 ms total.

The single short run does not replace the qualified q8192 median
23353.80795 ms. The below-10-second gate, retained 4187.415605 ms target,
prefix performance, larger contexts and release gates remain open. No
package, server or soak qualification follows from these CLI tests.

[Native and real-model evidence](../benchmarks/correctness/long-attention-pipeline-product-20260918.json):
1023567 bytes, SHA256
`8176417b89011cc7efc2dd0eb4f9fbb887f98a44ecac5ceacb1a9cf2976f2f08`.
Commands: `run-native-long-attention-provider-r1.ps1`,
`run-long-attention-product-r1.ps1`, and
`run-long-attention-q8192-product-r1.ps1`. The provider DLL is 2021376
bytes, SHA256
`9d1d94c8000d65d52cb17b35524ec7829a6a20f339b806e48e9568753b3f35d8`.

[Original 16k prefix continuation evidence](../benchmarks/correctness/long-attention-pipeline-prefix16k-20260918.json):
441473 bytes, SHA256
`c356e3a11436e6da0990ef038221388c9068dc573f8ad17f737570ce28000e99`.
Command: `run-long-attention-prefix16k-r1.ps1`; native process wall is
276968.093 ms, with a 480-second native deadline.
