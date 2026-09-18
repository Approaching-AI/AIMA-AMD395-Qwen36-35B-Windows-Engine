# Exact narrow-domain GDN matrix experiment

This isolated candidate applies the proved signed-zero / BF16 exponent95:159
domain from [exact QK](NARROW_DOMAIN_QK.md) to the complete GDN score, W/U,
recurrent-state and output sequence. Its K64/K128 dots are covered by the
same K256 carry bound. Every original K16 group remains in order, with its
original modulo sum and truncation. No score, probability or state work is
approximated or omitted.

The retained packed-BF16 row flags gain a second bit for the narrow domain.
They are rebuilt from actual operands, including each newly rounded state
checkpoint and each residual. Bitwise intersection admits a dot only when
both complete operands qualify. Admitted dots use a normal FP32 carry with
no per-group exceptional branch. Other dots use the original scalar routine
and its original broad eligibility and integer fallback. This preserves
extended carries outside the proved domain.

The candidate keeps the retained paired-score ownership, shared-arena
lifetimes, U=V alias rules, gates, BF16 conversions and final FMAs. It adds no
device allocation. Runtime dispatch, provider defaults and packages are
unchanged. The qualified q8192 model baseline remains 23353.80795 ms.

The native fixture compares three routes: retained paired scores and shared
arenas; narrow W/U, state and output with original paired scores; and narrow
arithmetic on all four surfaces. Both independent U and production U=V
ownership are checked. Eight boundary shapes and seven data families include
zero signs, unsupported operands, strong decay, sparse groups, nonfinite
values, exact domain endpoints and nearby excluded exponents. Independent
original CPU dots and every score/output/state/checkpoint/intermediate bit
remain the comparison boundary. The prior all-encoding and K16 host proof
is reused unchanged. All 336 native configurations pass, with zero raw
mismatches, intact guards and immutable inputs, and both U ownership modes.

Captured tests use the original GB10 q7169 GDN inputs and outputs. The q8192
extension repeats 1024 rows after 7168 original rows; those original rows
retain their external output/W/U/residual/checkpoint boundary. All extended
rows use original arithmetic checks without a new GB10 capture. One warmup
and three rotated completed-host samples cover the full score/WU/state/output
sequence in the retained 1024-token segments. Allocation, reset, transfers
and independent validation stay outside timing. Native component results do
not establish model TTFT, continuation or release acceptance.

Source `a77c0436402fcefd580526278b7c30661aec548f` completes native build,
generated safety tests and both captures on baiying. The complete sequence
has these medians:

| Captured shape / U ownership | Retained ms | Narrow matrices ms | Narrow matrices and scores ms |
| --- | ---: | ---: | ---: |
| q7169 / separate U | 76.2969 | 74.6125 | 74.4770 |
| q7169 / U=V | 76.7033 | 75.7583 | 74.1055 |
| q8192 / separate U | 84.8017 | 84.0563 | 85.9618 |
| q8192 / U=V | 83.8103 | 86.0711 | 84.9590 |

The production U=V q8192 comparison establishes no gain. Keep both candidates
outside runtime dispatch. Every original score, output, checkpoint,
intermediate and final state comparison passes on every attempt. The q7169
capture also matches all 29364224 original GB10 cells for each output, W, U
and residual surface, 113 checkpoints and 524288 final-state values. The
q8192 extension checks the 7168 original rows and 112 original checkpoints;
it has no external final-state boundary for the extended sequence.

Compiled VGPR counts rise from 78/113/77/70 to 114/148/111/105 for WU,
state, output and paired scores. Shared allocations stay unchanged and all
eight kernels have zero private storage and spills. This static metadata
does not establish occupancy or explain the measured slowdown by itself.

[Complete commands, source and binary hashes, original capture provenance,
samples and validation](../benchmarks/correctness/narrow-domain-gdn-native-components-20260918.json):
599900 bytes, SHA256
`c540d027c460aca202808c77ab160a099097e9f31af992ec75cf272b014d892e`.
The command file is `run-native-narrow-domain-gdn-r1.ps1`; its build, test,
q7169 and q8192 actions complete within their declared native timeouts.
All 333 compiler inputs and all 13 capture files are hash-audited. No model
run follows this rejected component comparison. The current qualified QK
control remains 23353.80795 ms; performance and release goals remain open.
