# Register FP32 rescaling in PV

Source `bae27305d014709bd6c80a4049c77bfe2b79de84` reduces complete captured
q8192 PV from 323.8687 to 269.5124 ms with identical tested raw arithmetic,
native error bounds and candidate membership. Advance the register-only
variant to a default-off provider option and real-model GB10 comparison.
The current component experiment changes no production dispatch or package.

## Arithmetic and generated code

The original native producer and canonical compacted replay use a volatile
FP32 temporary to round `accumulator * alpha` at every K32 tile. On gfx1151
the compiled barriers store to private memory, wait, then load the value.
Isolated kernel copies replace those barriers with the existing explicit
AMD `v_mul_f32` helper. K16 integer canonical arithmetic, native WMMA,
rescaling order, error recurrence, collection and complete replay remain
unchanged. An otherwise identical copied control checks the copy itself.

A second variant computes the original table-derived reciprocal once per
query/head, then shares that exact value between native production and
replay. It retains the original IEEE division and reciprocal delta table.
Its maximum workspace is 8192 bytes for 128 queries and 16 heads.

| Kernel variant | VGPRs | SGPRs | Private bytes | Static private store/load pairs |
| --- | ---: | ---: | ---: | ---: |
| Original / copied native PV | 102 | 42 | 16 | 24 |
| Register-rescale native PV | 102 | 40 | 12 | 16 |
| Register and row-reciprocal native PV | 102 | 40 | 0 | 0 |
| Original / copied exact replay | 71 | 44 | 12 | 2 |
| Register-rescale exact replay | 71 | 42 | 8 | 1 |
| Register and row-reciprocal exact replay | 71 | 38 | 0 | 0 |
| Separate row-reciprocal preparation | 10 | 10 | 8 | 1 |

These are declarations and static disassembly from the measured executable,
not measured occupancy or dynamic instruction counts. Private accesses use
`flat_store` / `flat_load` with the private aperture; searching only for
`scratch_` mnemonics would miss them.

## Correctness scope

The GPU multiply comparison passes 1048576 pairs, including random finite
carries, signed zero, subnormals, extreme finite values, infinities and NaNs
with finite alpha in [0,1]. Each native process repeats this same check.
Seven causal shapes, six value/score families and both butterfly orders
give 84 generated cases and 252 variant configurations. All 24920064 output
comparisons pass, representing 8306688 distinct generated output cells and
672 independent CPU full PV dots. Short workspace rejection, guards,
unused tails and input immutability also pass.

Checks compare every native pre-replay raw output, accumulator, denominator
and error surface, every complete corrected raw surface, original candidate
membership and the original all-cell replay BF16 outputs. Prepared row
reciprocals match an independent CPU evaluation of the original function.
Unsupported NaN cases preserve original replay behavior.

## Complete captured PV owners

One warmup precedes three rotated samples in the same executable. The
completed host clock includes native production, optional reciprocal
preparation, count initialization, collection and complete exact replay.
Allocation, reset and observation are outside the clock. QK, original
probabilities, preparation and V transpose are reported separately.

| Capture | Original, ms | Copied control, ms | Register rescale, ms | Register and row reciprocal, ms |
| --- | ---: | ---: | ---: | ---: |
| q7169 | 209.0290 | 208.4932 | 173.2841 | 171.5227 |
| q8192 | 323.8687 | 323.1832 | 269.5124 | 268.1614 |

Candidate counts remain 2198673 / 3127598 and replayed K16 groups remain
637019466 / 1083940700 at q7169 / q8192. Every attempt matches all original
raw PV surfaces and all 29364224 available GB10 context cells. Prepared QK
matches independent original tiled QK bitwise. Capture observations add
228 / 256 CPU PV dots. The q8192 extension repeats the first 1023 source
Q/K/V rows and compares all 33554432 outputs with original arithmetic;
it supplies no new model token loop. References are observers only.

## Evidence and next step

All baiying build, native safety and capture guards pass. Commands are in
`run-register-pv-r1.ps1`, with model reference `D:\models\Qwen3.6-35B-A3B`.
[The structured proof](../benchmarks/correctness/register-pv-arithmetic-native-components-20260917.json)
contains source/input fingerprints, full reports, command text, executable
metadata and disassembly analysis: 610967 bytes, SHA256
`fb1cc360db569b216bfb4c6e82b7614f5601841fc0a0627b0a9e039d106755d9`.
C smoke and repository hygiene pass; the full Rust/Python suite was not
rerun for this isolated native addition.

Register rescaling saves 54.3563 ms / 16.7834% in the q8192 PV component.
Shared reciprocals add only 1.3510 ms of improvement there, so the first
provider experiment will use register rescaling alone. Product TTFT,
512-token GB10 continuation, retained performance and release qualification
still require real-model measurements. All mission thresholds remain fixed.

## Provider option under qualification

`QRT_CK_SM121_REGISTER_PV_RESCALE=1` selects register rescaling in both the
native producer and compacted canonical replay. It defaults to zero and
accepts only `0` or `1` (unset/empty is also off). An eligible prefill must
end at or before token8192, have more than one query, and use the existing
final PV bound, direct operands, transposed V and non-selective QK owner.
An incompatible eligible configuration is rejected. Decode and calls
extending beyond8192 retain their original dispatch. No workspace, exported
ABI or numerical table changes are needed.

The native harness additionally exercises the production kernel and replay
launcher against the isolated copies and original arithmetic. The full
attention capture tool exposes `QRT_ATTENTION_REPLAY_REGISTER_PV_RESCALE`
to test the actual query launcher. Integration evidence and real-model
acceptance are pending; this option is not enabled in the package.
