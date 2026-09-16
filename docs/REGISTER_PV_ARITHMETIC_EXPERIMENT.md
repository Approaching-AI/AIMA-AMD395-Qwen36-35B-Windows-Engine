# Register FP32 rescaling in PV

Source `e669940997303b1361ecc31eee03ba17e0ae6333` integrates register
rescaling as a default-off provider option. Four real q8192/out512 processes
pass the complete GB10 boundary. The two ON runs reduce median TTFT from
27866.6750 to 27328.8799 ms. Retain the enabled option in the next
experimental q8192 control; code and package defaults remain off.
The earlier isolated source `bae27305d014709bd6c80a4049c77bfe2b79de84`
and its component evidence are preserved below.

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

## Integrated provider option

`QRT_CK_SM121_REGISTER_PV_RESCALE=1` selects register rescaling in both the
native producer and compacted canonical replay. It defaults to zero and
accepts only `0` or `1` (unset/empty is also off). An eligible prefill must
end at or before token8192, have more than one query, and use the existing
final PV bound, direct operands, transposed V and non-selective QK owner.
An incompatible eligible configuration is rejected. Decode and calls
extending beyond8192 retain their original dispatch. No workspace, exported
ABI or numerical table changes are needed.

Source `e669940997303b1361ecc31eee03ba17e0ae6333` adds the production
option. The native harness exercises its actual producer and replay launcher
against the isolated copies and original arithmetic. All336 configurations
and33226752 output comparisons pass, with84 distinct cases and672 distinct
CPU full dots. The1048576 multiply-pair comparison also passes.

In the same executable, complete q8192 PV is325.3524ms original,
268.7823ms isolated register rescaling and269.0993ms production rescaling.
All native raw surfaces,29364224 available GB10 cells and the3127598
original candidates match on every attempt. Repeated capture rows still
supply the final1023 tokens; this is not a model-token result.

The full attention capture tool exposes
`QRT_ATTENTION_REPLAY_REGISTER_PV_RESCALE` to exercise the actual query
launcher. Both original q7169 runs match all29364224 GB10 cells and produce
identical saved raw output, accumulator and denominator files. Completed
host times are669.963/631.168ms OFF/ON, including stage observations;
these single diagnostic samples are not product TTFT.

C smoke,54 Rust tests, clippy and485 Python tests pass, with2 existing
skips. Strict option parsing, incompatible owners, decode/long fallback,
the partial final slab, allocation and submission failures are covered.
All native build/run guards pass. The CK DLL is1737728bytes, SHA256
`825fbe9113e76813668b6511083115e2bf052bfc0f929b11291a788aaef78950`.
[Integration evidence](../benchmarks/correctness/register-pv-provider-native-20260917.json)
contains both bounded command files, all seven native runs and captured
file identities:481451bytes, SHA256
`52e9194069f736a8a90b0d16e9fafbfbce2aed31095541d611e8363ea3f1ff65`.
## Real q8192/out512 comparison

Four fresh processes run the real model at `D:\models\Qwen3.6-35B-A3B`
on `baiying`, in OFF/ON/ON/OFF order, using the same CK DLL and source above.
`run-register-pv-product-r1.ps1` pins all component identities and packaged
assets; the only mode difference is `QRT_CK_SM121_REGISTER_PV_RESCALE`.
All host guards finish normally. No profiling flags are enabled.

| Mode / repeat | Load, ms | TTFT, ms | TPOT, ms |
| --- | ---: | ---: | ---: |
| OFF / 1 | 21373.9896 | 27859.8360 | 101.111187 |
| ON / 1 | 21304.4507 | 27228.6387 | 100.860908 |
| ON / 2 | 21281.1225 | 27429.1211 | 101.316945 |
| OFF / 2 | 21334.4995 | 27873.5139 | 100.960863 |

Every run verifies all 8192 original prompt IDs, all 512 GB10 output IDs,
the actual 512 callback IDs and first logit10.375 against GB10's10.375
with tolerance0.125. Both enabled runs execute all ten intended attention
calls; disabled runs emit no activation marker. All160 dense and ten
coarse FA correction counts also match, as diagnostics.

Median TTFT improves537.79505ms /1.9299%; both ON observations are below
both OFF observations. Median load is21354.24455/21292.7866ms OFF/ON,
and TPOT is101.036025/101.0889265ms. This limited two-observation-per-mode
comparison supports the next experimental q8192 control, without a
long-term stability claim. The enabled option introduces no new workspace.

[Product evidence](../benchmarks/correctness/register-pv-rescale-product-20260917.json)
contains all four raw run summaries, original GB10 boundary checks,
runtime environments, DLL fingerprints and command text:758651bytes,
SHA256 `ce5fec00d91caa07e7a8ea0342075cd8804380e6ba138e6a3b4d1f75827cefed`.
Real TTFT remains above10000ms and the retained4187.415605ms target.
No prefix, long-context, packaged HTTP, retained-performance or release
qualification is granted. Code and package defaults stay off while broader
arithmetic and correction costs remain under investigation.
