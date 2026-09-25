# Ordered native GDN components

An experimental complete Triton GDN pipeline preserves the original GB10
arithmetic while compiling directly for gfx1151. The Linux-core Windows
prototype can select it with `AIMA_PORT_NATIVE_GDN_PREFILL=1`; it remains off
by default. Actual candidate AMD execution and product qualification are
pending. No new Windows execution or TTFT result is claimed.

Both arithmetic variants now also match every GDN layer of the original real
q8192 model request. The current candidate uses unsigned 32-bit accumulation;
the preceding 64-bit implementation remains an independently compared control.

[Complete component evidence](../benchmarks/correctness/native-gdn-ordered-inverse-integer-u-20260923.json)
binds the actual source bytes, original inputs, commands, compilation outputs,
CUDA results, memory guards and process cleanup. The source files are new
relative to the report's explicit controller base revision; their hashes
identify the tested candidate independently of that base.

## Inverse reduction order

The preceding AMD component run differs from the original inverse at six of
131,072 BF16 cells, including two within diagonal blocks. Original AMD assembly
uses four strided row chains: round product j+4 first, then FMA products j,
j+8 and j+12, then add adjacent chains. Host replay of that order reproduces
every bit of the actual native result, including all six errors.

The [ordered inverse](../native/providers/gdn/ordered_inverse.py) explicitly
pairs rows j/j+4 within each eight-row half, combines them in the original
tree, then combines the halves. Every 16-term merge uses ascending explicit
FP32 FMA and carries previous dot results into the next dot. Automatic FP
fusion is disabled; the explicit FMAs remain. The existing C++ arithmetic
helper and new CUDA kernel match the complete original first-chunk inverse.

The same CUDA kernel passes all 14,682,112 cells of the historical complete
q7169 inverse and 15 partial/chunk-boundary prefixes from 1 through 65 tokens.
Guards and input immutability checks pass. Partial cases preserve the same
64-token chunk boundaries and compare the original lower-triangular prefixes.

## U accumulation

The original first-chunk U requires one carried K64 accumulation with the
characterized 26-bit/group16 BF16 arithmetic. Two separate K32 dots, group8
arithmetic and sequential IEEE FMA produce 1, 6 and 15 BF16 differences,
respectively, across the 262,144 original outputs.

The [integer U kernel](../native/providers/gdn/integer_u.py) tiles output rows
and columns, rounds V times beta to BF16, aligns each group of 16 products
and its carry as integers, then performs the characterized truncation and
normalization. It computes integer bit width using a floating estimate plus
an exact integer correction at power-of-two boundaries. It introduces no
token, position, layer or reference selectors.

Its CUDA first-chunk result matches all 262,144 original BF16 U values and
all 262,144 FP32 results from the existing characterized host accumulator.
With FP32 observation disabled, it also matches all 29,364,224 historical
q7169 outputs and the same 15 prefix boundaries. All input and guard checks
pass. CUDA comparisons across both components total 47,425,536 BF16 values.

## Complete recurrent pipeline

The [ordered pipeline](../native/providers/gdn/ordered_pipeline.py) adds KKT,
W, local scores, residuals, state updates and outputs. It reuses the existing
complete model-independent exp2 table and the characterized integer group16
accumulator. W preserves both BF16 rounding boundaries. The state residual
uses the unrounded FP32 difference before applying decay and rounding to BF16.

A cold 64-token test initially passed every observed surface while concealing
a nonzero-state error. The original state update computes its carried K64
dot independently, then applies one explicit FP32 FMA for decayed prior state.
Putting the prior state inside that dot changes 232,042 FP32 cells on the
second chunk, including 21 BF16 cells at the next original checkpoint. The
corrected ordering removes all of those differences.

The complete q7169 chain now passes all 113 chunks, including its one-token
tail. KKT, inverse, W, U, residual output, core and final FP32 state match
147,345,408 original values bitwise. All 59,244,544 incoming BF16 checkpoint
values also match. The candidate carries its own FP32 state between chunks;
reference checkpoints are comparisons only. Guards and input immutability
pass. This scope starts at original normalized Q/K and chunk-local G cumsum;
it is not a whole-model test.

The same full chain also passes with the configuration embedded in the
prototype: four warps and 8-by-8 integer output tiles. U has no register spills;
the inverse drops from 209 spill slots and 820 private bytes to 12 slots and
52 bytes. The other six kernels have no spills. These are compiler resource
observations, not an AMD speedup measurement.

[Complete pipeline evidence](../benchmarks/correctness/native-gdn-ordered-complete-pipeline-20260923.json)
binds both configurations, the cold test and counterfactual, original records,
source bytes, commands, eight embedded images and all cleanup checks.

## Unsigned 32-bit accumulation and all-layer comparison

The integer accumulator separates positive and negative magnitudes. A single
aligned BF16 product is at most `255 * 255 * 2^11 = 133171200`; the FP32 carry
is at most `(2^24 - 1) * 4 = 67108860`. Each sum is therefore bounded by
`16 * 133171200 + 67108860 = 2197848060`, below `2^32`. Subtracting the smaller
unsigned sum from the larger preserves the sign and magnitude without 64-bit
arithmetic. Shifts beyond the 26-bit normalized magnitude produce zero.

The replacement matches 2,097,152 FP32 results from the preceding accumulator,
including signed zero, subnormals, cancellation, exponent boundaries and
maximum mantissas exceeding the signed 32-bit range. It also passes the entire
q7169 continuous-state comparison described above.

A separate GB10 real-model run observes all 30 original GDN calls at q8192.
It copies each call's actual inputs, compares both candidate implementations,
and returns the original output unchanged. Each candidate carries its own
FP32 state through all 128 chunks. Across both variants, 2,013,265,920 BF16
core values and 31,457,280 final FP32 state values match the original bitwise.
The two-warp G cumsum also matches all 7,864,320 original FP32 values. Original
inputs and outputs, candidate guards and the exp2 table remain unchanged.

The same loaded engine preserves all 576 original tokens from the q7169 and
q8192 short controls and the q8192 512-token continuation. First tokens/logits
remain 82/9.25 and 144/10.375. Frozen compiler selections and process cleanup
verify. This qualifies the CUDA operator comparisons; the candidate outputs
were not used by the model and do not establish native Windows inference.

[Unsigned accumulator and all-layer evidence](../benchmarks/correctness/native-gdn-unsigned32-all-layers-20260923.json)
binds those runs to the exact candidate sources and emitted gfx1151 images.
The seven integer-accumulator kernels use 8–13 fewer vector registers and
256 rather than 512 shared bytes each. They have no spills; the unchanged
inverse retains its 12 spill slots. These are compiler observations, with
native execution and performance still pending.

## Windows integration and remaining acceptance

The optional prototype path keeps original Q/K preparation and chunk64 cumsum,
then launches the complete ordered pipeline. Each layer uses 390 AOT launches.
It reuses 5 MiB of idle conversion scratch for two FP32 states and two BF16
chunk buffers. Scores reuse the retired KKT allocation; no additional device
allocation is introduced over the preceding optional route. The last chunk
writes the engine's resident FP32 state directly. The eight embedded images
total 651,736 bytes; Python and Triton remain build dependencies only.

ASan/UBSan host execution checks all 128 chunk bindings and state lifetimes,
eight image hashes, mandatory scratch ABI arguments, two clearing boundaries,
the final resident destination and 118 invalid pipeline bindings. All 327
imported files verify; preparation emits 60 compilation units, 17 overlays
and the same 72 imported images. These checks do not execute GPU arithmetic.

Actual gfx1151 execution remains queued behind the active full256k run.
Windows compilation, original q8192/out512 tokens, logits and callbacks, and
the unchanged product performance/load thresholds remain required.
