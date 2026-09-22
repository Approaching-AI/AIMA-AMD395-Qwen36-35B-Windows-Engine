# Ordered native GDN components

Two experimental Triton kernels preserve the original GB10 arithmetic while
compiling directly for gfx1151. They are not selected by the product or the
Linux-core Windows prototype. This is preparation for replacing the GDN
prefill pipeline; no new Windows execution or TTFT result is claimed.

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

## Remaining integration

Both kernels compile with Triton 3.6.0 for gfx1151, including U builds with
and without FP32 observation. Candidate AMD GPU execution remains pending
behind the active full256k product run. The initial compiled kernels have
register spills; resource changes and their actual timing require measurement.
KKT, W, state, output arithmetic and complete pipeline integration also remain
open. Original full-model outputs, logits and product performance remain the
acceptance boundary for retaining any acceleration.
