# Optional original-order q8192 operators

This candidate replaces the arithmetic used by selected dense and routed
projection replay and adds an optional cold-q8192 attention provider. All
three settings default to disabled. The ordinary product route is unchanged.

The reference comparisons observe the original GB10 BF16 model and run the
candidate operators on copied actual inputs. The model receives its original
outputs throughout. Across the independent captures, all 576 original control
and continuation tokens remain unchanged. The complete comparisons cover:

| Operator | Original calls | Complete values matched bitwise |
| --- | ---: | ---: |
| Dense projections | 160 | 5,452,595,200 BF16 |
| Full attention | 10 | 10,737,418,240 QK FP32 and 335,544,320 context BF16 |
| Routed expert projections | 80 | 8,053,063,680 BF16 |

`ordered_q8192_preparation.json` identifies the exact source files, original
model reports, composition, build preparation and reused host results. Each
host result is reused only after every recorded source hash matches this
candidate. No AMD arithmetic, native Windows compilation, product performance
or release acceptance is established by these checks.

`AIMA_PORT_PREFILL_ORDERED_REPLAY=32|64|128` selects the dense tile size and
`AIMA_PORT_ROUTED_ORDERED_REPLAY=32|64|128` selects the routed tile size. The
routed setting requires `AIMA_PORT_NATIVE_MOE_PREFILL=1`. Both projection
bindings preserve the existing producer, selector and queues, and read raw
BF16 operands without preparing the previous half representation. Enabling
both removes 1,283,457,024 bytes of preparation storage from the native-MoE
route. The existing batch-replay setting remains independent; group-major
weights are incompatible with the dense ordered binding.

`AIMA_PORT_ORDERED_ATTENTION_PREFILL=1` enables cold-q8192 attention. It owns
113,254,404 scratch bytes, borrows the existing verified exp2 and reciprocal
tables, and submits 193 AOT launches per full layer. It preserves the original
32-key softmax recurrence. With terminal prefill enabled, the final layer
retains its existing one-query route and the other nine full-attention layers
use this provider. Prefix/history and arbitrary-length attention are outside
this provider's supported scope.

Runtime images are embedded and hash checked before use. Triton and PyTorch
are used by the separate reference and compilation tooling; they are not
runtime dependencies. Exact arithmetic sources are preserved under
`native/providers/ordered_q8192`.

The next required evidence is bounded gfx1151 execution of the prepared
original-input components, followed by a Windows product run with the original
q8192/out512 GB10 token and first-logit gate. Product performance and release
selection require their own complete acceptance evidence.
