# Native FP32 K16 tree experiment

The current QK route multiplies BF16 operands exactly, then explicitly aligns
each product to the original width26 integer grid and normalizes each K16
carry. The isolated default-off `QRT_CK_SM121_QK_RZ_TREE=1` experiment removes
that per-product integer alignment. It reduces sixteen exact FP32 products
through a balanced tree under native round-toward-zero, then adds the incoming
carry. K16 traversal and the current four-score tile remain ordered.

This changes internal arithmetic. It is not equivalent to the original
width26 accumulator, and original-score bitwise identity is not claimed.
Only the original GB10 prompt, every captured output token, actual callbacks
and unchanged first-logit tolerance may establish product correctness. Neither
internal self-hashes nor a component drift estimate establish that boundary.

Preparation admits BF16 normal exponents96..158 and signed zeros. Across256
products, this range keeps nonzero FP32 products, intermediate sums and
carries normal and finite; cancellation remains on a grid above underflow.
Every unsupported row retains the complete original dot replay. Preparation
refreshes the existing arena for each call without additional workspace.
The option applies only to cold q8192 with the current independent-QK and
exact probability/PV pipeline. Q1, suffix and other shapes retain their route.

All active waves enter the FP32 rounding scope together, keep their barrier
duties and restore the previous mode before output scaling or original replay.
The native audit checks actual mode values before, during and after execution.
Its host addition oracle uses exact double arithmetic for nearby exponents
and explicit adjacent-FP32 handling when the exponent gap exceeds29. Generated
scores must match that independent specification; original-score differences
remain visible diagnostics.

Native source `0cc0459` passes compilation and the independent arithmetic
checks. The host and GPU each check 1,048,576 scalar additions. The 24 GPU
QK cases check 32,848,208 score slots for masks, guards and finiteness, every
one of 27,300 original fallback scores, 1,536 independent CPU dots, complete
prepared encodings and immutable inputs. All 352,128 audited waves restore
the prior mode. Internal score differences remain recorded diagnostics.
[Native evidence](../benchmarks/correctness/rz-tree-qk-native-20260918.json).

The complete same-DLL q8192/out512 product comparison rejects this arithmetic:

| Mode | Load ms | TTFT ms | TPOT ms | GB10 outputs |
| --- | ---: | ---: | ---: | --- |
| OFF | 21415.6378 | 23890.9927 | 101.003586 | All 512 match |
| ON | 21386.5601 | 23100.9178 | 101.018390 | 451 positions differ |

Both runs produce first token 144 with logit 10.375, but ON differs at output
index 1: expected 255, actual 244. All 512 actual callbacks and host checks
complete. Nine cold attention calls use the new route; ten original single
query replays retain their recorded behavior. The earlier offline observer
mistakenly counted all 19 generic attention markers as cold calls; its fix
preserves both raw runs and changes no numerical gate.
[Full product evidence](../benchmarks/correctness/rz-tree-qk-product-rejection-20260918.json).

Do not retain the ON timing as performance. Keep the option off and the
qualified 23902.4417 ms experimental baseline. Runtime/package defaults,
GB10 tolerances, context goals and all release thresholds remain unchanged.
