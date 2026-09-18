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

Native compilation, arithmetic checks and the original model continuation
are pending. The current accepted experimental control remains23902.4417ms
TTFT with complete GB10 output identity. Runtime/package defaults and all
mission, numerical and release thresholds remain unchanged.
