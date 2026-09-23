# Explicit-layout arithmetic experiments

The optional GDN runtime is prepared at `b07bf58e5140ea6e256c477f4aacfd39fc89695d`
on `codex/explicit-gdn-layout`. It uses seven Gluon kernels and the existing
ordered inverse, while preserving the characterized unsigned32 arithmetic.
The option remains disabled by default. No new AMD correctness or performance
result is claimed.

The original unsigned32 GDN images failed repeated gfx1151 comparisons on
identical inputs. The unsigned64 control passes q8192/out512 but takes
40680.6294 ms TTFT, above the qualified prototype's 20086.8502 ms observation
and the immutable target. Explicit layouts remove the seven kernels' LDS
layout conversions; higher register use still needs native measurement.

- [Arithmetic and continuous q7169 controls](../benchmarks/correctness/ordered-gdn-explicit-layout-controls-20260923.json):
  147345408 original surface values and 59244544 incoming BF16 checkpoint
  values match, with the candidate carrying its own FP32 state.
- [Runtime preparation](../benchmarks/correctness/ordered-gdn-explicit-runtime-preparation-20260923.json):
  eight exact trial images, reproducible executable content, guarded host ABI,
  128 chunk bindings, 327 unchanged imports and no added runtime dependency.
- [Fresh original-model all-layer comparison](../benchmarks/correctness/ordered-gdn-explicit-all-layers-20260923.json):
  the explicit candidate matches 1006632960 BF16 core values and 15728640 FP32
  final-state values across all 30 q8192 linear layers. The original model
  preserves all 576 control/output token IDs. Candidate outputs never enter
  model computation.

A separate [dense replay experiment](../benchmarks/correctness/ordered-dense-explicit-layout-controls-20260923.json)
keeps selected-pair addressing and ascending K16 arithmetic while assigning
pair/product lanes explicitly. CUDA checks pass 4362240 partial/queue/layout
values and 352321536 complete q8192 QKV/Z/OUT values. All three gfx1151 tiles
compile without LDS or spills. The unchanged native baseline/harness is
prepared for 54 comparisons; no runtime integration or speedup is accepted.

The [attention layout experiment](../benchmarks/correctness/ordered-attention-explicit-layout-controls-20260923.json)
also passes every query slab of the original q8192 layer. QK checks cover
1073741824 stored FP32 values. Selected PV and two complete-output PV tiles
match 100663296 BF16 context values in total. Probability, scale, empty/partial
queue and input/guard checks pass. All five gfx1151 images have no layout
conversion or spills; selected PV retains 16 shared bytes for its queue-range
reduction, while the other four have no shared-memory instructions. The native
plan compares eight full passes of old and explicit images with reversed order.

The [routed-expert layout experiment](../benchmarks/correctness/ordered-routed-explicit-layout-controls-20260923.json)
preserves the gate-up/down addressing and arithmetic. Three tile sizes match
603979776 full-output BF16 values and 24558 sparse-output values; all 15 invalid
queue controls pass. Each expected full tensor is first qualified against the
original model's complete output hash. The six gfx1151 images have no layout
conversions or spills, with 16 shared bytes for invalid-queue flag reduction.
The prepared native trial pairs 24 full comparisons with 17 queue controls.

The AMD GDN, dense, attention and routed measurements wait for the current full256k
owner to complete cleanly. Windows model tokens, logits, callbacks, load time
and q8192 TTFT remain required. These experiments do not change release gates.
