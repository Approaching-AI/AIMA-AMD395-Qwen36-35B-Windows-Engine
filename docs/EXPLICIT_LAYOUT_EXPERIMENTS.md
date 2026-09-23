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

A [combined optional runtime](../benchmarks/correctness/explicit-q8192-runtime-preparation-20260923.json)
is now committed at `c85259254b805788dc80680caad37685708c98a8`, on
`codex/explicit-q8192-operators`. It embeds the exact 13 dense/routed/attention
images and eight GDN images selected for the pending native trials. C++ launch
and ownership implementations are unchanged. Three ASan/UBSan host checks
pass, build preparation retains all 327 imports and 72 existing AOT objects,
and an offline rebuild reproduces the executable content of all 15 q8192
diagnostic images. The two tiled PV images remain diagnostic only. All four
runtime settings stay disabled by default; no native build or execution has
occurred for the combined candidate.

The [conditional product entry](../benchmarks/correctness/explicit-q8192-product-entry-20260923.json)
binds selected tile results, native cleanup and image identities before creating
a source manifest. Its 11 host controls reuse a previous actual dense result
and reject changed outputs, duplicate/missing rows, weight/queue changes and
guard failures. Independent unused-tile numerical failures remain recorded.
The actual C++ build inventory now comes directly from the builder; standalone
generator source is bound separately. Use the corrected GDN stage `r2` if
running that isolated candidate. Neither product manifest has been created.

The AMD GDN, dense, attention and routed measurements wait for the current full256k
owner to complete cleanly. Windows model tokens, logits, callbacks, load time
and q8192 TTFT remain required. These experiments do not change release gates.

The [combined all-layer comparison](../benchmarks/correctness/explicit-q8192-all-layers-20260923.json)
now passes all160 dense projections,80 routed projections and ten attention
layers on the original GB10 model. It compares5452595200 dense BF16,8053063680
routed BF16,10737418240 QK FP32 and335544320 attention-context BF16 values,
plus24558 sparse values and15 malformed queue controls. All comparisons are
exact; all576 original token IDs and first logits remain unchanged. Original
operands and outputs also match the prior per-family captures. Six arithmetic
sources are byte-identical to commit `c85259254b805788dc80680caad37685708c98a8`.
The observer leaves model arithmetic unchanged and returns no candidate output
to the model. Frozen compiler caches, host reserve and process cleanup pass.
Native gfx1151 correctness and whole-model performance remain untested for
these selected images. Report SHA256
`a0466be9a3825d6178d0a8e870984aa0e5e7c00a08a354ba11d325dc8f4fb085`.

A [persistent GDN recurrence](../benchmarks/correctness/persistent-gdn-explicit-controls-20260923.json)
now passes complete original q7169 outputs, final FP32 state,113 incoming BF16
state checkpoints and all residual V-new values. It also passes first64 and
a nonzero-seeded65-token tail. Each block owns one head and eight value rows
across all chunks, with6144 explicit shared bytes and barriers between phases.
The proposed q8192 recurrence needs one launch instead of384 per layer. Both
gfx1151 images compile without spills; native speed and correctness remain
unmeasured. An optional runtime binding is now prepared below. The original K16 arithmetic and state
fma are unchanged; two preceding CPU compiler failures remain recorded.

The [persistent all-layer comparison](../benchmarks/correctness/persistent-gdn-explicit-all-layers-20260923.json)
also passes all30 original q8192 GDN layers:1006632960 BF16 core values and
15728640 FP32 final-state values. All576 original output IDs and full first
logits are preserved, and the operands/output hashes match the earlier
qualified all-layer capture. The actual worker passes six ABI controls for
both compiled images. A15-case native comparison interleaves unfused control,
fused runtime and fused capture across first64 and complete q7169, reversing
order on the second pass. No native execution or runtime selection is claimed.

The [optional persistent runtime and product entry](../benchmarks/correctness/persistent-gdn-runtime-preparation-20260923.json)
are prepared at `01d1418b28f55339f28c29b6b47821c29626c984`, on
`codex/persistent-gdn-prefill`. Five original64 upstream images feed the fused
recurrence; six selected images add no device allocation or runtime dependency.
The cold-q8192 AOT count is7 per layer including cumsum, previously390.
Actual host ABI/failure checks and offline executable-content reproduction pass.
The entry requires all15 native controls, exact selected images, clean owner
completion, same-run activation markers and the original512-token product gate.
Optional dense/routed tiles and attention still require their own native results.
Nothing has been staged on AMD for this candidate, and no speedup is accepted.

The [compact persistent variant](../benchmarks/correctness/persistent-gdn-compact-controls-20260923.json)
reduces static expansion by tiling shared carriers into16-row blocks and
looping over K16 groups. Its runtime instruction section is65408 bytes,
previously615296, and register use is158 VGPRs, previously225. All original
q7169/core/state/checkpoint and seeded65 controls still pass. There are no
spills or layout conversions;6144 shared bytes are unchanged. The committed
runtime stays intact while full-layer and actual AMD comparisons qualify this
separate source. Instruction size and CUDA timing are not product speed claims.
