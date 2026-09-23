# Optional explicit-layout q8192 operators

This isolated candidate combines the GDN, dense replay, routed replay and
cold-q8192 attention images being compared on original GB10 model operands.
Every runtime option remains disabled by default. Native component, complete
model and product performance qualification are pending.

The arithmetic source and every selected image are bound in
`explicit_q8192_compile.json`. Its 13 dense/routed/attention images total
521376 bytes; eight GDN images total 743344 bytes. They are exactly the files
in the prepared native comparison plans, including the unchanged inverse and
value-transpose images. The two additional tiled PV images are diagnostic
candidates and are not embedded in the runtime.

The source comparison preserves original addressing and ascending K16
arithmetic. CUDA controls cover all 30 q8192 GDN layers, complete q8192
QKV/Z/OUT projections, all 64 attention query slabs and complete routed
first-layer gate-up/down projections. Original reference model tokens and
first logits remain attached to those operand captures. Candidate outputs
were never substituted into the original model. These controls do not by
themselves qualify gfx1151 arithmetic or complete candidate inference.

| Setting | Values | Scope |
| --- | --- | --- |
| `AIMA_PORT_NATIVE_GDN_PREFILL` | `0`, `1` | GDN, cold q8192 |
| `AIMA_PORT_PREFILL_ORDERED_REPLAY` | `0`, `32`, `64`, `128` | Selected dense replay tile |
| `AIMA_PORT_ROUTED_ORDERED_REPLAY` | `0`, `32`, `64`, `128` | Selected expert replay tile; requires `AIMA_PORT_NATIVE_MOE_PREFILL=1` |
| `AIMA_PORT_ORDERED_ATTENTION_PREFILL` | `0`, `1` | Full cold-q8192 attention |

Dense/routed bindings retain existing producers, selectors and queues, and
consume original BF16 operands. Enabling both removes 1283457024 bytes of
operand-preparation storage from the native-MoE route. Dense group-major
weights remain incompatible. Attention retains 113254404 scratch bytes and
193 launches per full layer; terminal prefill keeps the last layer's existing
one-query route. Prefix/history and arbitrary-length attention are outside
this optional provider's scope.

The existing C++ ownership and launch implementations are unchanged. Host
ASan/UBSan checks cover GDN carried-state pointers, 144 dense and 12 routed
binding cases, full attention slabs, invalid spans, allocation failures and
cleanup. Expected dynamic shared-memory sizes now match the actual explicit
images. No numerical GPU execution occurs in these tests.

`tools/compile_linux_core_q8192_explicit.py` rebuilds all 15 q8192 diagnostic
images; `tools/compile_linux_core_gdn_explicit.py` rebuilds GDN. Both require
Triton 3.6.0 and all three device-visibility variables set to `-1`. Offline
rebuilds reproduce executable ELF content and launch metadata; differing
nonloaded debug paths are recorded separately. Runtime image hashes still
refer to the exact files selected for the pending native trials. Runtime
Python, PyTorch or Triton dependencies are not introduced.

Build preparation preserves 327 imported files and 72 existing AOT objects,
with 61 compilation units and 17 generated overlays. Detailed source,
compiler, host-check and preparation identities are recorded in
`explicit_q8192_preparation.json`. Product selection requires actual baiying
q8192/out512 tokens, first logit, callbacks, load time and TTFT; no acceptance
threshold or retained target changes with this candidate.
