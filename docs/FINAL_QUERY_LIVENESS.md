# Final-layer query liveness

The current q8192 next-token plan requests row8191, but the validated final
full-prefix provider computes all8192 attention queries. Its completed profile
measures894.791ms of layer39 attention. Complete layer39 K/V remains necessary
for decode. Removing query work must preserve those cache inputs.

The existing `QRT_QWEN36_FINAL_LAYER_FULL_PREFIX=0` route fails the original
GB10 q8192/out512 boundary on the current stack: first token144 and logit10.375
match, but462 outputs differ, starting at output1 (expected255, actual244).
All actual callbacks and host guards pass. Its23890.0486ms TTFT is rejected.
That route also changes projection, normalization, gating and MoE; the run
does not isolate the first incorrect intermediate. Later Q1 decode arithmetic
repairs do not establish correctness for this separate terminal-prefill route.

[Complete failed run](../benchmarks/correctness/final-query-existing-route-20260918.json):
374560 bytes, SHA256
`959f79691a2dc8b1ad4e4606ed38b1f61c8c66f0225704e9389648606b4ddd76`.
Command `run-final-query-probe-r1.ps1` pins baiying, the real model at
`D:\models\Qwen3.6-35B-A3B`, whole6e4908b, CKdf2ea51, MoE9235750,
FLA7b20c90 and CLI24c4304. The diagnostic's offline observer initially
shadowed the golden list with a marker dictionary. Its correction preserves
the original failed process and every raw byte without repeating GPU work.

`QRT_QWEN36_FINAL_QUERY_LIVENESS=1` is a new, default-off experiment. It keeps
`FINAL_LAYER_FULL_PREFIX=1`, complete QKV production and resident KV capture,
and calls the existing exact BF16 suffix ABI for only query8191. Its scope
requires an explicit single-last-row consumer, layer39, cold q8192 and the
compact BF16 owner. Multi-row outputs, prefix continuation, checkpoint capture,
chunked prefill and MTP retain the current route. An incompatible active owner
fails instead of silently changing arithmetic.

Unused contexts are explicitly zeroed. Gate, OUT, residual normalization and
MoE still run their current full-shape row-local implementations. Their earlier
hidden rows are no longer model results and have no consumer in this scope.
The suffix ABI stages immutable K/V and one query in its existing83886080-byte
workspace; this experiment adds that allocation to the cold full-prefix route.
No dependency or package default changes. Native compilation and full GB10
continuation checks are required before any performance retention. The broader
10-second, retained-performance, context and release goals remain open.
