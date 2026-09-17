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

Unused contexts are explicitly zeroed. With output liveness disabled, gate, OUT, residual normalization and
MoE still run their current full-shape row-local implementations. Their earlier
hidden rows are no longer model results and have no consumer in this scope.
The suffix ABI stages immutable K/V and one query in its existing83886080-byte
workspace; this experiment adds that allocation to the cold full-prefix route.
No dependency or package default changes. Native compilation and full GB10
continuation checks are required before any performance retention. The broader
10-second, retained-performance, context and release goals remain open.

Source2899246 passes native HIP compilation and four fresh same-DLL
OFF/ON/ON/OFF model runs. All2048 original GB10 IDs, first logits10.375,
actual callbacks and host/owner checks pass. OFF TTFTs are24644.0982 and
24594.1967ms; ON TTFTs are24316.0438 and24258.9025ms. Medians improve
24619.14745 to24287.47315ms (331.6743ms,1.3472%). Both ON observations are
below both OFF observations. All loads remain below30seconds; no decode gain
is established. ON became the preceding experimental control, still default off.
[Complete product evidence](../benchmarks/correctness/final-query-liveness-product-20260918.json)
pins the new13453312-byte whole DLL, SHA256
`f967ffd755869e6a2a29ddffa76541eefb3dd53855281a3a5f51727629446d0b`.

Zeroing dead contexts makes the original coarse OUT interval selector replay
almost every zero endpoint:16775587 candidates versus3034571 in OFF.
The first pair's completed last-layer OUT rises108.9004 to297.0241ms.
`QRT_QWEN36_FINAL_QUERY_OUTPUT_LIVENESS=1` is a subsequent default-off
experiment, active only inside the existing liveness scope. It reuses the
original Q1 K16/width26 output kernel for the last2048 values and zeroes unused
OUT rows directly. Residual normalization, complete KV capture and MoE remain
unchanged. It adds no workspace.

Source `ddacdc9` passes native HIP compilation and a separate same-DLL
OFF/ON/ON/OFF q8192/out512 comparison with query liveness held at 1. All
2048 original GB10 IDs, prompts, actual callbacks and first logits 10.375
pass; every owner and host check passes. OFF TTFTs are 24160.1204 and
24160.9340 ms; ON TTFTs are 23890.3918 and 23914.4916 ms. The median improves
24160.5272 to 23902.4417 ms (258.0855 ms, 1.0682%). Both ON samples are
below both OFF samples. Load median is 21281.16445 ms, with no established
decode gain. Both liveness flags are now enabled in the experimental control;
code and package defaults remain unchanged.

The output marker records one 2048-feature row at K4096, original K16 and
BF16 endpoint, zeroed dead rows and unchanged residual normalization. The
last coarse OUT replay is absent; the other nine coarse, 130 dense and 30
adaptive-linear counts are unchanged. Complete KV remains available to decode.
Earlier hidden rows are not model results and remain outside the declared
single-row consumer scope. Other prompts, prefix/checkpoint/chunked/MTP
consumers, long contexts and total peak memory are not qualified by this case.

[Native build](../benchmarks/correctness/final-query-output-native-build-20260918.json)
and [complete product evidence](../benchmarks/correctness/final-query-output-product-20260918.json)
pin `run-final-query-output-product-r1.ps1`, baiying, the real model and the
13454848-byte whole DLL, SHA256
`2589bdd1d4b35904f5dd040721bb73a09c88bb483af7c3c93d729aab92b08dde`.
This remains above the 10-second gate; all retained and release goals remain open.
