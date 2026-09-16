# Coarse linear-attention OUT experiment

The retained coarse OUT owner covers ten full-attention layers. A completed
profile attributes about 2.92 seconds to OUT in the thirty linear-attention
layers. Those projections have the same q8192, 2048-row, K4096 geometry, so this
experiment extends the existing owner to that larger call surface.

`QRT_QWEN36_COARSE_LINEAR_OUT_PRODUCER` is a strict, default-off `0`/`1` option.
It applies only when linear OUT already requests correction with radius 512
and PPB 1000, and residual/post-normalization consumes the BF16 endpoint.
Host diagnostic materialization, other token counts and streamed linear
workspace paths retain their original dispatch. An 8192-token seed inside a
longer request can use this route; a q8192 test does not qualify that request.
Conflicting original OUT sweeps, terminal diagnostics, shadow audits and
residual/interval filters are rejected when this route would execute.

The owner retains its 295,739,396-byte descriptor-owned arena, C64 matrix
producer, conditional native error coefficient 2^-19, complete compaction,
bounded original K16 replay and BF16 RNE endpoint. There is no PPB reduction
or changed external numerical tolerance. An optional F32 carrier is populated
from the actual BF16 output for later tracing; it is explicitly a rounded
carrier, not a record of the unrounded original matrix result. Both output
conversions are drained before the arena returns to its owner, including
failure paths. Existing full-attention calls keep their original interface
and do not request this extra conversion.

This is a numerical and performance experiment, not an accepted runtime
profile. The empirical native matrix error premise is not a universal
hardware proof. Qualification requires a same-DLL control and experiment on
baiying using the real model, original 8192 prompt IDs, all 512 GB10 output
tokens, the unchanged first-logit tolerance of 0.125 and actual streaming
callbacks. The original TTFT and load targets remain in force. Nothing is
published or enabled by this source change.

## Real-model result, 2026-09-16

Source `768d0544bbab73b4b2ca874afb800d1aa043b4c0` passes local C/ABI,
54 Rust and 470 Python checks (two skipped), clippy and public hygiene in
220.649260 seconds. The sanitized owner covers 17 successful paths and 342
injected failures, including the rounded F32 carrier. Native whole-provider
compilation completes in 92715.828 ms on baiying with all host guards.

Both runs use `D:\models\Qwen3.6-35B-A3B`, the same new whole DLL, retained
Dense1000/MoE512, expert-order MoE, coarse full-attention OUT and unchanged
CK/FLA/CLI. DLL SHA256 is
`05556424b4a55a9b71feaaf0ac188216abfac4f12136b0b1f67eba892a6456e1`.
The command file is `run-native-coarse-linear-out-product-r1.ps1`, SHA256
`07c5d606b963bfcdd05cea7a58741da7b8b81502a0b8aef8d02b58b289bd2a9b`.

| New linear OUT option | GB10 output IDs | First logit | Load ms | TTFT ms | TPOT ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | 512/512 | 10.375 | 21483.5723 | 27891.0528 | 101.818945 |
| 1 | 512/512 | 10.375 | 21278.7260 | 28559.2248 | 100.569905 |

Original prompt IDs, every actual streaming callback, callback ordering and
both first-logit boundaries pass. The enabled route executes exactly the
thirty linear layers, selects 88762686 cells and takes 3331.0106 ms across its
complete owner calls. The control selects 58489767 linear OUT cells;
its 1171.125 ms correction clock excludes its matrix producer and norm
bounds and cannot be compared directly with the complete owner clock.

The enabled product TTFT is 668.172 ms slower in this pair. Keep the option
closed in the retained configuration; there is no performance reason to
expand this experiment's qualification. A correct full-attention optimization
did not transfer profitably to the linear projection's smaller selector.
Both loads meet 30 seconds; both TTFT values remain above 10 seconds. No
retained-performance, prefix, long-context, package or release gate is met.

Evidence: [coarse-linear-out-product-20260916.json](../benchmarks/correctness/coarse-linear-out-product-20260916.json),
372432 bytes, SHA256
`4ca379c5d760aef9ea43db92d25a185aedbbfa34c2e2b0563c6d11a8083c4b41`.
