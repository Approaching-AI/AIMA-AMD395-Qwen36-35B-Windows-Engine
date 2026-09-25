# Original reference reproducibility

The September23 GB10 capture reproduces all608 original output token IDs and
all four complete first-token logit vectors. It retains the original model,
prompts, token/logit anchors, compute sources, container image and request
geometry. Source `3fcbdafb10a14e94c15bd94a1f821f180bc95373` uses dependency
`f963ff0c890dd94e0167057b19428e539e097628` on `gb10-4t`, runtime host
`aitopatom-66c4`, model `/mnt/data/models/Qwen3.6-35B-A3B`.

The [qualified reference evidence](../benchmarks/correctness/gb10-prefix256-step189-reference-20260923.json)
records the command file and text, source inputs, cache profile, original
outputs, full first-logit hashes, raw capture hashes and cleanup. Complete
original tensors are retained on GB10; all1,415 compact downloaded files
verify locally. The selected surfaces are original bytes, not derived values.

| Input position | Input token | Next output token | Output logit | Complete surfaces |
| ---: | ---: | ---: | ---: | ---: |
| 263291 | 471 | 4980 | 26.625 | 700 |
| 263356 | 279 | 8240 | 22.875 | 700 |

Each position includes all40 layers and full FP32 states before and after
all30 linear-attention transitions. All700 surfaces shared with the prior
qualified263291 capture are bit-identical. These internal comparisons diagnose
the implementation; the original token/logit boundary remains authoritative.

## Compiler configuration repair

Two preceding attempts return q7169 first token220/logit9.375 instead of
original82/9.25. Their32 outputs, complete first-logit vector and500 common
observed surfaces repeat the earlier rejected step124 capture. All eight
frozen FLA autotune caches remain intact. The earliest previously observed
prefill difference is layer3 post-attention norm; those observers did not
capture the attention internals, so this is not a first-bad-operation claim.

Read-only inspection of stopped failed and qualified containers finds one
changed file among26 common compiler artifacts: an Inductor `best_config`
for a strided256-component head normalization. Its generated Python source
is identical. The original IR shows different FP32 reduction orders:

| Selection | XBLOCK | R0_BLOCK | Warps | Reduction |
| --- | ---: | ---: | ---: | --- |
| Original qualified | 2 | 256 | 4 | Combine i/i+128 first; warp butterfly; four-warp tree |
| Rejected | 8 | 256 | 16 | Individual components; warp butterfly; eight-warp tree |

The repaired command mounts the exact original qualified `best_config` bytes
read-only alongside the eight unchanged FLA caches. Pre/post checks verify
the selected cache and generated kernel source. That run restores all608
original outputs and full first logits; no oracle or model compute source is
changed. Guards and process cleanup pass.
[Rejected attempts, original caches and compiled IR](../benchmarks/correctness/gb10-step189-control-cache-repair-20260923.json).

The evidence establishes the changed selection and reduction order, and the
restored complete boundary with the original configuration. It does not
include a forced alternate-kernel single-operation counterfactual. The new
capture qualifies reference data only; Windows correctness, performance and
release acceptance require their own actual model runs.
