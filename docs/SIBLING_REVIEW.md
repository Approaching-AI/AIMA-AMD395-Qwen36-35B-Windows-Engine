# Linux sibling fixes reviewed for the next Windows release

Review date: 2026-09-11. The Windows upstream's open issue remains
[#1](https://github.com/skyguan92/AIMA-AMD395-Qwen36-35B-Windows-Engine/issues/1):
the 8191/8193 TTFT cliff. The fork has issues disabled. No issue is closed and
no candidate is published on the strength of this comparison.

The Linux sibling's recent releases (v1.4.1, v1.5.0, v1.5.1 and
v1.5.1-native-vl.4/.5/.6/.7) were reviewed along with its issues #1, #5, #6,
#7 and #12. The relevant release sources are
[v1.5.1](https://github.com/skyguan92/AIMA-AMD395-Qwen36-35B-Linux-Engine/releases/tag/v1.5.1),
[.5](https://github.com/skyguan92/AIMA-AMD395-Qwen36-35B-Linux-Engine/releases/tag/v1.5.1-native-vl.5),
[.6](https://github.com/skyguan92/AIMA-AMD395-Qwen36-35B-Linux-Engine/releases/tag/v1.5.1-native-vl.6),
and [.7](https://github.com/skyguan92/AIMA-AMD395-Qwen36-35B-Linux-Engine/releases/tag/v1.5.1-native-vl.7).

| Area | Windows action and remaining boundary |
|---|---|
| Variable cold lengths / serial tails (Linux #1/#5) | Preserve acceleration and logical token counts; validate both sides of every optimized boundary with output 1 and ordinary multi-token requests. The frozen q7169 32-token numerical gate now passes with a slow compatibility profile; neighboring-length performance and the unified configuration remain open. |
| Logical GEMM/FLA extents | Audit logical sequence lengths separately from padded allocation/launch extents. Linux `3d284a3` and `e402b3a` are useful arithmetic references, not portable Windows binaries or authority to loosen token/logit gates. |
| Thinking (Linux #6, .5) | Add the validated top-level object, legacy-alias conflict handling, tokenizer parity and live no-tools reasoning streaming. Explicitly preserve the Windows disabled default and repair empty-kwargs inconsistency. |
| Repeated tools (Linux #7, .5) | Canonical deduplication, declared-function admission, one-call mode and explicit conservative no-progress metadata now have CPU/HTTP regressions. Semantic retry strategy and side-effect authorization remain with the caller. |
| Control-plane responsiveness (.5) | Existing bounded FIFO retained; a gated backend test proves health/shutdown and the first reasoning delta arrive before generation completes. Native packaged-server validation is still required. |
| Default VL reasoning (.6) | Windows remains text-only; no vision implementation or inherited VL result is claimed. |
| Partial shared prefixes (.7 / closed Linux #12) | Port safe saved checkpoints spanning KV, recurrent, convolution and hidden state. Windows currently can recompute a matched seed; it must demonstrate actual restoration, suffix-only work, restored bytes/time, and divergent/unrelated-prefix isolation. |

The GitHub latest-release endpoint was rechecked on September 11 and still
selects `.7`; no newer stable release is inferred from local branch commits.

The `.7` release was published on September 9 at 13:36:02 UTC, immutable tag
`9bd8a0fabcf2fc6ef1b882c10b04390c0e31fb00`, native source
`edb584ee16f0ee1fc5f902459598b9446d96387e`. Its issue #12 is closed. Source
inspection confirms up to three eligible checkpoints per LRU owner, bounded
active KV restoration, 32-token alignment for longer prefixes and cold
fallback for unsafe short cross-block continuations. The published divergent
Chinese-chat reproduction restores 15 tokens and computes 11. Its 52
generation pairs, 68 full-vocabulary comparisons, 19 text-matrix cells and
one-hour soak are sibling qualification, not Windows acceptance.

The protocol changes are independently implemented in the existing Rust server;
no new runtime dependency, GPU binary, arithmetic threshold or model weights
were introduced. Source tests are not real-model acceptance. The next release
still requires a clean all-component Windows build, exact archive/inventory
binding, crash-safety regression, GB10 numerical and continuation gates,
correctness-attached retained performance, prefix/stream/API tests and the
issue's randomized neighboring-length continuity matrix. Linux's looser/different
quality metrics cannot replace the Windows contract. Any inherited evidence
must explicitly identify unchanged components; immutable release tags must not
be moved to cover later source changes.
