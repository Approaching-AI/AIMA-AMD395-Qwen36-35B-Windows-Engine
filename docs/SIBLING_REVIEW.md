# Linux sibling fixes reviewed for the next Windows release

Review date: 2026-09-12. The Windows upstream's open issue remains
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
| Variable cold lengths / serial tails (Linux #1/#5) | The strict r4 components pass all eight cold cases, 1,216 raw outputs and eight exact first logits. q8192 actual callback TTFT is 58.839 s; retained performance and the wider length matrix remain open. See the [complete cold record](../benchmarks/correctness/dense-norm-complete-cold-20260912.json). |
| Logical GEMM/FLA extents | Audit logical sequence lengths separately from padded allocation/launch extents. Linux `3d284a3` and `e402b3a` are useful arithmetic references, not portable Windows binaries or authority to loosen token/logit gates. |
| Thinking (Linux #6, .5) | The top-level object, legacy-alias conflicts, tokenizer parity and live reasoning streaming pass in the relocated r4 archive. The Windows disabled default and explicit thinking both pass; empty-kwargs behavior is consistent. |
| Repeated tools (Linux #7, .5) | Canonical deduplication, declared-function admission, one-call mode and conservative no-progress metadata have CPU regressions. Native packaged HTTP also passes tool output and continuation. Semantic retry strategy and side-effect authorization remain with the caller. |
| Control-plane responsiveness (.5) | The relocated r4 archive passes all seventeen positive protocol checks across thirty requests: FIFO, timeout503, overflow429, health and shutdown, with the active q8192 retaining all32 tokens and exact first logit. All owned processes exit normally. See the [portable HTTP record](../benchmarks/correctness/dense-norm-portable-http-20260912.json). |
| Default VL reasoning (.6) | Windows remains text-only; no vision implementation or inherited VL result is claimed. |
| Partial shared prefixes (.7 / closed Linux #12) | Complete K64 checkpoints and seeded FP32 FLA qualify the saved7168 single-input continuation. Relocated r4 HTTP also passes four actual divergent branches, 256 raw tokens, twelve first logits, four SSE comparisons and twelve complete owner rollbacks after decode. General long-prefix batch suffixes remain open: the registered16384+1024 sequential route still fails raw output102 despite an in-tolerance first logit. See the [long-prefix gap](../benchmarks/correctness/long-prefix16k-prefix-gap-20260912.json). |

The GitHub release list was rechecked again on September 12 and still
starts with `.7`; no newer stable release is inferred from local branch commits.

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
were introduced by those protocol changes. The r4 archive's native build,
inventory, relocation, API and short-prefix checks are now recorded; later
GPU changes need matching qualification. The next release still requires
correctness-attached retained performance, general long-prefix and context
coverage, the issue's neighboring-length continuity matrix and a resident
soak. Linux's looser/different
quality metrics cannot replace the Windows contract. Any inherited evidence
must explicitly identify unchanged components; immutable release tags must not
be moved to cover later source changes.
