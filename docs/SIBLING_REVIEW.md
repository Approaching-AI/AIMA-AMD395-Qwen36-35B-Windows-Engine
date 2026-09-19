# Linux sibling fixes reviewed for the next Windows release

## September 20 status: original continuations and archive boundaries

The [API refresh](../benchmarks/correctness/linux-windows-release-review-refresh-20260920.json)
at 2026-09-19T18:02:10Z rechecks the latest five releases: `.10`, `.9`, `.7`,
`.6` and `.5`. All five bodies match the preceding review, and the latest `.10`
tag still resolves to `0522a57caf24bf21e0e6fcc5b234a7fc361fbc94`, with declared
native source `ec9934446911fdf376da8eebcd83e7b137efbb7c`. The source review below
therefore remains applicable. This refresh executes no model.

The subsequent Windows long-context repair is independently GB10-anchored.
A [single-round RoPE correction](../benchmarks/correctness/prefix96-rope-single-round-root-cause-20260919.json)
fixes the BF16 midpoint at position91018, layer15. The
[original128k product run](../benchmarks/correctness/rope-single-round-prefix128k-product-20260919.json)
then matches the combined owner32 and both complete512-token suffixes,
first logits, actual callbacks, restoration and changed-prefix rejection on
whole32a/FLA1d. Separate cold owner512 and other binary combinations retain
their own acceptance boundaries.

The newer wholeb3/CK370 storage stack separately passes
[original q8192/out512](../benchmarks/correctness/resident-ordered-compact-query-native-q8192-20260919.json)
and [cold32k/out512](../benchmarks/correctness/resident-ordered-compact-query-cold32k-out512-20260919.json),
with all original IDs/callbacks and zero first-logit error. Its original full256k
owner and both suffixes are still running. q8192 TTFT remains above10 seconds;
the immutable retained-performance target has not changed.

Serverb3d75e9 is now included in the unpublished R6 archive. Its actual
[protocol, saved-prefix and control-plane checks](../benchmarks/correctness/current-portable-r6-protocol-prefix-controlplane-20260919.json)
pass45 protocol requests, saved-prefix branches and55 control requests, and its
[exact ZIP inventory](../benchmarks/correctness/portable-archive-inventory-20260919.json)
is independently verified. R6 still lacks its exact-archive13-case matrix and
one-hour soak. R8's newer whole/FLA stack and test commands are prepared but
unrun. Later model components cannot inherit archive qualification from R6.
No new archive is release-qualified or published. Windows still rejects visual
media explicitly; Linux VL and numerical/soak results do not qualify it.

## September 18 refresh: Linux `.10` tool content

The release API retrieved at 2026-09-17T21:30:40Z identifies
[`v1.5.1-native-vl.10`](https://github.com/skyguan92/AIMA-AMD395-Qwen36-35B-Linux-Engine/releases/tag/v1.5.1-native-vl.10),
published 2026-09-16T20:58:30Z, as the latest release. Its tag resolves to
`0522a57caf24bf21e0e6fcc5b234a7fc361fbc94`; declared native source is
`ec9934446911fdf376da8eebcd83e7b137efbb7c`. Comparing it with `.9` native
source `29f67beda7199575c62020d2b95e24c0a754c9a9` changes only
`native/src/native_chat_protocol.cpp` under native source. GPU arithmetic
is unchanged. The [source review](../benchmarks/correctness/linux-tool-content-source-review-20260918.json)
preserves the release response, immutable reference, file hashes and diff.

Linux now accepts image/video/mixed tool results through its existing media
pipeline. It keeps original status text separate from visual placeholders:
media-only payload can count as progress, while explicit text/JSON failures
cannot reopen an exhausted retry window merely by including media. Matching
preceding calls, unique result IDs, a real user query and existing media
limits remain required. Linux's published numerical matrix, soak and unchanged
dependency inventory qualify that Linux artifact, not Windows.

The audit found a Windows text-only analogue: the template concatenated text
parts, but the retry policy treated every nonempty array as useful output.
Server source `ac1a41f` now classifies the same concatenated text. Empty parts,
explicit errors, encoded JSON failures and failed final-output wrappers retain
their retry counts. Useful results and different-command silent repairs can
still reopen the window; JSON arrays encoded inside result text remain data.
Image/video and mixed-media tool results continue to fail explicitly at chat
and tokenization admission. No Windows visual capability is inferred.

The [local/native build record](../benchmarks/correctness/tool-text-parts-local-native-20260918.json)
contains a before-fix failing regression, 53 passing Rust tests on both hosts,
Windows MSVC/Rust service build, 10 caller-document tests and 2 verifier tests.
The new in-process checks cover 56 JSON/SSE recovery requests and 27 media
admission requests. Server C ABI sources, Cargo dependencies and provider math
are unchanged. The [real-model HTTP record](../benchmarks/correctness/tool-text-parts-native-http-20260918.json)
qualifies eleven JSON/SSE retry cases, four tokenizer parity cases and 27 media
rejections. Its first process completed all 512 GB10 outputs and logit 10.375;
an out32-only usage observer then reported failure. The original failure and
independent out512 revalidation are retained. Only the remaining 61 HTTP
requests were run in the next process, including final q8192/out32 SSE/logit.

Source `b3d75e9` additionally peels nested final-output wrappers iteratively.
The [native build](../benchmarks/correctness/tool-text-parts-bounded-native-20260918.json)
passes 54 Rust tests, including 16,384 nested wrappers with error, empty, JSON
and useful payloads in string/part form. Its new Windows executable is SHA256
`6b3e5dd79a8e39652efcaf469410ebce1a106a499661f7fbe31ecdeba8d67dae`.
The [same-process native HTTP check](../benchmarks/correctness/tool-text-parts-bounded-native-http-20260918.json)
passes 45 requests: fresh q8192/out512, three nested-wrapper JSON/SSE pairs,
four tokenizer pairs, 27 media rejections without inference, and final
q8192/out32 SSE. All 512 original GB10 IDs and both first logits 10.375 pass;
actual early callbacks, queue drain and normal shutdown pass.

Both server experiments use the unchanged R4 provider/AOT/table inventory and
numeric profile. The latest server loads in 21529.5239 ms; its cold TTFT is
30231.4271 ms and TPOT 100.080780 ms. These are protocol-qualified R4 timings,
not a replacement for the newer experimental performance stack. The `.9` R4
archive remains unpublished and unchanged. These HTTP-only measurements do
not establish later archive or soak acceptance. Retained performance and
long-context gates remain open, as described in
[performance evidence](PERFORMANCE.md). The R6 packaging and remaining archive
boundaries at the September20 review are recorded above.

## Initial September 13 review

The Windows upstream's previously reviewed open issue is
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
| Partial shared prefixes (.7 / closed Linux #12) | Complete K64 checkpoints and seeded FP32 FLA qualify the saved7168 single-input continuation. Relocated r4 HTTP passes four divergent branches, 256 raw tokens, twelve first logits, four SSE comparisons and twelve complete owner rollbacks after decode. Later [chunked 16384+1024](../benchmarks/correctness/cold-prefill-chunks-20260913.json) and [32768+1024 with corrected OUT admission](../benchmarks/correctness/prefix32k-admission-product-20260913.json) each pass all512 GB10 tokens and owner restoration on their declared binaries. These results replace the earlier general long-prefix correctness gap for those cases; the latest performance stack,64k and broader branch coverage still need qualification. |

The GitHub release list was rechecked on September 13 and still
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

The clean local `.7` evidence checkout at `51d606f` was also inspected on
September13. Its cache selection, cache header, resident engine and linear
prefill source are byte-identical to the declared `edb584e` native source.
The [source review](../benchmarks/correctness/linux-prefix-source-review-20260913.json)
records file hashes and primary source links. Checkpoint inheritance, owner
self-replacement and publication only after all recurrent/conv/hidden slices
are complete remain relevant Windows checks. Linux explicitly disclaims
universal BF16 partition bitwise identity; its top-1/KLD threshold does not
qualify a different Windows continuation or justify changing the GB10 gate.

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
