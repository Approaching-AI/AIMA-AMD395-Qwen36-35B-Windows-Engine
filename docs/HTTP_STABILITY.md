# Unreleased native HTTP stability qualification

Updated 2026-09-12. The strict runtime passes all eight cold CLI cases and
1216 GB10 outputs, plus q8192 HTTP from the extracted portable archive. The
archive loads in 20206.2284 ms and returns all 32 frozen nonstream tokens;
both requests report first token144/logit10.375. SSE text/usage, health, queue
drain and normal exit pass. Server source is 545636c; see
`benchmarks/correctness/runtime-portable-archive-20260912.json` for the complete
component inventory and native command. These are reported HTTP timings;
the selected cold CLI actual callback remains 58662.0706 ms, above the
unchanged 4187.415605 ms target. No release is qualified.

Positive protocol tests expose two additional gaps. The strict short/smooth-tail
MoE ABI cannot publish unrounded residual variance. Explicitly selecting the
existing dynamic full ABI fixes five-token text and 17-token chat, including
their SSE comparisons, but the first 294-token tool request then returns 500:
an out1 seed did not establish the live prefix state assumed by the bridge.
See `benchmarks/correctness/http-short-provider-gap-20260912.json` and
`benchmarks/correctness/http-prefix-seed-gap-20260912.json`. These short
functional outputs have no GB10 numerical qualification yet.

The repair at b619793 selects the compatible logical full-MoE ABI automatically
for strict prompts below 4096, retains the padded selected route above that
boundary and excludes incompatible smooth-tail calls in strict mode. The
server now reuses only complete native checkpoints and otherwise prefills.
Fallback cannot repeat already emitted callbacks. Host regression executes
the real C bridge against controlled native responses.

The repaired r2 archive now passes the positive protocol matrix on baiying:
text/chat/SSE, structured tool calls, tool-result continuation, tool SSE,
FIFO queue, context rejection, default-disabled thinking, explicitly enabled
thinking with separate reasoning/answer SSE, queue timeout/overflow and
shutdown cancellation of waiters. The active q8192 request finishes normally
during shutdown with all 32 GB10 tokens and first token144/logit10.375.
The final process loads in 19877.3452 ms and exits normally; host checks pass.
Its reported q8192 TTFT is 54730.0623 ms, not a cold callback measurement.

The 124959485-byte ZIP has 280 verified files and retains the same portable
profile with no temporary dynamic-MoE override. Three bounded runs share its
source and artifact hashes. The first two original controller failures remain
recorded: the first assumed the VL thinking default instead of the documented
text default; the second expected a final answer before a valid 256-token
length cutoff. The 512-token test completes reasoning and the visible answer
323, with matching ordinary/SSE output and no think delimiters leaking.
See `benchmarks/correctness/http-positive-protocol-20260912.json` for the
commands, build/archive inventory, all responses and these explicit limits.
Short chat/tool output has no independent GB10 token qualification yet.
The later a797b62 whole/core/server r3 archive also passes saved-prefix HTTP
after a normal 32-token owner response. All four branches really restore saved
state: 256 raw tokens and twelve first logits match GB10, four SSE comparisons
pass, and all twelve transactions restore the advanced owner and allocations.
Owner replacement and the old branch's subsequent cold fallback pass. Load
is 20181.3837 ms; the four branches' 141.1245–519.2788 ms reported warm TTFT
excludes the 47705.8181 ms owner prefill. The default checkpoint switch remains
off while wider suffixes are unqualified. See
`benchmarks/correctness/http-prefix-after-decode-20260912.json`.
Short chat/tool numerical qualification, long contexts, the renewed full cold
matrix and retained performance remain open. No release is qualified.

## Historical 2026-09-09 HTTP run

The following evidence used the earlier fast profile and its limited q8192
boundary. It does not qualify the current strict arithmetic route. Its then
unavailable GB10 service is restored, and the q7169 gate now passes in the
strict eight-case matrix above.

The retained-package preload mismatch is fixed at `310e872`: ordinary and
split-tail profiles require eight power-of-two modules, while explicit
dense-ceil still requires nineteen. Previously the ordinary eight-module
package failed before readiness because unused dense modules were demanded.
The fix changes host-side loading only, with no relaxed kernel, correction,
memory or numerical threshold. Its real-model HTTP regression is safe.

Server `a5eac5f` also corrects the aggregate-versus-mean TPOT field and provides
explicit read-only first-token observations. Its native Windows build passes
40 server tests. The latest native Windows real-model run uses that server
with the separately fingerprinted `310e872` provider, BF16 batch one on
baiying. The model reference is `Qwen3.6-35B-A3B`.

| Check | Result |
|---|---|
| Native model/engine load | 19971.7296 ms |
| Non-stream q8192 output | Exact 32-token GB10 continuation |
| Both first-token observations | 144 / raw logit 10.375 |
| Native TTFT, non-stream / SSE | 4195.6053 / 4207.6657 ms |
| Mean TPOT, non-stream / SSE | 32.6037290 / 32.5930258 ms |
| SSE | Expected text, usage, length finish and DONE |
| Health / queue / shutdown | Responds during generation; queue drains; normal exit |
| Host after run | Same boot, no remaining engine, healthy driver/memory |

Two verifier defects are separately repaired: null intermediate SSE usage
and a comparison of differently seeded prompt digests. In observation v1,
offset `14695981039346656037` yields `1bfa7bb4a1d00a65`; the frozen QRT oracle
uses historical offset `1469598103934665603`, yielding `1584e34d56e5d78b` for
the identical SHA-verified prompt. Neither the oracle nor the tokens were
changed. Verifier `9eb31fd` validates both against the exact prompt and checks
the saved native first-token/logit observations. Original failed controller
records are retained, with independent offline validation records; no extra
GPU work is submitted to repair a comparator. Local checks pass 223 Python
and 45 Rust tests, C ABI, clippy, q16 and public hygiene.

Core fingerprints:

- Server: `7326159893a2308c42eeb90d46e860fe5c128840874ac5ba02a780d0166a3159`.
- Provider: `2b91dfb18f623038def217143049f75d0bf0036247d0e8f190867fe722f7e60d`.
- Original live result: `deb472d11110dc6acd89f5c16d29fec0e76c677bcd89995ec0e9a9398de0d7a7`.
- Host-guard result: `a1794f9c04773dcfd1b9b2d591490a39a1a22c5ac7b976e0ea494bd765de3626`.
- Offline validation: `ce0bc36222224ade86db9e664d1b556577dd43ac710ebb358a5503b28ea8f6fd`.

The commands are `baiying_build_qrt_server.ps1` and
`verify_baiying_q8192_http.py --execute`, each inside
`baiying_guarded_inference.ps1`, using the frozen q8192 contract and a fresh
loopback foreground service. Exact configuration, command lines, model path,
component provenance and host snapshots remain in the local qualification
records. The test rechecks all 211 original runtime inventory entries; this
is a composite candidate, not a newly qualified single release archive.

The first-token/logit/non-stream continuation boundary passes. SSE exposes
text rather than raw token IDs. Prefix isolation and arbitrary lengths are
not proven by these two identical q8192 requests. Native TTFT is not a
client first-token measurement, and it also misses the retained
4187.415605 ms target. q7169, randomized neighbor lengths, retained caller
performance and complete package qualification remain open. No publication
or issue closure follows from this result.
