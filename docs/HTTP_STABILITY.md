# Unreleased native HTTP stability qualification

Date: 2026-09-09. This record does not approve a release. The known null-output
repair remains in force; observed healthy runs do not exclude every possible
hardware/driver lockup. The q7169 gate remains failed and the GB10 reference
machine is currently unreachable.

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
