# Checking a cached owner's continuation

`qrt-product run --expected-owner-output FILE` adds a separate correctness
transaction after the existing prefix fallback seed and initial suffix retry.
It requires `--prefix-tokens`, accepts an original owner continuation of
2 through512 token IDs, and cannot be combined with `--checkpoint-owner`.
The reference file is validated before model loading.

The fallback's actual first generated token is appended to a copy of the
original owner prompt. The existing prefix-stream API then consumes that one
token and generates the remaining outputs. Expected IDs are used only for
comparison. No runtime ABI, kernel, provider or model arithmetic changes.

The observer checks every actual output ID, input identity, callback order,
phase and contents, token-bound finite first logit, copy-on-write transaction
and restored prefix state. Callback rows use type `owner_token`, with indices
relative to the additional continuation. The `prefix_owner_continuation`
record includes the original first token, complete combined output IDs,
prompt hashes, returned counts and status, restoration and stream results.
The reported continuation logit belongs to combined output index1. An
external oracle must separately supply any numerical tolerance applied to it;
the original owner's first logit remains in its cold-prefill evidence.

The following ordinary suffix hit still executes and checks its own complete
output and restoration. A failed owner check exits before that timed hit.
The owner transaction has its own wall clock and summary field
`prefix_owner_continuation_ms`; it is excluded from both `prefix_seed_ms`
and the existing timed-hit TTFT/TPOT. Complete process wall still includes
all work. Enabling this extra request changes warmup history, so comparisons
must declare it and keep it consistent across timing arms.

This boundary combines a cold owner's first output with a continuation from
its restored cache. It is not a single cold request producing all outputs,
and does not replace a separate cold-continuation or cold-TTFT measurement.
Existing commands without the option submit the same requests as before.

The host test executes the complete CLI against a deterministic test backend,
replacing only Windows DLL preload and native engine calls. ASan/UBSan cases
cover2/32/512 outputs, the unchanged default route, incorrect references,
untrusted counts, input mutation, malformed/missing callbacks and restoration
failures. A backend that falsely reports restoration is detected by the next
ordinary hit. These tests are observer and orchestration checks only; the
separate native qualification follows below.

All45 host unit tests pass: five product tests,32 route contracts and eight
packaging contracts. The new full-CLI fixture executes45 process cases,
including31 backend failure modes and the separately detected false-restore
case. C11 strict syntax checks pass. Every other compiled CLI source/header
and the Windows build script remain byte-identical to the qualified
`24c4304` CLI source.

[Host commands, source hashes and complete results](../benchmarks/correctness/prefix-owner-continuation-local-20260918.json)
retain the initial test fixture's signature correction. No real inference,
model timing or release acceptance is claimed by that record.

Native qualification, September18: CLI source
`6d9602cb9e51096f4e336ecf07a2b220e53f6a54` builds on baiying. Its949248-byte
executable SHA256 is
`153c0970dc2c4d1d3f8536ed698005842169431d412741a736cee185f731c78d`.
All eight compiled inputs match that commit. With the existing whole
`ddacdc9`, CK `ea6faff`, MoE `9235750` and FLA `7b20c90`, the original
q8192 request matches all512 GB10 IDs, first logit10.375 and all callbacks.
The owner option is disabled for this regression. Load is21264.07 ms,
TTFT23392.727499 ms and TPOT97.640613 ms; this unpaired run does not replace
the retained performance baseline.

The full16384-prefix run checks cold first token16/logit25.625, all31
additional owner outputs and their callbacks, then both512-token suffix
requests. Every original GB10 ID matches. Both suffix logits are5.6875;
restoration and changed-prefix rejection pass. The additional owner
transaction takes5037.9647 ms. Its index1 logit24.5 is checked for finite
token-bound reporting only, because the original reference has no index1 raw
logit. Load is21300.7954 ms; the later suffix hit takes8410.8466 ms TTFT and
158.067077 ms TPOT. These remain separate from a single cold out32 request.

The first external analyzer used the standard FNV offset basis, while this
repository's existing CLI and original oracle use1469598103934665603. A
versioned correction preserves the runtime and reference files, checks all
raw IDs, and independently verifies original prompt/output u32le SHA256.
[Native commands, binaries, reference boundaries and the analyzer correction](../benchmarks/correctness/prefix-owner-continuation-native-product-20260918.json)
are attached to the same runs. Package, larger-context and retained-speed
acceptance remain open.
