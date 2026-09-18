# Long cold reference continuations

The original16384,32768,65536 and131072-token owner prompts now each have
a512-token continuation captured on GB10. Prompt IDs are unchanged. Both
immutable q7169/q8192 controls pass, and every existing owner32 continuation
and complete248320-value first-logit tensor reproduces exactly.

| Prompt tokens | First token | Raw first logit | Captured outputs |
| ---: | ---: | ---: | ---: |
|16384|16|25.625|512|
|32768|16|24.75|512|
|65536|16|24.25|512|
|131072|16|25.0|512|

The authority is `gb10-4t`, actual host `aitopatom-66c4`, using the original
model at `/mnt/data/models/Qwen3.6-35B-A3B`. Capture source remains
`1c9511395bf7f9dcc78a8aedcb05c700feee0568` with dependency source
`9b2a99edeede83ce77a4b775a509256f65894a7f`. The pinned container and numerical
configuration remain unchanged, including the existing reference MTP setting.
The first-logit observer returns the original output unchanged. Prefix caching
is disabled and no native engine tensor or expected output is supplied as input.
Both controls precede all new requests. The1800-second capture, host memory
guard, frozen autotune cache check and GPU process cleanup pass.

The [new independent oracle](../contracts/gb10_long_cold512_actual_tokens_20260919_oracle.json)
adds these continuations without replacing earlier contracts. Its raw capture
SHA256 is `d834107c2266ed7065ade8141630fa684e7332d42f3567ca32a4d462c0e27374`.
The [reference verification](../benchmarks/correctness/gb10-long-cold512-reference-20260919.json)
binds all53 downloaded files, prompt/output fingerprints, all six full-vocabulary
logit files, source and command identities. Intermediate layer transactions were
not captured in this run; existing long-prefix transaction evidence is separate.

These references enable the Windows cold out512 checks. They do not establish
Windows correctness or performance at any newly captured length.

A separate bounded capture now observes the original layer33 transaction at
positions114688–122879 of the same131072-token prompt. The pinned read-only
observer saves Q/K/V, gates, beta, initial/final recurrent states and the core
output, totaling207093760 bytes. Every value is finite. Its8192 input IDs
match that exact slice of the original prompt. All512 outputs and the complete
first-logit tensor reproduce the earlier independent reference; both immutable
controls also pass. The original model methods, numerical configuration and
frozen autotune inputs remain unchanged. This capture receives no Windows
tensor as input.

The [layer33 window evidence](../benchmarks/correctness/gb10-prefix128-layer33-window-20260919.json)
binds all1619 downloaded files, the original transaction14, eight tensor
identities, prompt/output checks, command and source hashes. The command is
`run-qrt-gb10-prefix128-layer33-20260919-r1.py` on `aitopatom-66c4`; source and
dependency commits are unchanged from the table above. Host, deadline and GPU
cleanup checks pass. It supplies a GB10 comparison for the repeated Windows
delay location, without claiming Windows128k acceptance or identifying the
delay's cause.

The original window now passes the native seeded key-major interface using
FLA provider `7a33fc9`: every33554432 BF16 output and524288 FP32 state cell
matches bitwise, the seeded repeat is stable, and the zero-state control
differs. The [native component evidence](../benchmarks/correctness/fla-completed-delay-native-diagnosis-20260919.json)
pins the separate tool and provider identities. This remains a component
comparison; full-model Windows128k qualification is pending.

The same independent capture now covers layer8 at positions114688–122879.
All 576 output IDs across the original owner and two controls pass, with
zero first-logit error and the original full owner first-logit tensor.
Eight finite tensors total207093760 bytes. Source, dependency, numerical
configuration and observer remain unchanged. Command:
`run-qrt-gb10-prefix128-layer8-20260919-r1.py` on `aitopatom-66c4`.

The post-run wrapper mistakenly looked for `linear-8`; the observer writes
`linear-08`. Its original failed validation record is preserved. A bounded
download recovered the eight already captured files using their original
hashes, and local verification checks all1619 files and the actual prompt
transaction. The model was not rerun. The
[layer8 window evidence](../benchmarks/correctness/gb10-prefix128-layer8-window-20260919.json)
includes this recovery and the independently recomputed qualification.

The Windows failure capture's completed output at positions119808–120831
matches this reference slice bitwise: all4194304 cells, maximum error0.
The [direct failed-output comparison](../benchmarks/correctness/fla-failed-layer8-gb10-output-comparison-20260919.json)
binds the original Windows log, segment order and captured values. It
provides external component evidence for the completed-latency repair;
full-model Windows128k qualification remains separate.
