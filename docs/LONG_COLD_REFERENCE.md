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
