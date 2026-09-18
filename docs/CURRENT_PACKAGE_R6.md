# Current portable candidate R6

The unpublished `v1.0.2-current-stack.20260919.r6` archive is132397965 bytes,
SHA256 `2cd3798411e33f10cc31b944aa69e2453005a0761f1f637f7f0b7d33aa3349c7`.
It contains server `b3d75e9`, CLI `6d9602c`, whole provider `ddacdc9`,
CK `ea6faff`, MoE `9235750` and FLA `2b33665`. The later opt-in failure
capture diagnostic is not part of this archive.

On baiying, the archive passes CRC, all284 release-file and268 runtime-file
hashes, relocation while the original staging directory is unavailable,
six packaged document-checker cases and CPU preflight. Its532-option profile
expands33 portable paths. The other499 option values match the qualified16k
component stack. R5's failed CPU preflight remains preserved: its configuration
mistakenly used the CLI checkout for the independently pinned server identity.
R6 assigns the actual server checkout as the HTTP working directory.

Three fresh real-model service processes pass on the extracted archive:

| Scope | HTTP requests | Native wall ms | Observed boundary |
| --- | ---: | ---: | --- |
| Tool/text and admission |45|206287.381|Original q8192/out512 IDs and first logits, nested repair JSON/SSE, text tokenization,27 media rejections before inference, final q8192/out32 SSE|
| Saved partial prefix |15|140983.645|256 raw branch IDs,12 first logits, four SSE branches, original state restoration and owner replacement|
| Protocol and queue |55|478147.778|30 protocol checks, actual queue timeout/overflow, waiting-request cancellation, active q8192 completion during shutdown, tools and thinking|

All host guards pass and all three services exit normally. Original GB10
outputs and the0.125 raw first-logit tolerance remain mandatory. SSE text,
usage and completion are checked separately; SSE does not expose raw token
IDs. The [combined evidence](../benchmarks/correctness/current-portable-r6-protocol-prefix-controlplane-20260919.json)
includes raw response bodies, commands, component identities and independent
boundary replay.

This establishes the listed package and service checks only. The13-case cold
matrix, one-hour soak, remaining long contexts and immutable performance
targets remain open. There is no publication or release qualification.
