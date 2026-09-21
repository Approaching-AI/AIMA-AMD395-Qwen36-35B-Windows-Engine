# Cold prefill final chunks

Current result: source99fb67f builds on baiying and passes all512 ordinary
q8192 outputs/callbacks, first144/logit10.375. Its admitted q8193 cold tail
completes but emits64 instead of the original GB10 token220. The new single
input bridge below has passed local checks and still needs a Windows build
and the unchanged original-model boundary. See
[actual controls, diagnosis and local bridge evidence](../benchmarks/correctness/single-tail-q1-local-20260921.json).

The cold chunk coordinator previously accepted only multiples of 8192 and
an optional 1024-token tail. This rejected the original q8193 and
q262140/q262142/q262143 native MTP cases before their correctness boundary
could run. It now retains the true final extent from 1 through 8192 inputs.
Every earlier chunk still contains exactly 8192 original tokens; the prompt
is never padded or substituted.

The suffix convolution ring also assumed at least four new inputs. A final
chunk of one, two or three inputs would underflow its source row. The actual
kernel now replaces only the new ring slots and preserves the preceding
prefix slots. Both FP32 and BF16 ring layouts follow this rule.

Finally, the C entry point previously split prompts of 8193 through 9216
tokens into an 8192-token seed and a prefix/decode suffix before the whole
provider saw the request. With native MTP and cold chunking both enabled,
that shortcut now delegates the complete prompt to the chunk owner. This
preserves the original full-prompt shifted inputs and checkpoint provenance.
Other C entry modes retain their existing behavior. Native MTP remains opt-in.

These changes do not expand the public prefix-cache suffix interface or
permit native MTP to start at or beyond its 262144-token drafter limit.
The existing standalone native route already admits q8191; the q7169/q8192
restriction belongs to a separate diagnostic capture mode.

## Local verification

The actual coordinator passes 26 ordinary prompt lengths and 13 native MTP
lengths, including single-token tails and the three original retirement
prompts. Failure checks cover short-tail publication, known and unknown
completion, checkpoint saves and cleanup. The extracted production ring
kernel passes 18 extents, four prefix residues and both formats: 144 shapes
under ASan/UBSan, including unchanged slots and allocation guards. The C
bridge checks the two route flags independently, cancellation, callback
clocks, metadata and ownership.

The following focused runs pass, totaling 23 distinct Python tests:

```sh
PYTHONPATH=tests python3.12 -m unittest test_prefill_chunks test_prefix_batch_suffix test_bounded_prefill_suffix
PYTHONPATH=tests python3.12 -m unittest test_prefix_batch_suffix.PrefixBatchSuffixTests.test_actual_ring_update_keeps_unreplaced_prefix_slots test_attention_suffix test_prefix_fla_suffix test_mtp_chunked_prefill_capture
PYTHONPATH=tests python3.12 -m unittest test_mtp_native_prefill_integration test_native_mtp_decode test_mtp_drafter
make c-smoke
python3.12 scripts/test_q16_transaction_contract.py
```

The C smoke and all seven transaction cases also pass. Three separately
restored old implementations compile successfully and fail their respective
new checks: ring underflow, coordinator tail rejection and C shortcut
misrouting. The initial evidence writer failed to serialize subprocess
bytes after executing these controls; its partial record and fixtures are
preserved, and the corrected writer records all expected outcomes.

[Committed source hashes and completed local outcomes](../benchmarks/correctness/cold-prefill-tail-local-20260921.json)
bind these checks to a7e2092. Positive command results are recorded from the
local tool output; complete stdout logs are not claimed. The negative-control
report includes its commands, generated-source hashes and actual failures.

## Actual native controls and terminal admission

The a7e2092 whole provider, normal CLI and prefix probe now build on baiying
with all173 source inputs verified. The ordinary q8192/out512 control passes
all original outputs and callbacks, first144/logit10.375. Three native MTP
controls also pass: q7169/out32, q8191/out32 and q8192/out512, with31/31/511
native target commits respectively and no ordinary decode commits. Their
first tokens/logits are82/9.25,168589/11.375 and144/10.375. These are functional
controls; native MTP remains opt-in and no retained performance changes.

The actual q8193/out32 run reaches the final one-input chunk, then fails
at layer39 before emitting any token or callback. Its full-prefix predicate
required more than one input, disabling the whole/QKV provider needed by
the compact attention path. The repair admits one input only inside the
actual terminal-only cold transaction. Standalone one-token requests retain
their existing selection even when the full-prefix or MTP flags are set.

The [actual runs and local repair checks](../benchmarks/correctness/terminal-cold-tail-local-20260921.json)
record204 predicate cases and66912 actual provider-selection cases under
ASan/UBSan. Restoring the old predicate reproduces the failure. Four related
coordinator/C-bridge/clock regressions pass. An initial test extractor error
selected a forward declaration; the corrected test selects the production
definition, and both logs are retained. Its actual99 build takes119961.528ms.
The ordinary q8192 control passes with load21405.1728ms, TTFT23359.7708ms and
TPOT101.500041ms. The actual q8193 run completes8192+1 inputs and31 native
commits but returns first64/logit9.75 instead of220/logit9.75; native exit6,
wall57197.527ms. A second observed run reproduces that wrong output boundary.

All80 tail norm files compare against the original qualified GB10 transaction
at position8192/input63. Layer0 input is bit-exact; its postattention norm
differs974/2048 BF16 values, maxabs0.03125. This is not an isolated LM-head
diagnosis. The initial observer run exits before output because the numeric
environment cache retained fallback position8191 when the next chunk had one
input. Explicit local position0 permits the complete norm capture. Text
carrier/final-norm markers are absent from the filtered log.

The new terminal bridge sends that one actual input through the existing
resident q1 export with its original prefix arithmetic. It resizes empty,
unpooled tails to1536 tokens, uses a private request without callbacks, and
copies the fenced final BF16 normalization into the actual MTP target row.
The cold coordinator promotes KV/recurrent counters and publishes the final
sample only after both states finish. A cold-input scope does not select the
different arithmetic used after MTP retirement. No original oracle input or
output is substituted. q1 tails leave batch descriptor timings empty.

The actual bridge passes31 ASan/UBSan cases covering exact input/norm/sample
handoff, nested scope restoration, exceptions, stale metadata and failed
copies. Coordinator shape/failure tests, C bridge and native retirement
regressions pass. Numeric u32/i32/u64 caches preserve configured values and
absence, but resolve each caller's fallback afresh;752 checks and three actual
old-parser negative controls pass. The initial expanded parser fixture chose
a forward declaration; the corrected extractor and both outcomes are kept.
The new bridge still needs its actual Windows build and q8193 token boundary.

The separate8612387 full256k run completed with a mismatch at output124.
Its matching first-step operands do not qualify the later continuation or
the three original native retirement cases, which remain unrun.

The [new build preparation](../benchmarks/correctness/cold-prefill-tail-prepared-20260921.json)
binds 173 source inputs: 167 whole-provider, 10 prefix-probe and 8 normal CLI
inputs in overlapping compilation closures, plus the build scripts. Only the
three repaired implementation files differ from the prior build closure.
Both Windows command files parse successfully. The active-process admission
check performs zero dispatch calls. Compilation deadlines are 240/180/180
seconds, within a 660-second guarded process and 750-second transport bound.

The current seven-case native transport preserves the earlier observer logic
and original reference files. Its r3 binding uses the successful a7 build and
the actual completed long-run failure record; the two earlier preparations
remain historical. New terminal-repair binaries require fresh bindings before
repeating the short controls and proceeding to the retirement cases.

The [server rebuild preparation](../benchmarks/correctness/cold-prefill-tail-server-prepared-20260921.json)
also binds the HTTP server's statically linked C core to a7e2092. Its 25
source/build inputs include nine C inputs and ten Rust inputs. Rust sources
match the prior qualified server; qrt.c and qrt.h have changed. All 54 existing
Rust library tests and formatting pass on the Mac, with complete logs and
unchanged source hashes. These tests do not exercise Windows model inference.

The Windows server command parses. The dispatcher's missing-build check
makes zero network or native calls before the same-source whole/CLI/probe
build completes. Its
test/build deadline is 900 seconds inside a 960-second guard and 1050-second
transport bound; Git child processes have separate 30-second deadlines.
The server executable remains unbuilt. The ordinary q8192 generator also
stops before binding binaries until the new build exists. Earlier R9 package
bindings refer to older components and must be replaced before packaging this
repair. Final-artifact HTTP, prefix, context and one-hour soak checks remain open.

The [new portable-package preparation](../benchmarks/correctness/cold-tail-package-prepared-20260921.json)
requires the actual a7 whole/CLI/probe and 54-test server builds, plus the
same-artifact original q8192/out512 boundary, before binding any executable
hash. It will replace the whole DLL, normal CLI and static-C-core server
together and retain the separately pinned CK, FLA and MoE providers. Every
provider identity and the complete535-option portable profile must match the
declared functional control. Its three profile differences are explicit;
chunked prefill still requires final-package model tests.

Preparation passes Windows parsing, zero-dispatch missing-build checks and
five deliberately incorrect profile/baseline bindings. The existing861 control
checks profile normalization only and does not qualify the a7 package. No
bound package input, archive or relocated runtime exists yet. Expected inventory
is269 runtime artifacts and285 release files, including the model-independent
square-root table. Staging will create an unqualified candidate for the final
archive tests; all context, native retirement, HTTP, soak, load and performance
requirements remain open. This preparation does not reuse the stale R9 binary
bindings or claim that the active861 full256k run qualifies the repaired C core.

The [final-archive regression preparation](../benchmarks/correctness/cold-tail-package-regressions-prepared-20260921.json)
now covers all three HTTP suites, all13 cold CLI cases and the complete
one-hour same-process soak. Their shared validator requires actual component
builds, matching archive inventories, relocation and the numerical profile.
It checks the repaired whole DLL, CLI and static-C-core server together.
Historical R6 stage records are only a positive control for this validator;
twelve in-memory faults are rejected. All20 missing-binding paths stop before
any subprocess or network call. Four Windows command files parse successfully.

The HTTP workers preserve the original45 protocol,15 saved-prefix and55
control-plane requests. The CLI matrix retains all1856 original output IDs,
first logits within0.125 and actual callback order. The soak retains its full
3600-second active window,4200-second controller,120-second request bounds,
4500-second native guard and original resource/cleanup checks. Its preparation
requires offline replay of both the same-archive HTTP results and all13 cold
cases before binding. No executable hashes, requests or model results are
created by these preparations.

After actual staging, the checks and matrix generators create their bound
plans. Run the three HTTP actions and their analyzer, and all13 CLI cases
with the short/cold analyzers. Their completed evidence then supplies the
soak generator. The source report retains the exact generator, worker,
dispatcher and analyzer files. Earlier unrun package scripts remain historical.

Next validation uses the original q8191/out32 and q8193/out32 boundaries,
same-build ordinary and native q8192 controls, then the three original native
retirement cases. Host transport checks establish no native numerical,
performance or release acceptance.
