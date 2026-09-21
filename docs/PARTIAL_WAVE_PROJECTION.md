# Partial-coefficient wave-owned projection

The completed `5d63d5c` experiment passes all240 generated GPU configurations
and the complete original q8192 QKV/OUT outputs on baiying. The candidate is
slower for both operators and is not selected by the product runtime.

| Full preparation and replay | Original staged2 ms | Wave candidate ms |
| --- | ---: | ---: |
| QKV, 67,108,864 original BF16 outputs | 41.2941 | 641.18 |
| OUT, 16,777,216 original BF16 outputs | 152.258 | 598.795 |

Every warmup and timed attempt matches all GB10 operator outputs, unrounded
candidate bits, inactive outputs, guards and immutable inputs. Each median uses
three rotated completed sequences and includes preparation, classification,
bitmap work where used, unsupported-row replay and completion. OUT compares
the original midpoint correction; the retained coarse-OUT provider was not
measured here. No model is loaded by these component probes, and no inference,
product performance or release gate is closed.

The standalone and replay builds complete in4956.668ms and123435.977ms.
All five native actions pass host checks and leave no process. QKV admits
3,454,884 matrix candidates and retains872,632 original replays; OUT admits
8,453,029 and retains1,026. Despite those admissions, this implementation adds
599.8859ms for QKV and446.537ms for OUT within the measured component scope.
[Commands, source/build bindings, original reference, all240 safety cases and
complete timing evidence](../benchmarks/correctness/partial-wave-projection-native-20260921.json).

The following design and preparation notes preserve the earlier sequence;
statements about pending GPU execution below describe those preparation stages.

This isolated replay replaces independent candidate dots with a wave-owned
16x16 projection tile. Four IU8 matrix instructions form each exact signed
coefficient product sum. Each lane retains eight separate carries and consumes
K16 groups in the original order. The candidate uses the previously audited
partial-coefficient and signed-remainder arithmetic; exceptional products keep
their original BF16 values and individual alignment. Rejected group certificates
use the original narrow arithmetic.

Both prepared operands use group-major 64-byte rows. Preparation certifies each
complete original row as signed zero or BF16 exponent95..159; widths are multiples
of16 through8192. A selected cell enters matrix replay only when both whole-row
flags pass. Unsupported rows use the original four-lane BF16 replay. Candidate
identities and inactive outputs are unchanged. Every metadata broadcast occurs
before candidate divergence, so unselected lanes remain valid suppliers. Partial
tiles zero-fill only nonexistent rows. No product dispatcher uses this component.

The generated GPU fixture prepares240 configurations: six shapes, four operand
families, five candidate patterns and matrix/forced-original wave variants. It
compares every selected ordered carry and output against independent original
integer arithmetic, including full-domain unsupported rows, signed zeros, large
exponent spans and a bad operand in the final group. It also checks complete-row
flags, every lossless prepared operand, inactive output/trace cells, memory guards,
immutable inputs after each invocation and production/trace parity. This fixture
has not run on the GPU yet.

The real q8192 fixture compares the original staged2 schedule with the new wave
schedule. Each clock includes its complete operand preparation, classification,
candidate bitmap where needed, unsupported-row replay and completion. One warmup
precedes three rotated timed sequences. Every attempt compares all unrounded
candidate outputs, all inactive outputs and all original GB10 BF16 operator
outputs. QKV keeps matrix4 and PPB1000; OUT keeps matrix0 and PPB10000. The OUT
comparison is the original midpoint replay, not the retained coarse-OUT provider.
The earlier r1 plan repeats the first1023 captured q7169 input rows to reach8192.
The current r2 plan uses the complete original q8192 modes described below.

The reference observer now offers `QRT_GB10_FULL_PREFILL_PRODUCT_OPERANDS=1`
to capture the original `q8192-out512` prefill transaction. It records the
layer0 normalized input and QKV projection, plus the existing complete layer3
attention surfaces and logical KV cache. Other matrix cases keep their original
observation scope; conflicting window plans and wrong prompt extents reject.
The 1 GiB per-case ceiling, original model calls, prompt IDs and complete token
matrix checks remain in force. All 37 boundary/token-observer unit tests pass.
The new capture now passes all 1,216 original matrix outputs and every first
token logit with error 0. The GB10 process completes in 370.3101 seconds,
including model startup and all eight cases, with passing host checks and
no remaining GPU process. All 12 complete tensors, 654,311,424 bytes, come
from the original q8192 transaction; none of its input rows are repeated.

ASan/UBSan host sampling on those original tensors passes 8,192 QKV and
8,192 OUT dots, all sampled GB10 BF16 endpoints and 3,131,392 eligible ordered
carries. All 8,192 QKV input rows and 8,080 of 8,192 weight rows are eligible.
OUT has 8,191 eligible input rows and all 2,048 weight rows eligible. Its one
unsupported input row is outside the deterministic dot sample; complete-row
classification rejects it. Both expected outputs now come directly from the
qualified whole-model reference. These results do not replace the existing
native fixtures or establish GPU behavior or performance.
[Complete reference, raw-logit checks and host arithmetic evidence](../benchmarks/correctness/gb10-real-q8192-prefill-operands-20260921.json).

The replay harness now accepts `--real-qkv-full-q8192` and
`--real-out-full-q8192` for those complete original tensors. It requires all
8,192 input and reference rows and reports zero repeated rows. The existing
q7169 and repeated-row modes retain their shapes. Arithmetic, producer choices,
candidate selection and timing scopes are unchanged. The complete native build
and original-data GPU comparisons remain pending the long owner's cleanup.

On the controller, ASan/UBSan sampling checks8192 dots per operator, with128 evenly
spaced input rows and64 weight rows whose phase rotates between samples. This
sample is independent of midpoint selection and is not a timing estimate.

| Host sample | QKV, K2048 | OUT, K4096 |
| --- | ---: | ---: |
| Complete eligible original input rows |7169/7169|7168/7169|
| Complete eligible weight rows |8080/8192|2048/2048|
| Eligible sampled dots |8080/8192|8192/8192|
| Ordered eligible groups |1034240|2097152|
| Original matrix certificate |593593|840113|
| Additional signed-remainder certificate |223228|565818|
| Additional partial-coefficient certificate |217419|691221|
| Individually aligned exceptional products |251420|903631|
| Original narrow group fallback |0|0|
| Carry mismatches / sampled GB10 BF16 mismatches |0/0|0/0|

The one unsupported original OUT input row is outside this deterministic dot
sample; its complete-row rejection is checked. The generated GPU fixture covers
unsupported-row execution. Captured whole-model failures remain in the evidence.
In particular, OUT's GB10 operator reference was evaluated on the identical
captured native gated-context input; this does not establish equality with the
original GB10 whole-model context or qualify that historical inference run.

The new standalone GPU fixture also passes controller C++ syntax checking using
declaration-only HIP stand-ins. That check does not compile device code or
validate wave semantics. Native compilation, the240 generated GPU configurations
and both q8192 operator comparisons remain unrun while the existing long-context
model run owns the device. No model-performance, correctness or release gate is
closed by this preparation. Current q8192 TTFT remains23134.0106ms for the c268
ordinary functional run; the10000ms boundary and retained4187.415605ms target
remain open.

[Source bindings, raw host sample, original reference provenance and syntax
receipt](../benchmarks/correctness/partial-wave-projection-local-20260921.json):
71383bytes, SHA256
`4c054a0a09cd793a3869fcd3cfe3ec809ecf87ebd97e41a79f66e6ca06799fe7`.

The earlier native plan freezes source501569f, a26-file standalone quoted-include closure
and an891-file source inventory. The inventory is explicitly not a preprocessed
compiler dependency list. Read-only baiying checks verify the six existing input,
weight and GB10-output tensors plus the unchanged execution guard; the native
script parses. The result parser rejects53 damaged records. All five actions
(build, safety, replay-build, QKV and OUT) reject the active owner before creating
controller output directories or calling any subprocess. Positive parser rows
are synthetic checks, not GPU measurements.

The source bundle remains on the controller, and the experimental checkout has
not been created on baiying. Compilation also waits for long-run cleanup because
the physical reserve is near the existing12-GiB coexistence threshold. Native
deadlines are240s for standalone compilation,300s for safety,330s for the replay
build and300s per captured operator, with a480s transport deadline. All use the
unchanged exclusive execution guard after cleanup.
[Frozen plan, script, parser and prepared checks](../benchmarks/correctness/partial-wave-projection-prepared-20260921.json):
154971bytes, SHA256
`36731bff0942ec81332a9c09293006cc1196ab0ced5fc584f0d48297d32046fa`.

The current plan uses source `5d63d5c` and the complete original q8192 capture.
Projection plan r2 and attention plan r3 share eight verified tensors plus a
manifest: 419,434,109 raw bytes, 339,667,106 compressed bytes. QKV and OUT now
require their original 8,192 reference rows and zero repeats. All nine staging,
build and GPU actions reject the active long owner before any subprocess or
output directory. All execution retains the existing exclusive host guard.

The updated parsers reject 55 projection and 28 attention faults, including
the old reference extent. Four native scripts parse and seven existing native
weight/table/guard/prerequisite files match. Parsing passes over stdin after
the first read-only command exceeds Windows' command-line limit; that failed
attempt is preserved. No input archive or source bundle was uploaded, no new
checkout exists and no new native build or GPU comparison ran.
[Complete-original-data native plans and preparation checks](../benchmarks/correctness/original-q8192-native-prepared-20260921.json).

A subsequent ASan/UBSan host audit covers the previously unsampled rejected
OUT input row, index558: all2,048 original outputs match GB10. Only its feature
2123 is outside the narrow domain, with biased exponent94. All112 rejected
QKV weight rows also match at16 original input positions each, giving1,792
additional endpoints. Their very small coefficients retain the original
arithmetic; no domain threshold changes.

The unchanged extracted projection replay also passes72 controller coordinate
cases with32 cooperating host threads per wave and scalar modeled matrix
instructions. It checks6,670 selected outputs,32,362 ordered carries and13,868
inactive outputs across partial tiles, candidate masks and rejected complete
rows. All30,336 broadcasts remain uniform and metadata/redzones are unchanged.
Deliberately changing the broadcast row or output position causes rejection.
This model does not execute the GPU unsupported-row subgroup or establish
physical WMMA behavior. Those remain part of the pending native safety suite.
[Original rejected-row endpoints and host coordinate evidence](../benchmarks/correctness/original-q8192-projection-followup-local-20260921.json).

Earlier local and preparation evidence now use public path views; embedded
artifact hashes still identify their unchanged originals at the recorded source
revision. [View identities](../benchmarks/correctness/public-evidence-home-path-views-20260922.json).
