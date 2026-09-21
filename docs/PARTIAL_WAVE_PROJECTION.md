# Partial-coefficient wave-owned projection

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
Both operators repeat the first1023 captured q7169 input rows to reach8192.

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
70679bytes, SHA256
`6d3719adc83d720c07d7f7885895726ba0990c9750ecc70a0cda1ff4c50b5365`.

The native plan freezes source501569f, a26-file standalone quoted-include closure
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
154247bytes, SHA256
`32e426f4e40599a91a5174d9155a088e1c1c6bed50e33788a0806277f5b5f2b8`.
