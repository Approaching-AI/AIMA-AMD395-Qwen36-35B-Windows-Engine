# Separate GDN state calculation and replay

This isolated candidate moves the narrow-domain state calculation and the
retained integer fallback into separate kernels. The intended benefit is to
reduce the resources needed by the common path. No register, occupancy or
speed improvement has been measured. Runtime dispatch is unchanged.

Each CTA owns eight complete value columns through a 1024-token segment.
The fast kernel checks the actual W, rounded H, K and residual operands
against the existing signed-zero / BF16 exponent95:159 proof before each
dot. It preserves every ordered K16 group, original truncation, BF16 endpoint
and final FMA. A rejected CTA exits uniformly without publishing its final
recurrent state. It may have written partial H and residual outputs.

The second kernel reads one completion receipt per CTA. It skips successful
CTAs and replays every chunk of each rejected CTA from the unchanged original
state. The replay overwrites all H and residual cells owned by that CTA before
output consumers launch. Columns do not depend on one another. The replay
body is copied from the retained shared-arena kernel and checked byte for
byte by the host test, apart from its receipt argument and initial skip.
The design needs 2048 receipt bytes per segment. The fixture retains all
segment receipts to inspect admission after timing, without extra host
synchronization inside the measured sequence.

An audit of original GB10 q7169 layer0 operands finds that W, K or H excludes
1126 of 3584 CTAs across the seven full 1024-token segments. Adding scaled
residual intervals increases that count to1129, about31.5%. The remaining
2455 CTAs satisfy the original-data operand checks. Across all eight segments,
including the one-token tail, the audit predicts2901 fast and1195 replayed
CTAs. These are predictions; actual native receipts remain unobserved.
This measurement covers one captured layer and does not establish
model performance. The prior combined narrow-domain kernel did not improve
the production U=V component comparison; separating its two paths is a new
experiment, not a retained result.

The [residual auditor](../tools/audit_gdn_replay_domain.cpp) uses the inclusive
FP32 rounding interval implied by each original BF16 Vnew value. Multiplying
its endpoints by the original SM121 decay bounds the scaled residual without
inventing an unrounded reference value. It includes both gradual underflow
and possible flushed zeros. Individual uncertain cells cannot admit or reject
a CTA; the complete W/K/H and residual audit leaves no unresolved segment
classification in this capture. All228480 rounding-bin probes and the complete
capture scan pass under ASan/UBSan. The wider64:190 operand comparison is
diagnostic and does not establish a corresponding carry domain.
[Inputs, source, commands and complete interval results](../benchmarks/correctness/separate-state-gdn-residual-domain-20260921.json)
remain separate from native correctness or speed qualification.

Two [follow-up admission audits](../benchmarks/correctness/separate-state-gdn-admission-followup-20260921.json)
keep the existing product-exponent window. Aggregate operand ranges admit
2912 of4096 original CTAs; individual K16 group checks admit3038. These add
11 and137 CTAs over the existing2901 prediction. Both auditors pass their
completed ASan/UBSan scans. Those admission audits alone do not qualify the
arithmetic or measure native receipts or speed. The checked product-domain
implementation below adds independent integer validation and a selective
consumer certificate; the runtime remains unchanged.

The host test runs the candidate bodies with64-thread transport and an
independent wide integer reference. Seven cases cover partial chunks,
late W/K rejection, excluded initial state, strong decay, subnormal operands,
disjoint columns, completion receipts and guarded output ownership. It checks
that rejected CTAs leave recurrent state unchanged before replay. All18 fast
and10 replayed CTAs match, including6 partial-output replays. ASan and UBSan
pass. Both deliberately broken variants are detected: replaying already
completed CTAs, and publishing a success receipt before completion.

The prepared native fixture uses256 threads and compares the complete
score/WU/state/output sequence, both U ownership modes and224 generated
configurations. It also compares original q7169 inputs and all captured
outputs; its q8192 extension repeats1024 rows and remains a component test.
Native compilation, component timing, a real q8192 product run and any
performance retention remain pending. The qualified model median stays
23353.80795 ms; neither the below10-second gate nor the4187.415605 ms target
changes.

Run the local ownership test with
`python3.12 -m unittest tests.test_fla_separate_state`.

[Source hashes, host commands, negative controls, original operand audit and
the unrun native plan](../benchmarks/correctness/separate-state-gdn-local-20260921.json)
are recorded separately from any native inference or performance acceptance.

## Checked hybrid retry

The new isolated `hybrid_state_replay.h` inserts a retry kernel between fast
calculation and retained replay. It skips every successful fast CTA. For a
W/H dot outside the original narrow domain, an absolute bound must prove both
BF16 residual outputs identical; otherwise the entire CTA remains for replay.
In-range W/H dots retain the original narrow calculation. K/V dots use the
checked product-domain carry without approximating the FP32 recurrent update.
Only a complete successful segment publishes state and receipt2. Failed
retries preserve the original state, and retained replay overwrites every
partially written H and residual output before consumers launch.

`sm121_product_f32_carry.h` admits an ordered K16 group when its largest normal
product exponent is in[-64,64]. It also admits all-zero finite products.
Nonfinite operands and unsupported carry ranges decline without publishing.
A product involving a subnormal operand is omitted only when its conservative
exponent is at least27 below the normal maximum. Its aligned magnitude is
then strictly below1 in the original integer algorithm as well. Tiny normal
products flushed by scalar multiplication likewise cannot reach an integer
alignment quantum. This keeps the existing carry floor and modulo reduction;
the wider[-101,64] group window is not implemented.

The [local hybrid evidence](../benchmarks/correctness/hybrid-state-gdn-local-20260921.json)
records196052 admitted dots,147972 declined dots and298452 exact ordered carry
boundaries, including simulated FTZ. Every normal exponent pair and every
BF16 encoding is covered, with signs, cancellation and late rejection. Two
deliberately weakened guards produce detectable numerical differences.
The actual kernel bodies pass nine guarded host cases:20 fast CTAs,8 retry
successes and8 retained replays. Both false completion and partial state
publication are detected by independent state/ownership checks. ASan, UBSan
and float-conversion overflow checks pass.

The original q7169 layer0 audit certifies both BF16 consumers for4533406 of
4539662 W/H dots outside the narrow operand range. Combining those certificates
with conservative K16 product intervals predicts3463 of4096 segment CTAs
admitted, including2953 of3584 full-segment CTAs. W/H certificates alone predict
3070. Widening the product window adds only35 CTAs and is not part of this
candidate. Native receipts may differ from conservative interval predictions.

A separate original-input sample computes3670528 W/H dots with the independent
wide integer implementation; every captured GB10 BF16 residual matches. It
also verifies3103453 narrow carry results and both endpoints of566274 certified
residuals. Among462848 sampled K/V dots,427924 checked carry results match the
wide integer result, including91652 excluded by the old operand rule. The
remaining34924 decline without changing output. This reconstructs the actual
scaled residual from the original dot rather than substituting the captured
rounded Vnew for its unrounded value.

`hybrid_state_gdn_selftest.cpp` prepares three complete-chain controls:
retained, fast/replay, and fast/retry/replay. It preserves the original capture,
all output/intermediate comparisons, both U owners and independent CPU samples.
Its336 generated configurations and rotated q7169/q8192 component comparisons
are not yet compiled or executed on gfx1151. Native resources and timing will
decide whether the extra retry work is worthwhile. None of these local results
qualifies model inference, changes retained performance, or permits release.

The [frozen native plan](../benchmarks/correctness/hybrid-state-gdn-prepared-20260921.json)
binds428 inputs to sourcee14f64b52739a8f9aaa2c60b7f749dd3cf5f2b00. The command
file parses on Windows; dispatch and resource collection both stop before
any remote/native call while their prerequisites are absent. Compilation is
bounded to180seconds, component execution to300, transport to420 and each Git
child to30. The resource collector will inspect retained, fast, retry and
replay kernels in the same executable after validating the actual build and
all source hashes. Its existing retained-ELF parser check passes. No candidate
resource count is available before compilation.

## Default-off provider submission

`QRT_FLA_GDN_STATE_REPLAY` now exposes mode1 (fast/replay) and mode2
(fast/hybrid retry/replay) in the provider. Mode0 remains the default. Selection
requires the existing paired scalar configuration, no coarse or fused-state
override, at most1024 actual tokens and no internal checkpoint export. A
checkpoint-bearing segment retains its original state-capture implementation.
This includes ordinary and seeded segments; no reference activation enters
the candidate.

The batched route reuses the first2048 bytes of its otherwise unused legacy
temporary state for512 completion receipts. Every pipeline slot already owns
a separate allocation. No device allocation or storage-reporting change is
needed. Fast initialization precedes retry/replay on the caller's stream;
every launch error stops the remaining chain, and the existing segment owner
retains responsibility for completion and error cleanup. The wrapper checks
receipt overlap with all operands, outputs, FP32 state and exponential table.

The [local submission evidence](../benchmarks/correctness/state-replay-provider-local-20260921.json)
passes104 wrapper cases and320 actual-provider caller cases under ASan/UBSan,
including launch failures, receipt aliases, partial chunks, disabled mode and
checkpoint fallback. Two broken chains are detected. Seven existing state,
checkpoint, pipeline and completion tests also pass. An initial test-only
extraction error stopped compilation at the default aggregate argument; the
corrected extractor removes only that unused default. The failed log remains
preserved. No production arithmetic changed during that correction.

`state_replay_provider_selftest.cpp` uses the actual submission wrapper in
the three-variant complete-chain fixture. The arithmetic kernels remain
identical to the previously checked hybrid source. Windows compilation,
all336 generated configurations, original q7169 operands, q8192 component
shape, provider probes and same-DLL real q8192/out512 comparisons remain
pending. This default-off integration adds no new numerical, performance
or release qualification. The separately frozen e14f64b fixture is still
reproducible and need not be mistaken for a build of this provider.

The [provider component plan](../benchmarks/correctness/state-replay-provider-prepared-20260921.json)
freezes430 inputs at6003bcdee4be820c8901e11139ae64dabde32d4f. It includes the
actual cooperative translation unit once with `-include`, avoiding duplicate
header-defined HIP kernels, and uses the provider's `-O2` optimization. Both
the command and builder parse on Windows. Five dispatch actions and resource
collection stop with zero subprocess/network calls while prerequisites are
missing. Synthetic report-format checks cover all six variant/owner rows.
Native build/component/transport bounds remain180/300/420 seconds, with30-second
Git children. No component has executed and no resource or speed result exists.
