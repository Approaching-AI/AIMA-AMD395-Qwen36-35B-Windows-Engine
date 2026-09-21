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
