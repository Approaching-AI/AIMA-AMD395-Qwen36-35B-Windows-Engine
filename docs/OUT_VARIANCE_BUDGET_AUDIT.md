# Adaptive OUT variance-budget audit

Default-off source `c3b4ad5` adds a read-only prospective replay scheduler after
original OUT completion. It regenerates the algorithm0 projection in private
storage, transports the existing PPB/midpoint envelope through residual
rounding, then refines the remaining variance interval in at most 18 rounds.
Priority comes only from squared-magnitude interval spans computed from
producer/residual endpoints. Each round hypothetically replays the largest
remaining contributions before checking every normalized BF16 output.

The certificate uses the original FMA reduction tree and enumerates every
actual FP32 variance in a range of at most 4,096 ULPs through the original
reciprocal-square-root function. It assumes no monotonicity of the hardware
function or its correction table. The final round restores every remaining
candidate. Completed original outputs stand in for the hypothetical replay
passes, and never determine priority. Actual model inputs remain unchanged.
Classification is conditional on the existing empirical projection envelope;
observed containment and endpoint failures are reported independently.

Local checks pass 54 Rust tests, clippy, 476 Python tests (two skipped), C/q16
ABI and public hygiene. The actual owner passes ASAN/UBSAN with 24 injected
transport failures, 130,560 interval edges and 74,576 enumerated BF16 values.
Those host tests do not execute the GPU classifier. The bounded baiying native
build completes in 93,881.497 ms. The DLL is 13,298,176 bytes with SHA256
`7e92e62faeccc3c3bf26ece7a6e0f2c7086e23d6f75f21e8ae362e331d60ef81`.
The audit kernel declares 133 VGPRs, 4,188 LDS bytes, zero private bytes and
wave32. Static resources do not establish occupancy or audit speed.

Native evidence: [source, local validation and Windows build](../benchmarks/correctness/out-variance-budget-audit-native-20260917.json),
22,892 bytes, SHA256
`849b1ca59da3e37bbf3ae6ccb98d3f20d98bd22edc881a8753093e8a660e470a`.

The same-DLL real q8192/out512 pair preserves every original GB10 output ID,
actual callback and first logit 10.375. All 160 original dense and ten coarse
OUT per-call correction counts remain identical. Only the audit option differs
between the two environments; unrelated experiments are explicitly disabled.

| Audit | Load ms | TTFT ms | TPOT ms | Audit calls |
| --- | ---: | ---: | ---: | ---: |
| Off | 21454.0273 | 27818.6548 | 101.096413 | 0 |
| On | 21334.0045 | 29193.8827 | 100.166127 | 40 |

The single observed 1,375.2279 ms TTFT increase is diagnostic overhead. It
does not measure an implemented candidate scheduler or saved execution.

| Classification | Linear, 30 calls | Regenerated FA algorithm0, 10 calls |
| --- | ---: | ---: |
| Original candidates | 58,489,767 | 106,478,115 |
| Prospective residual-ambiguous first replay | 13,455,017 | 51,291,030 |
| Prospective additional variance replay | 15,843,176 | 51,965,178 |
| Pending candidates certified removable | 29,191,574 | 3,221,907 |
| Prospective removed fraction | 49.9089% | 3.0259% |
| Average rounds per row | 6.9698 | 9.7956 |
| Complete pending-replay fallback rows | 225 | 24 |
| Final certified rows | 245,760 | 81,920 |

Every recorded projection/variance containment check and residual/normalized
endpoint comparison passes, with zero final uncertified rows. The maximum
initial variance spans are 87,572 and 45,903 ULPs; those wide ranges trigger
further refinement rather than an unsupported rsqrt certificate. All original
model inputs remain unchanged. FA classification regenerates algorithm0 and
does not describe the retained coarse FA producer's candidates.

The linear algorithm0 shape subsequently received an [actual default-off replay experiment](OUT_VARIANCE_REPLAY_EXPERIMENT.md). Its full 30-layer run reproduces these aggregate counts and passes all GB10 boundaries, but the single 99.3839 ms TTFT difference does not establish a repeatable gain.
The conditional 49.91% candidate reduction warrants testing whether real
saved work exceeds scheduling overhead. Keep the observer default-off and
retain the existing product stack. That follow-up supplies the actual candidate replay and complete GB10 comparison;
this observer alone does not measure either. No performance, prefix,
long-context, package or release acceptance follows. The original 10,000 ms
TTFT gate, 4,187.415605 ms retained target and 30,000 ms loading limit remain
unchanged.

Product evidence: [same-DLL runs and complete observer counters](../benchmarks/correctness/out-variance-budget-audit-product-20260917.json),
436,942 bytes, SHA256
`c975f9b91732b1d4a44bf9a4afb618af7cea1bebfa273ae531033b5763d29866`.
