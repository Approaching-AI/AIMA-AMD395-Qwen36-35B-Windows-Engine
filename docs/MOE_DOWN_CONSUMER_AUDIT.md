# MoE down consumer audit

Source `abb26a8970102bb098be13c1f4c461faa5deca6e` adds a default-off,
read-only observer under `QRT_QWEN36_MOE_DOWN_CONSUMER_AUDIT=1`.
Every original producer, selector, expert permutation, selected K16 replay
and production combine remains active. The observer supplies no inference
values and uses no reference output as an input.

## Consumer and numerical scope

Before selected down replay, the observer snapshots each token/channel's
eight weighted contribution intervals and original BF16 residual. Selected
intervals use the unchanged empirical PPB Cauchy envelope together with both
neighboring BF16 endpoints. Tiny, nonfinite or wider-than-nine-endpoint
ranges receive no certificate. Unselected contributions stay fixed.

The interval follows the production VT4 tree: pairs 0/4, 1/5, 2/6 and 3/7,
then ordered pair addition. It checks routed BF16 rounding, the already
completed gated shared BF16 contribution, and combined BF16 rounding.
A fixed combined endpoint fixes the complete **unrounded F32 residual**,
including the subsequent RMSNorm input. A fixed rounded residual alone
would not suffice. The original residual is captured before possible
output/input aliasing.

After every original replay, the observer checks each corrected selected
endpoint against its proposed range. After the original production combine,
it compares all reconstructed F32 residual bits with the actual output and
checks every proposed invariant against that corrected execution.
Private workspace is 335550536 bytes, with checked guards. The actual owner
handles asynchronous failures and drains queued work before freeing memory.

## Validation

Full local `make check` passes 54 Rust tests, 479 Python tests (two skips),
C/q16 ABI, clippy and hygiene. The host interval harness executes the actual
extracted production merge for 524288 interval corners; 229888 certified
outputs preserve its complete F32 result. It checks 16502 BF16 endpoints.
The actual asynchronous owner passes 15 injected transport failures and eight
invalid reports under address/undefined sanitizers. Three native fixture
calls were subsequently updated with explicit null observer arguments; the
native build and suites cover that update.

On baiying, the complete HIP safety build, 78 expert-order reports, six
original routed reports and production DLL build pass. These validate the
unchanged disabled arithmetic, not inference by themselves.

The same new MoE DLL is used in two fresh processes on
`D:\models\Qwen3.6-35B-A3B`, with original q8192 prompt IDs, all 512 GB10
output IDs, first logit tolerance 0.125, and actual streaming callbacks.
Whole provider `6e4908b` stays in mode 0; CK, FLA, CLI and all AOT assets
remain fixed. Dense PPB1000, MoE PPB512, expert-order replay and coarse FA OUT
remain enabled. Only the down-audit option differs.

| Audit | Load ms | TTFT ms | TPOT ms | GB10 IDs / callbacks | First logit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Off | 21291.0897 | 27925.6884 | 100.336913 | 512 / 512 | 10.375 |
| On | 21323.7382 | 28486.2800 | 100.807279 | 512 / 512 | 10.375 |

All 40 intended MoE calls execute the audit. Across 671088640 actual F32
outputs, production comparison, range containment and all proposed
invariants have **zero errors**. Each call's independent snapshot candidate
count matches its actual selected replay count. All 160 other dense and ten
coarse FA candidate/dispatch counts match between modes.

| Classification | Count |
| --- | ---: |
| Original selected down replays, all still executed | 158740988 |
| Valid selected contribution ranges | 156620130 |
| Candidate output cells | 143239419 |
| Candidate cells with valid complete intervals | 141121504 |
| Replays removable at fixed routed BF16 boundary | 114864539 (72.3597%) |
| Replays removable at fixed combined BF16 boundary | 126904922 (79.9446%) |

The enabled TTFT includes observation and is 560.5916 ms higher in this
single pair. No correction is actually omitted and no speedup is measured.
The envelope remains empirical: complete same-case containment does not
establish a universal hardware error bound.

## Decision and evidence

Keep the observer off. The complete consumer has enough removable work to
justify an actual default-off filter, retaining every uncertified replay.
The existing same-call shared-done event can provide the required device
dependency. Actual removal, complete GB10 continuation and net product time
must be measured separately. TTFT still exceeds 10000 ms; retained targets,
long-context, package and release gates remain unchanged and open.

- [Native provenance and numerical/owner checks](../benchmarks/correctness/moe-down-consumer-audit-native-20260917.json), 177377 bytes, SHA256 `10f326242f690b1135d1033b8328579c44a87daa4859482b7ad4623b6fd77d69`.
- [Same-DLL model pair, complete counters and command](../benchmarks/correctness/moe-down-consumer-audit-product-20260917.json), 431898 bytes, SHA256 `d047261ab5838f284cd8f5b8c5cdda58d9b2c1fbc862180c564775203a95fb12`.
