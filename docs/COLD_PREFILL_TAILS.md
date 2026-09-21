# Cold prefill final chunks

Source `c2683cd5cb55478099f1adffe47b98e5cef79fba` builds on baiying and
passes the original q8193 native MTP boundary. The final prompt input is
`63` at position 8192; it produces `220 / 9.75`, followed by all 31 original
continuation tokens. The ordinary q8192 control and all four native short
controls also pass. Native MTP remains opt-in. The full 256K continuation,
native retirement cases, final package and performance gates remain open.

[Actual build, token, callback and server evidence](../benchmarks/correctness/single-tail-q1-native-20260921.json)
binds the command files, model, source revisions, binary hashes, original
references and host cleanup. Every model run uses baiying and
`D:\models\Qwen3.6-35B-A3B`.

## Runtime behavior

The cold coordinator retains a true final extent from 1 through 8192 inputs.
Every earlier chunk contains exactly 8192 original tokens. A convolution ring
update with fewer than four new inputs replaces only the new slots and
preserves preceding prefix slots in both FP32 and BF16 layouts.

With native MTP and cold chunking enabled, the C entry point delegates the
complete prompt to this owner. This preserves shifted-input and checkpoint
provenance across the former 8193–9216 cold-suffix shortcut. The public prefix
suffix interface and the drafter's 262144-token admission limit are unchanged.

A final one-input chunk uses the full resident q1 export with the original
prefix arithmetic. Before execution, empty unpooled KV tails return to the
standard 1536-token q1 layout. A private request consumes the actual prompt
input, completes all layers and fences the final BF16 normalized row. That
row and the actual sample enter the private MTP seed; the cold coordinator
promotes KV and recurrent counters, then emits its one caller callback.
The private request has no callback. Its cold-input scope preserves nested
retired-target ownership without selecting retired-target arithmetic.
Batch descriptor metrics are empty for this q1 tail.

Numeric u32/i32/u64 environment caches freeze configured values or their
absence/invalidity, then resolve each caller's fallback on every invocation.
Dynamic probes and the original parsing contracts are preserved.

## Actual Windows results

The whole DLL, normal CLI and prefix probe verify 173 source inputs and build
in 119915.977 ms. Their compilation closures contain 167, 8 and 10 inputs,
with overlap. Whole DLL SHA256 starts `84e3cedc`; CLI SHA256 starts `813dd56d`.
All compiler flags match the declared control.

| Mode and original case | Outputs and callbacks | First token / logit | Native continuation commits | Load ms | TTFT ms |
| --- | ---: | --- | ---: | ---: | ---: |
| Ordinary q8192/out512 | 512 / 512 | 144 / 10.375 | disabled | 21496.2425 | 23134.0106 |
| Native q7169/out32 | 32 / 32 | 82 / 9.25 | 31 | 21466.160699 | 25633.5616 |
| Native q8191/out32 | 32 / 32 | 168589 / 11.375 | 31 | 21452.8289 | 29500.1411 |
| Native q8192/out512 | 512 / 512 | 144 / 10.375 | 511 | 21460.0423 | 24529.5116 |
| Native q8193/out32 | 32 / 32 | 220 / 9.75 | 31 | 21749.386 | 25665.424 |

All original prompt IDs, output IDs, first logits and actual callback order
match. The four native cases contain no ordinary generated-token commits;
q8193 separately records its one cold q1 input and completed MTP handoff.
Every run completes with passing host checks and no remaining process.
These are functional controls. The ordinary TTFT still exceeds 10000 ms;
its TPOT is 101.444817 ms. Retained performance targets and medians do not change.

The same-source Windows server also builds, verifying 25 source/build inputs,
including nine C and ten Rust inputs. All 54 existing Rust tests and formatting
pass. Native wall is 49708.972 ms; executable SHA256 starts `57a40483` and its
PE stack reserve is 268435456 bytes. This build does not qualify HTTP inference.

The first whole build and ordinary q8192 attempt launch no native process
because the output disk reserve check fails. Moving the source and evidence
to P preserves the 10 GiB guard. All 2349 worktree files, 249383232 bytes and
prior failure records are verified before the duplicate D copy is removed.
The ordinary retry retains its original working directory, dependencies and
numerical environment; only evidence paths move. Both failures are retained.

## Diagnosis and local validation

The preceding a7 source rejects the final q8193 chunk at layer 39. A scoped
terminal predicate repair at 99fb67f admits it, but outputs token 64 instead
of 220. Both the ordinary and observed native failures are retained in the
[diagnosis and local repair report](../benchmarks/correctness/single-tail-q1-local-20260921.json).
Of 80 observed tail norms, only layer 0 input is bit-exact with the original
GB10 transaction. Layer 0 postattention differs in 974/2048 BF16 values,
maximum absolute error 0.03125. This did not isolate the LM head as the cause.
Carrier/final-norm text markers were filtered and are not claimed as compared.

The first observer fails before output because a cached fallback position of
8191 is reused for the one-input tail. An explicit local position permits
the complete norm capture; the cache repair removes this caller-default bug.
The final bridge passes 31 ASan/UBSan cases, including actual input, norm and
sample handoff, scope restoration, stale metadata, exceptions and copy failure.
Numeric parser validation covers 752 checks and three actual old-parser
negative controls. Coordinator, C bridge and native retirement regressions pass.
Initial fixture extraction failures and corrected logs remain in the report.

The earlier [tail-layout tests](../benchmarks/correctness/cold-prefill-tail-local-20260921.json)
cover 26 ordinary and 13 MTP prompt lengths, 144 ring shapes, cancellation,
checkpoint failure and publication cleanup. The [terminal predicate checks](../benchmarks/correctness/terminal-cold-tail-local-20260921.json)
cover 204 predicate and 66912 provider-selection cases. New observer validation
separates cold input from generated commits and rejects 23 short-case and
72 retirement-boundary faults. Synthetic checks establish observer behavior;
actual original-model runs establish the token boundary.

## Remaining validation

The previous complete 256K owner run matches the first 124 suffix outputs,
then emits 8984 instead of 4980. The qualified original reference supplies
input 471 at position 263291. The new output-only capture targets this point
and position 263290 on c268. Its comparator first checks the actual generated
history; an old native control matches all 628 compared surfaces at position
263168 against the new reference. That control does not qualify later steps.

An [offline arithmetic replay](../benchmarks/correctness/prefix256-step124-arithmetic-host-20260921.json)
uses independently qualified GB10 operands at263168,263290 and263291 with the
actual c268 headers and ASan/UBSan. Both current RoPE helpers match60 original
Q/K surfaces,138240 BF16 values. All30 recurrent layers at all three positions
match47185920 FP32 state values and368640 BF16 core outputs bit-for-bit, using
each original transaction's operator selection. The older double-round RoPE
control also matches these particular rows. These component results do not
establish the Windows operands, accumulated state or cause of the divergence.

The [native recurrence replay preparation](../benchmarks/correctness/step124-gdn-native-prepared-20260921.json)
binds those90 original rows to the existing GPU probe. All eight compilation
inputs are identical to c268; the actual probe, seven preceding build/control
records and35 parameter/arithmetic tables verify on baiying. Both scripts
parse,20 damaged-report controls reject and active-owner admission refuses
to launch. The planned180 layout comparisons remain unrun. Reference operands
are inputs only to this isolated component comparison.

The original native q262140/q262142/q262143 retirement cases remain unrun.
The [current package preparation](../benchmarks/correctness/single-tail-package-prepared-20260921.json)
now binds the actual c268 whole DLL, CLI and static-C-core server together
with the declared CK, FLA and MoE dependencies. Read-only baiying checks verify
all six files, their source/build records and the original q8192/out512 control.
All535 portable options and34 paths match that control after the three declared
profiling/chunking differences. The planned inventory contains269 runtime
artifacts and285 release files; no new archive has been created or relocated.

The preserved final-artifact workload contains45 protocol,15 prefix and55
control-plane requests,13 cold cases with1856 outputs and the full3600-second
soak. Preparation checks reject five profile faults,12 damaged historical
records and20 attempts with missing actual archive results. Stage/test admission
also waits for the existing long job's completed cleanup. Five native scripts
parse successfully. The first read-only binding observer's absent optional
MoE-size field failure is retained; its corrected reader still requires the
original SHA256. These checks qualify preparation only. Actual final-file
inference, archive verification and release acceptance remain open.
