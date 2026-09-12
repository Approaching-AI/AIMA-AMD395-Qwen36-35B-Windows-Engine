These three BF16 input/output rows were observed in the original GB10
32768-token owner prefill, layer11, positions18553–18555. The manifest binds
their transaction, model index, weight slice and unchanged canonical oracle.
Model weights are read from the user's local model and are not bundled here.

The replay calls the actual native hipBLASLt producer, L2 bounds, candidate
predicate and K16 correction. It repeats the three inputs across8192 rows to
preserve the producer shape and compares only the three original positions.
The1000ppb negative control must reproduce a missed BF16 output;10000ppb must
match every observed output. This is an operator regression, not an inference
or performance acceptance test.

Run locally on baiying from a clean checkout, with no other GPU job:

```powershell
scripts/baiying_build_whole_provider.ps1 -AttentionAdmissionReplay -HawkeyeReplayLanes 4 -CompileTimeoutSeconds 180 -OutDir build/attention-admission-replay
scripts/baiying_attention_admission_replay.ps1 -ProviderBuildDir build/attention-admission-replay
```

The build uses a180-second compiler deadline. The replay has a60-second host
guard, verifies model and fixture hashes, checks repeated input immutability,
and saves the command, commit, results and host checks under its output path.
