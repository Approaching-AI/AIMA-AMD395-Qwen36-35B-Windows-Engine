# Cold prefill KV reservation

The chunk coordinator now reserves the known final prompt capacity once for
each of the ten full-attention layers after the first real 8192-token chunk.
Subsequent chunks append only their new K and V bytes. Previously, every
boundary allocated a larger owner and copied all preceding history again.
The numerical kernels, original prompt inputs and attention key counts are
unchanged. A q8192 request does not enter this coordinator.

The motivating observation is the original 256k prefix case on
whole provider `32a1b96af4126d44c7e80208dc76ea31160a010e`. At
2026-09-19T05:25:43Z on baiying, 17 of 32 owner chunks had completed. The AMD
adapter counter reported 105855643648 committed bytes, including 6940114944
shared bytes. This is an adapter-wide counter; the process-specific counter
timed out. The run later stopped under the unchanged 8 GiB physical-memory
guard after 20 complete chunks (163840 inputs). Its minimum available physical
memory was 8201998336 bytes and native wall was 8219702.075 ms. No output token
was produced. All final host and process-cleanup checks pass. This establishes
the observed stop condition, not a leak or token-mismatch diagnosis. See the
[completed memory-guard record](../benchmarks/correctness/rope-single-round-prefix256k-memory-limit-20260919.json).

While the cold transaction holds the session mutex, V begins after the
reserved K capacity. The committed byte counts and history length still cover
only the consumed inputs. A failed allocation, initial copy or old-owner free
retains the original owner. Failed appends can alter only the uncommitted
region; they do not advance either byte count or history counter. The
coordinator discards a failed transaction. The last append fills the reserved
capacity and clears the temporary reservation before publishing the ordinary
compact resident layout. Decode tails remain separate.

Four local tests pass, including ASan/UBSan execution of the actual reservation,
append and coordinator functions. They cover original K/V bytes, allocation
and copy failures, old-owner release failure, invalid capacity/stride, appends
with allocation disabled, the 1024-token final tail, partial reservation
cleanup, callback publication, and the existing suffix ring/state layouts.
Source `b35ae93a2ea6c123801ff71dc6108fd88af32d16` now builds on baiying in
95856.795 ms, with all 93 inputs and the retained compiler flags verified.
The 13508608-byte DLL has SHA256
`f4c7742fb510c81b4617c6f3ebe520cc10eaecc3d6dcb496ed49e4f58e605390`.
The [native build record](../benchmarks/correctness/prefill-kv-reservation-native-build-20260919.json)
does not load a model.

The compiled candidate now passes the original cold32k/out512 request on
baiying: all 512 IDs and callbacks match, first logit is 24.75 with zero error,
all four chunks complete, and the exact 671088640-byte reservation marker is
present. Source and host/cleanup checks pass. Load is 21280.1456 ms, TTFT
257781.5184 ms and TPOT 217.390692 ms. See the
[original cold32k boundary](../benchmarks/correctness/prefill-kv-reservation-cold32k-out512-20260919.json).

The paired original whole32a run also passes. Minimum sampled system available
physical memory is 21918986240 bytes for the control and 21916213248 bytes for
the candidate, a difference of -2772992 bytes. Peak process private bytes
differ by +1409024 bytes. These single runs do not establish a memory
improvement or resolve the earlier 256k stop. The
[sampled memory comparison](../benchmarks/correctness/prefill-kv-reservation-cold32k-memory-20260919.json)
includes startup and cleanup and is subject to other system activity.

The same DLL also passes original q8192/out512, all callbacks and first
logit 10.375 with zero error. Load is 21298.2486 ms, TTFT 23422.0822 ms and
TPOT 101.650893 ms. See the
[q8192 regression](../benchmarks/correctness/prefill-kv-reservation-native-q8192-20260919.json).
The original 256k owner/suffix case is being rerun with the unchanged memory
guards and 28800-second process deadline. Full context, retained performance
and release acceptance remain open.
