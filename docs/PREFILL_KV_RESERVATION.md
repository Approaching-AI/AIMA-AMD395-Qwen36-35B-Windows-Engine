# Cold prefill KV reservation

The chunk coordinator now reserves the known final prompt capacity once for
each of the ten full-attention layers after the first real 8192-token chunk.
Subsequent chunks append only their new K and V bytes. Previously, every
boundary allocated a larger owner and copied all preceding history again.
The numerical kernels, original prompt inputs and attention key counts are
unchanged. A q8192 request does not enter this coordinator.

The motivating observation is the still-running original 256k prefix case on
whole provider `32a1b96af4126d44c7e80208dc76ea31160a010e`. At
2026-09-19T05:25:43Z on baiying, 17 of 32 owner chunks had completed. The AMD
adapter counter reported 105855643648 committed bytes, including 6940114944
shared bytes. This is an adapter-wide counter; the process-specific counter
timed out. It establishes memory pressure, not a leak diagnosis. The new
reservation has not yet been built or measured on baiying and does not inherit
that run's numerical results or qualify an archive.

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
Native build, original GB10 token comparisons and a measured memory comparison
remain required. No performance improvement or release acceptance is claimed.
