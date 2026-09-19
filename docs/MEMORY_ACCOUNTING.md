# Runtime allocation and Windows memory observations

Source `04a9dfda53dc7ccc746ede2092810fd39b3f2ea0` corrects the MoE scratch
report by adding allocated optional workspaces. The fixed-capacity ABI base
stays unchanged. Optional router outputs, norms, compaction/order metadata,
staged operands, replay flags and shared FP32 projections now contribute only
when their pointers exist. Registered weight metadata was already counted and
is not added twice. Driver/library allocations and transient per-call owners
remain outside this provider capacity report.

The qualified ordered-storage cold32k run on MoE `9235750` reports 160 staged
replay calls using 1207959552 weight bytes and 75497472 input bytes. That
source's scratch getter omits both owners: at least 1283457024 allocated bytes
are absent from its reported value. All raw model matrices in that run are
borrowed views. The compact-alias marker describes a final-layer interpretation;
its zero value on earlier layers does not imply duplicated model weights.

Host sanitizers compare the actual allocation requests with the new helper at
every successful allocation. They cover 49152 option/capacity combinations,
5328 injected allocation failures and 890118 allocation observations. Partial
failure counts only the successful prefix; flags with null owners add zero.
All 11 MoE regressions pass. The initial host fixture failed on unused extracted
constants under `-Werror`; its corrected declarations pass without changing
runtime source. Windows compilation and use of this repair remain pending.

The ongoing original full256k experiment still uses whole `b3af8b6`, CK `3701495`
and MoE `9235750`. A read-only Windows snapshot at 2026-09-19T15:56:08Z identifies
`qrt-product` PID 12488, started at 13:47:58Z, and reports:

| Counter | Bytes |
| --- | ---: |
| Process GPU DedicatedUsage | 98869354496 |
| Process GPU SharedUsage | 2192842752 |
| Process GPU TotalCommitted | 101062197248 |
| Host process private bytes | 27496542208 |
| Host process working set | 972066816 |

Dedicated GPU memory is reserved for GPU use; shared GPU memory uses system
memory. Per-process usage can include allocations shared with other processes,
so summing process values can double-count them. These are distinct from
provider capacity and host private-memory accounting.
[Microsoft's explanation](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/)
describes these distinctions. Separate WMI class reads are not an atomic sample.
The first process-counter timeout is preserved alongside the successful
filtered retry.

The first cold chunk's allocation-pool marker is printed after releasing its
pool. Its 2756558868 cached bytes cannot establish the current suffix pool's
live size. A separate read-only CK workspace snapshot belongs to this ordered
run; the earlier whole334 snapshot belongs to a different experiment.

The [memory accounting record](../benchmarks/correctness/runtime-memory-accounting-20260920.json)
pins both native snapshots, their observers, actual source and process identity,
the qualified prior model evidence, and all local repair checks. These readings
do not establish a leak, memory savings, full256k correctness or performance.
The original owner and both suffix continuations must still complete.
