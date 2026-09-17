# Shared operands across independent GDN dots

This isolated component batches two or four independent scalar dots per lane
through WU, recurrent state and output. It shares packed operand reads while
retaining each dot's original K16 ordering, alignment, normalization and full
exceptional fallback. It preserves all BF16 intermediates, FP32 state FMAs,
64-token checkpoints and the complete U=V ownership boundary. Production
dispatch does not include these kernels.

`tiled_scalar_gdn_selftest.cpp` compares the retained scalar state8 stack with
1x2/256-thread and 2x2/128-thread candidates. Complete q8192 runs submit eight
1024-token WU/state/output segments, preserving state between them. This matches
the product's segment boundary, unlike earlier unsplit 8192-token state probes.
One warmup and three rotated completed host samples per arm include all three
stages. Allocation, reset, transfers and verification remain outside timing.
There are no intermediate host waits or device-phase timing claims.

Every attempt compares all W/U, residual, checkpoint, output and final state
bits with the original native control. Independent wide CPU samples check
normal/zero families. Guards, immutable inputs, the inactive U buffer and both
U ownership modes are checked. Safety includes lengths 1, 63, 64, 65, 129,
1023, 1024 and 1025 with zero, normal and subnormal families.

The host helper test checks 458752 ordered K16 states and 57600 raw outputs,
including mixed valid/invalid rows, forced original fallback, padded layouts
and longer dots. Native build, complete q8192 timing and any real-model token
qualification must be recorded separately. This source alone establishes no
speedup, model correctness, runtime adoption or release acceptance.

The initial native source `6b9b409` passes all 144 safety configurations and
complete q8192 comparisons. Separate-U medians are 78.2491 / 104.4255 /
206.7089 ms for control / 1x2 / 2x2; U=V medians are 72.5307 / 105.0905 /
206.1345 ms. Both candidates regress. Their state kernels declare 164/376
private bytes and register spills, versus zero private bytes in the control.
These static declarations motivate revising live ranges, without establishing
measured occupancy or a complete explanation of the slowdown.

The next component revision keeps only one row of product groups live and
uses the existing exact FP32 carry representation on its eligible domain.
Any failed operand/endpoint classification restarts the complete tile with
original arithmetic, including every early group. It processes exceptional
dots individually after retiring fast products. The host audit explicitly
checks late overflow restarts and overwrites every speculative carry trace.
This revision requires a fresh native build, safety check and complete q8192
comparison before any adoption decision.
