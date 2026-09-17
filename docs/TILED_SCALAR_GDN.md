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
