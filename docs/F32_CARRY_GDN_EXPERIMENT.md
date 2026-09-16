# Compact carry across GDN, 2026-09-16

Source `f49c5fafde972cbce6faf8c6c3a4e16db660d862` substitutes the existing
strict FP32 K16 carry in all six scalar dots used by W/U, recurrent state
and output. The kernel bodies, packed BF16 inputs, shared layouts, rounding,
checkpoints, final FMAs and U=V ownership remain unchanged. Invalid rows,
tiny groups and unsupported endpoints restart the complete original dot.
The new component kernels have no product option or dispatcher integration.

Full local checks pass 54 Rust and 472 Python tests (two skipped), C/q16
ABI, clippy and public hygiene. Sanitized host tests compare 7680 accepted
dots against independent integer arithmetic and verify that 4608 declines
leave the destination untouched. They cover widths 64/128, four/eight
columns, signed zero, cancellation, exponent boundaries, invalid late
operands and endpoint overflow. Native build and safety on `baiying` finish
in 3633.584 and 1344.148 ms, with all guards passing.

All 72 generated native safety cases pass, including every W/U value,
checkpoint, residual, raw output and FP32 final state. The q8192 fixture
checks the same boundaries after one warmup and three rotated samples per
variant and alias mode. Independent CPU columns cover the complete W/U
and recurrent calculation, with 172032 dots at q8192.

| Complete q8192 W/U, state and output, ms | Original | Compact carry |
| --- | ---: | ---: |
| Separate U/V buffers | 84.6648 | 74.9052 |
| U=V, as in production | 81.0788 | 80.6714 |

Samples overlap and vary. The product ownership pattern shows only a
0.4074 ms difference. Keep this route isolated: these results do not
establish a material real-model gain. Static W/U, state and output VGPRs
change from 78/113/80 to 78/112/79; LDS stays 10816/42820/28736 bytes and
private allocation stays zero. These are compiler declarations.

Timing covers all shared preparation and complete fallback. Phase events
resolve after final completion without intermediate host synchronization;
their overhead remains in the joint host clock. Generated q8192 recurrence
uses one full segment and differs from the product's 1024-row partitions.
There is no model load, new GB10 token result or retained performance claim.
The real q8192 TTFT and release gates remain open.

Full evidence is
[`f32-carry-gdn-native-components-20260916.json`](../benchmarks/correctness/f32-carry-gdn-native-components-20260916.json),
233161 bytes, SHA256
`5113665c2f9f74525cf4f3caba600369a6cb7d6a076a976b0da927f568de0d3c`.
