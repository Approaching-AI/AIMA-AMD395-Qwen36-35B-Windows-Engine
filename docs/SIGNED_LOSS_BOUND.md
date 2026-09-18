# Directed canonical loss experiment

This isolated experiment keeps the coarse producer's conditional native
error coefficient at `2^-19`. It uses each operand's sign and nonzero masks to
charge positive product truncation only downward and negative truncation only
upward. A complete C64 prefix barrier, including the previous uncertainty,
absolute products and original canonical loss bound, permits one-sided carry
and normalization charges only when every internal sign is established.
No observed original or golden carry enters the bound.

The UBSan host check covers all 65,536 BF16 encodings, all 65,536 support masks,
16,384 generated sequences and 393,216 original C64 boundaries. Three synthetic
producer modes use host RN products or positive/negative perturbations within
the unchanged native premise. Every checked interval contains the original
accumulator, every old certificate remains valid, and no new certificate is
false. These checks do not execute native WMMA or prove its hardware premise.

The input-independent 65,536-cell sample of the captured q8192 OUT geometry
passes all sampled original GB10 BF16 endpoints. In the unperturbed host mode,
selected endpoints fall from 12,270 to 9,708 (20.88%). QKV selection falls from
8,219 to 6,962 (15.29%), including 896 unsupported dots that both routes replay
in full. The QKV comparison is against original host arithmetic, without an
external output reference. All three modes and all supported intermediate
boundaries pass. The geometry extends 7,169 captured input rows by repeating
the first 1,023; repeated rows do not provide independent model-token evidence.

The first QKV diagnostic stopped at the domain guard because the harness had
omitted whole-row fallback. The revised harness retains that fallback and
reports its work explicitly; it does not extend the admitted exponent domain.

The native comparison below includes metadata preparation, bound arithmetic
and complete replay. No provider, runtime option, qualified model result or
package changes.

Evidence: [host record](../benchmarks/correctness/signed-loss-bound-host-20260918.json),
12,635 bytes, SHA256
`5da40ee7382e06700362f3581681c140d652d4133b7de80db33c2cbd02ff2b85`,
pins source `af735be`, all commands, input hashes and the revised harness result.

## Complete native comparison, 2026-09-18

Source `0d20e48` constructs actual C64 sign/nonzero masks inside a shared
producer. Its 64x64 and 64x32 schedules preserve every raw center, directional
endpoint and selected cell. All 88 generated configurations pass, including
unsupported operands, independent two-byte skews, exponent limits, tails,
sign-stable blocks and sign crossing. Every generated cell has an independent
CPU comparison. Each captured warmup and measured attempt checks all original
GPU endpoints, external GB10 BF16 projection outputs, 256 independent CPU
dots, complete candidate permutations, prepared words and storage guards.

| q8192 operator shape | Symmetric shared ms | Directed 64x64 ms | Directed 64x32 ms | Selected, symmetric / directed |
| --- | ---: | ---: | ---: | ---: |
| OUT | 85.8026 | 153.7327 | 96.0616 | 3124922 / 2487560 |
| QKV | 138.5076 | 318.2591 | 172.4419 | 8285915 / 6957701 |

Medians cover three rotated complete samples after one warmup. Preparation,
mask construction, production, compaction, host count and all selected original
K16 replay are included. The smaller tile eliminates the original 34 VGPR
spills and 140 private bytes, using 169 VGPRs and 14592 LDS bytes; these are
static compiler resources, not measured occupancy. Its OUT/QKV producer and
selection medians still rise from 26.706/54.6145 to 49.0112/103.2104 ms.
Every directional owner sample is slower than every shared control sample.
Keep both schedules isolated despite the 20.40%/16.03% candidate reductions.

The shapes extend 7169 captured rows by repeating 1023 rows. OUT has a complete
8192-row external operator reference; QKV repeats corresponding reference rows.
This is component evidence, without a real 8192-token model request or a proof
of the native error premise. The qualified model TTFT remains 23353.80795 ms.
The earlier spilling prototype and a subsequent C++ array-declaration build
failure are preserved; that failed build executed no numerical GPU work.

[Native source, bounded commands and results](../benchmarks/correctness/signed-loss-projection-native-components-20260918.json):
524761 bytes, SHA256
`88acccca492dc46272de60954bab387fd71ae140251a32c05007cc05b15ce6c9`.

## Deferred envelopes and reusable masks

Source `ec96fbd` prepares each operand's 16-byte C64 sign/nonzero mask once.
The producer accumulates nonnegative error statistics and finalizes their
coupled conservative bound once, removing per-block inherited-error, exponent
and outward-rounding calculations. The original native coefficient remains
2^-19. The bound derivation and exact rational growth check cover at most128
C64 blocks; unsupported widths and operands require original replay.

Host ASan/UBSan checks701616 boundaries at nine widths16..8192, three synthetic
producer modes,65536 BF16 encodings and65536 support masks. All88 native
configurations pass. Every deferred bound contains the original directional
bound, and both schedules preserve all raw centers, bounds and selections.
Complete captured attempts also pass the original arithmetic, GB10 operator
references, independent CPU dots, prepared masks, candidate and storage checks.

| q8192 operator shape | Symmetric shared ms | Deferred64x64 ms | Deferred64x32 ms | Selected, symmetric / deferred |
| --- | ---: | ---: | ---: | ---: |
| OUT | 85.8982 | 91.1399 | 88.0962 | 3124922 / 2999076 |
| QKV | 139.7932 | 177.6558 | 183.6243 | 8285915 / 8011330 |

All preparation, selection and original replay are included in the three
rotated complete samples. Each candidate sample remains slower than every
matching shared control. Extra masks occupy10485760/8388608 bytes. Compiler
metadata reports256/169 VGPRs and24 private bytes for the two candidates,
with no reported VGPR/SGPR spills; this does not establish their runtime cost.
The looser finalization retains only4.03%/3.31% candidate reduction. Keep both
routes isolated and pursue another provider surface. Model TTFT, package and
release status remain unchanged; these captures are repeated-row operators.

[Deferred source, derivation, bounded commands and complete results](../benchmarks/correctness/deferred-loss-projection-native-components-20260918.json):
295333 bytes, SHA256
`7f49f94db543fa474c3912f25d4eeae48cb9981936493accd77159d6e2153108`.
