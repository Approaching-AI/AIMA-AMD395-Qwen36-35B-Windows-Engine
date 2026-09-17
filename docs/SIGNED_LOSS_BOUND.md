# Directed canonical loss experiment

This isolated host experiment keeps the coarse producer's conditional native
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

Keep this as a host building block. Selection falls modestly before counting
metadata preparation, extra bound arithmetic or replay scheduling. It does not
establish a native component gain or a seconds-scale product route. No provider,
runtime option, qualified model result or package changes.
