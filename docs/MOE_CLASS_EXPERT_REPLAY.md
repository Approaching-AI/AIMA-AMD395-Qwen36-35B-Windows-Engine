# Classified MoE expert replay experiment

`QRT_QWEN36_MOE_CLASS_EXPERT_ORDER_REPLAY=1` is a default-off q8192 experiment.
It requires the existing compact, four-lane staged-half routed replay and
expert ordering. It preserves the current candidate predicates, norms, BF16
endpoints, ordered K16 carries, and shared/routed stream ownership.

The existing lossless Row36 preparation also reduces a whole-row class in its
256-thread CTA. A row is either eligible for all-nonzero transformed replay,
eligible with possible zeros, or requires the original staged fallback. This
pass supports widths 512, 1024, 2048, and 4096; each row contains whole wave32s
and stays inside one CTA. New class arrays are separate from the registered
weight norm/validation cache and refresh with every prepared operand surface.

The original collected list is then permuted into 768 buckets: arithmetic
class first, expert second. Three replay kernels read disjoint device ranges;
they do not filter three copies of the original list. The fast variants use
the existing exact transformed arithmetic, including its original ordered
carry. The fallback retains the original staged dot. Gate, up, ordinary down,
and the retained compact down owner all use these ranges. Up finalization
still follows completion of every replay class on the routed stream.

At the maximum 4,194,304-candidate window, the ordered workspace is 16,786,436
bytes, 6,144 bytes above expert-only ordering. Separate activation and weight
class arrays add 2,359,296 bytes. No host count read or candidate removal is
introduced. Other prompt lengths and the option's default retain the existing
runtime route.

The local harness checks actual selectors, all three replay phases, sparse
and dense windows, debug outputs, allocation/release faults, and every
submission failure. Native fixtures compare the fused encoding and row class
against original BF16 inputs, each raw K16 carry against an independent
original primitive, all 768 permutation buckets, exact-once range consumption,
unchanged inputs, and buffer guards. These are component checks. The experiment
requires a same-DLL q8192 comparison with all 512 captured GB10 continuation
tokens and the first-token logit before any performance result can be retained.

Status: implementation and qualification in progress; no product result or
release acceptance is claimed.
