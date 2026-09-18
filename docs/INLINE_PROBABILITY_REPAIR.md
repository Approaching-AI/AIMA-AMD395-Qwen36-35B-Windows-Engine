# Inline probability repair experiment

This isolated component replaces the joint-context scheduler's multiple
score-selection, probability initialization and histogram passes with one
wave per query/head. Each original K32 tile repairs all possible prefix
maxima and ambiguous BF16 probabilities using the unchanged prepared K16
dot, with a full original-dot restart for unsupported inputs. The wave then
propagates original-order lower, center and upper denominator recurrences.
It stores probabilities and alphas for native PV and exact selected PV replay.

The score bound and complete Cartesian context certificate are unchanged.
An unresolved context row triggers one complete remaining-score replay using
the original 2x2 microtile. Its denominator is then recomputed, and any
remaining ambiguous numerator receives original PV replay. A row must pass
the original certificate or its exact-point fallback. No reference values
select work, no coefficient is reduced, and the runtime does not dispatch
this experiment.

This removes global lower/upper probability arrays, per-score queues and
priority arrays, repeated interval scans and the histogram. It does add
divergent scalar QK work inside the first probability traversal. Native
measurements must determine whether the removed passes outweigh this cost
and the full-row fallback fraction.

The native fixture first checks the initial probability pass independently,
before native PV or complete score fallback can hide a repair error. It then
checks the complete attention result, every causal score enclosure, BF16
probability and alpha, denominator and full original PV numerator enclosure,
all context outputs, guards, unused tails and input immutability. Eight
boundary shapes and five data families precede the captured comparison.

The q8192 component repeats the first 1023 rows of real q7169 layer-3 Q/K/V.
The original 29364224 GB10 context cells are comparison-only; repeated rows
use original arithmetic without a new external reference. One warmup and
three rotated completed-host samples include the entire candidate pipeline
and all fallback. Preparation, allocation, transfers and observers are outside
timing. Native validation is pending; no model, TTFT or release claim follows.
